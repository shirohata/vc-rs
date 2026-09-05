//! Python pickle parser for PyTorch checkpoints.
//!
//! Ported from rvc-onnx-web (https://github.com/visgotti/rvc-onnx-web),
//! MIT License — `src/pickle.ts`.
//!
//! Deliberate trim from the TS source: only the *binary* opcodes (protocols
//! 1–5) are implemented. PyTorch ≥1.6 writes `data.pkl` with protocol 2, so
//! the text-based protocol-0 opcodes (INT/LONG/FLOAT/STRING/UNICODE/PERSID/
//! GET/PUT/INST/OBJ) never appear in a supported checkpoint; hitting one
//! fails with a clear error instead. GLOBAL is kept even though it reads
//! text lines — older torch emits it inside protocol-2 streams.
//!
//! Values use `Rc`/`RefCell` because pickle aliases via the memo: a container
//! is memoized *before* it is filled (EMPTY_DICT → MEMOIZE → … → SETITEMS),
//! so every reference must observe later in-place mutation.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::{anyhow, bail, Context, Result};

use crate::torch;

/// A `module.name` reference pushed by GLOBAL / STACK_GLOBAL.
#[derive(Debug)]
pub(crate) struct Global {
    pub module: String,
    pub name: String,
}

impl Global {
    pub fn full_name(&self) -> String {
        format!("{}.{}", self.module, self.name)
    }
}

/// Placeholder for Python objects we don't specialize (TS `PythonObject`).
/// Fields are populated for `Debug`-formatted error context only.
#[derive(Debug)]
#[allow(dead_code, reason = "read only through the Debug impl in error paths")]
pub(crate) struct PyObject {
    pub module: String,
    pub name: String,
    pub args: Vec<Value>,
    pub state: Option<Value>,
}

/// Storage produced by `persistent_load`; shape attaches later via
/// `_rebuild_tensor_v2`. Data is already widened per `torch::widen_storage`.
#[derive(Debug)]
pub(crate) struct Storage {
    pub data: crate::tensor::TensorData,
}

#[derive(Clone, Debug)]
pub(crate) enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Bytes(
        #[allow(
            dead_code,
            reason = "no torch checkpoint field consumes bytes; kept for pickle completeness"
        )]
        Rc<[u8]>,
    ),
    List(Rc<RefCell<Vec<Value>>>),
    Tuple(Rc<[Value]>),
    /// Insertion-ordered, like Python dict.
    Dict(Rc<RefCell<Vec<(Value, Value)>>>),
    Global(Rc<Global>),
    Object(Rc<RefCell<PyObject>>),
    Storage(Rc<Storage>),
    Tensor(Rc<crate::tensor::Tensor>),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::None => "None",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::Str(_) => "str",
            Value::Bytes(_) => "bytes",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Dict(_) => "dict",
            Value::Global(_) => "global",
            Value::Object(_) => "object",
            Value::Storage(_) => "storage",
            Value::Tensor(_) => "tensor",
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }

    /// Numeric coercion for config fields that may pickle as int or float.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::Float(f) => Some(*f),
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            _ => None,
        }
    }

    /// Dict lookup by string key (checkpoint dicts always key on str).
    pub fn dict_get(&self, key: &str) -> Option<Value> {
        match self {
            Value::Dict(entries) => entries
                .borrow()
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v.clone()),
            _ => None,
        }
    }
}

/// Resolves a persistent-id storage key (e.g. `"0"`) to raw bytes. Returns
/// owned bytes: the payload is immediately re-encoded into a typed vector by
/// `torch::widen_storage`, so borrowing wouldn't save the copy that matters.
pub(crate) type StorageResolver<'a> = dyn Fn(&str) -> Result<Vec<u8>> + 'a;

pub(crate) struct Unpickler<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    marks: Vec<usize>,
    memo: Vec<Value>,
    storage_resolver: &'a StorageResolver<'a>,
}

impl<'a> Unpickler<'a> {
    pub fn new(data: &'a [u8], storage_resolver: &'a StorageResolver<'a>) -> Self {
        Unpickler {
            data,
            pos: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            memo: Vec::new(),
            storage_resolver,
        }
    }

    pub fn load(&mut self) -> Result<Value> {
        while self.pos < self.data.len() {
            let opcode = self.data[self.pos];
            self.pos += 1;
            match opcode {
                // Protocol markers
                0x80 => {
                    // PROTO
                    self.pos += 1;
                }
                0x95 => {
                    // FRAME: skip length, the whole stream is in memory
                    self.pos += 8;
                }
                0x2e => {
                    // STOP
                    return self.pop();
                }

                // Stack manipulation
                0x28 => self.marks.push(self.stack.len()), // MARK
                0x30 => {
                    // POP
                    self.pop()?;
                }
                0x31 => {
                    // POP_MARK
                    self.pop_mark()?;
                }
                0x32 => {
                    // DUP
                    let top = self.top()?.clone();
                    self.stack.push(top);
                }

                // Singletons
                0x4e => self.stack.push(Value::None), // NONE
                0x88 => self.stack.push(Value::Bool(true)), // NEWTRUE
                0x89 => self.stack.push(Value::Bool(false)), // NEWFALSE

                // Integers
                0x4a => {
                    // BININT
                    let v = self.read_i32()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                0x4b => {
                    // BININT1
                    let v = self.read_u8()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                0x4d => {
                    // BININT2
                    let v = self.read_u16()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                0x8a => {
                    // LONG1
                    let n = usize::from(self.read_u8()?);
                    let v = self.read_long_bytes(n)?;
                    self.stack.push(Value::Int(v));
                }
                0x8b => {
                    // LONG4
                    let n = self.read_i32()?;
                    let n = usize::try_from(n).map_err(|_| anyhow!("negative LONG4 length"))?;
                    let v = self.read_long_bytes(n)?;
                    self.stack.push(Value::Int(v));
                }

                // Floats
                0x47 => {
                    // BINFLOAT — big-endian!
                    let bytes: [u8; 8] = self.read_slice(8)?.try_into().expect("len checked");
                    self.stack.push(Value::Float(f64::from_be_bytes(bytes)));
                }

                // Strings
                0x55 => {
                    // SHORT_BINSTRING (latin-1)
                    let len = usize::from(self.read_u8()?);
                    let s = self.read_latin1(len)?;
                    self.stack.push(Value::Str(s.into()));
                }
                0x54 => {
                    // BINSTRING (latin-1)
                    let len = self.read_i32()?;
                    let len =
                        usize::try_from(len).map_err(|_| anyhow!("negative BINSTRING length"))?;
                    let s = self.read_latin1(len)?;
                    self.stack.push(Value::Str(s.into()));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let len = usize::from(self.read_u8()?);
                    let s = self.read_utf8(len)?;
                    self.stack.push(Value::Str(s.into()));
                }
                0x58 => {
                    // BINUNICODE
                    let len = self.read_u32()? as usize;
                    let s = self.read_utf8(len)?;
                    self.stack.push(Value::Str(s.into()));
                }
                0x8d => {
                    // BINUNICODE8
                    let len = self.read_u64_len()?;
                    let s = self.read_utf8(len)?;
                    self.stack.push(Value::Str(s.into()));
                }

                // Bytes
                0x43 => {
                    // SHORT_BINBYTES
                    let len = usize::from(self.read_u8()?);
                    let b = self.read_slice(len)?;
                    self.stack.push(Value::Bytes(b.into()));
                }
                0x42 => {
                    // BINBYTES
                    let len = self.read_u32()? as usize;
                    let b = self.read_slice(len)?;
                    self.stack.push(Value::Bytes(b.into()));
                }
                0x8e => {
                    // BINBYTES8
                    let len = self.read_u64_len()?;
                    let b = self.read_slice(len)?;
                    self.stack.push(Value::Bytes(b.into()));
                }
                0x96 => {
                    // BYTEARRAY8 — treated as immutable bytes like the TS port
                    let len = self.read_u64_len()?;
                    let b = self.read_slice(len)?;
                    self.stack.push(Value::Bytes(b.into()));
                }

                // Collections — empty
                0x5d => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))), // EMPTY_LIST
                0x29 => self.stack.push(Value::Tuple(Rc::from(Vec::new()))), // EMPTY_TUPLE
                0x7d => self
                    .stack
                    .push(Value::Dict(Rc::new(RefCell::new(Vec::new())))), // EMPTY_DICT

                // Collections — from mark
                0x6c => {
                    // LIST
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(Rc::new(RefCell::new(items))));
                }
                0x74 => {
                    // TUPLE
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items.into()));
                }
                0x85 => {
                    // TUPLE1
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a].into()));
                }
                0x86 => {
                    // TUPLE2
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b].into()));
                }
                0x87 => {
                    // TUPLE3
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c].into()));
                }
                0x64 => {
                    // DICT
                    let items = self.pop_mark()?;
                    let mut entries = Vec::with_capacity(items.len() / 2);
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        entries.push((k, v));
                    }
                    self.stack.push(Value::Dict(Rc::new(RefCell::new(entries))));
                }

                // Collection mutation
                0x73 => {
                    // SETITEM
                    let value = self.pop()?;
                    let key = self.pop()?;
                    match self.top()? {
                        Value::Dict(d) => d.borrow_mut().push((key, value)),
                        other => bail!("SETITEM on non-dict {}", other.type_name()),
                    }
                }
                0x75 => {
                    // SETITEMS
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::Dict(d) => {
                            let mut d = d.borrow_mut();
                            let mut it = items.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                d.push((k, v));
                            }
                        }
                        other => bail!("SETITEMS on non-dict {}", other.type_name()),
                    }
                }
                0x61 => {
                    // APPEND
                    let item = self.pop()?;
                    match self.top()? {
                        Value::List(l) => l.borrow_mut().push(item),
                        other => bail!("APPEND on non-list {}", other.type_name()),
                    }
                }
                0x65 => {
                    // APPENDS
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Value::List(l) => l.borrow_mut().extend(items),
                        other => bail!("APPENDS on non-list {}", other.type_name()),
                    }
                }

                // Object construction
                0x63 => {
                    // GLOBAL (text lines; still emitted inside protocol-2 streams)
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack
                        .push(Value::Global(Rc::new(Global { module, name })));
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Some(name), Some(module)) = (name.as_str(), module.as_str()) else {
                        bail!("STACK_GLOBAL expects two strings");
                    };
                    self.stack.push(Value::Global(Rc::new(Global {
                        module: module.to_owned(),
                        name: name.to_owned(),
                    })));
                }
                0x52 => {
                    // REDUCE
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let result = torch::reduce(&callable, &args)?;
                    self.stack.push(result);
                }
                0x81 => {
                    // NEWOBJ
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let result = torch::newobj(&cls, args)?;
                    self.stack.push(result);
                }
                0x92 => {
                    // NEWOBJ_EX — kwargs are dropped like the TS port drops them
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let result = torch::newobj(&cls, args)?;
                    self.stack.push(result);
                }
                0x62 => {
                    // BUILD
                    let state = self.pop()?;
                    match self.top()? {
                        Value::Object(obj) => obj.borrow_mut().state = Some(state),
                        // Storages receive a BUILD with device/size state we
                        // don't need; dicts can receive __setstate__ dicts.
                        Value::Storage(_) | Value::Tensor(_) => {}
                        Value::Dict(d) => {
                            if let Value::Dict(state) = state {
                                d.borrow_mut().extend(state.borrow().iter().cloned());
                            }
                        }
                        other => bail!("BUILD on unsupported value {}", other.type_name()),
                    }
                }

                // Memo
                0x71 => {
                    // BINPUT
                    let idx = usize::from(self.read_u8()?);
                    self.memo_put(idx)?;
                }
                0x72 => {
                    // LONG_BINPUT
                    let idx = self.read_u32()? as usize;
                    self.memo_put(idx)?;
                }
                0x94 => {
                    // MEMOIZE
                    let idx = self.memo.len();
                    self.memo_put(idx)?;
                }
                0x68 => {
                    // BINGET
                    let idx = usize::from(self.read_u8()?);
                    self.memo_get(idx)?;
                }
                0x6a => {
                    // LONG_BINGET
                    let idx = self.read_u32()? as usize;
                    self.memo_get(idx)?;
                }

                // Persistent ID (tensor storage)
                0x51 => {
                    // BINPERSID
                    let pid = self.pop()?;
                    let storage = torch::persistent_load(&pid, self.storage_resolver)?;
                    self.stack.push(storage);
                }

                other => bail!(
                    "unsupported pickle opcode 0x{other:02x} at position {} \
                     (text protocol 0 and out-of-band buffers are not supported)",
                    self.pos - 1
                ),
            }
        }
        bail!("unexpected end of pickle data (missing STOP opcode)")
    }

    // Stack / memo helpers

    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle stack underflow")
    }

    fn top(&mut self) -> Result<&Value> {
        self.stack.last().context("pickle stack underflow")
    }

    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mark = self.marks.pop().context("pickle mark stack underflow")?;
        Ok(self.stack.split_off(mark))
    }

    fn memo_put(&mut self, idx: usize) -> Result<()> {
        let top = self.top()?.clone();
        if idx >= self.memo.len() {
            self.memo.resize(idx + 1, Value::None);
        }
        self.memo[idx] = top;
        Ok(())
    }

    fn memo_get(&mut self, idx: usize) -> Result<()> {
        let value = self
            .memo
            .get(idx)
            .cloned()
            .ok_or_else(|| anyhow!("pickle memo index {idx} not set"))?;
        self.stack.push(value);
        Ok(())
    }

    // Binary reading helpers

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8]> {
        let slice = self
            .data
            .get(
                self.pos
                    ..self
                        .pos
                        .checked_add(len)
                        .context("pickle length overflow")?,
            )
            .ok_or_else(|| anyhow!("pickle data truncated at position {}", self.pos))?;
        self.pos += len;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_slice(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self.read_slice(2)?.try_into().expect("len checked");
        Ok(u16::from_le_bytes(bytes))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let bytes: [u8; 4] = self.read_slice(4)?.try_into().expect("len checked");
        Ok(i32::from_le_bytes(bytes))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self.read_slice(4)?.try_into().expect("len checked");
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_u64_len(&mut self) -> Result<usize> {
        let bytes: [u8; 8] = self.read_slice(8)?.try_into().expect("len checked");
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| anyhow!("pickle length too large"))
    }

    /// Little-endian two's-complement long (LONG1/LONG4 payload).
    fn read_long_bytes(&mut self, n: usize) -> Result<i64> {
        if n == 0 {
            return Ok(0);
        }
        if n > 8 {
            bail!("pickle LONG wider than 64 bits ({n} bytes) is not supported");
        }
        let bytes = self.read_slice(n)?;
        let mut buf = if bytes[n - 1] & 0x80 != 0 {
            [0xFF; 8] // sign-extend
        } else {
            [0; 8]
        };
        buf[..n].copy_from_slice(bytes);
        Ok(i64::from_le_bytes(buf))
    }

    fn read_utf8(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_slice(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_latin1(&mut self, len: usize) -> Result<String> {
        let bytes = self.read_slice(len)?;
        Ok(bytes.iter().map(|&b| b as char).collect())
    }

    fn read_line(&mut self) -> Result<String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != 0x0a {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            bail!("pickle data truncated inside text line");
        }
        let line = String::from_utf8_lossy(&self.data[start..self.pos]).into_owned();
        self.pos += 1; // skip the newline
        Ok(line)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;

    fn load(bytes: &[u8]) -> Result<Value> {
        let resolver = |key: &str| Err(anyhow!("no storage in test: {key}"));
        Unpickler::new(bytes, &resolver).load()
    }

    #[test]
    fn parses_scalars_and_collections() {
        // pickle.dumps({"a": 1, "b": [True, None, 2.5]}, protocol=2)
        let mut stream = vec![0x80, 0x02]; // PROTO 2
        stream.push(0x7d); // EMPTY_DICT
        stream.push(0x71);
        stream.push(0); // BINPUT 0
        stream.push(0x28); // MARK
        stream.extend_from_slice(&[0x58, 1, 0, 0, 0]); // BINUNICODE "a"
        stream.push(b'a');
        stream.extend_from_slice(&[0x4b, 1]); // BININT1 1
        stream.extend_from_slice(&[0x58, 1, 0, 0, 0]);
        stream.push(b'b');
        stream.push(0x5d); // EMPTY_LIST
        stream.push(0x28); // MARK
        stream.push(0x88); // NEWTRUE
        stream.push(0x4e); // NONE
        stream.push(0x47); // BINFLOAT (big-endian)
        stream.extend_from_slice(&2.5f64.to_be_bytes());
        stream.push(0x65); // APPENDS
        stream.push(0x75); // SETITEMS
        stream.push(0x2e); // STOP

        let value = load(&stream).unwrap();
        assert_eq!(value.dict_get("a").unwrap().as_int(), Some(1));
        let b = value.dict_get("b").unwrap();
        let Value::List(items) = b else {
            panic!("expected list")
        };
        let items = items.borrow();
        assert!(matches!(items[0], Value::Bool(true)));
        assert!(matches!(items[1], Value::None));
        assert!(matches!(items[2], Value::Float(f) if (f - 2.5).abs() < 1e-12));
    }

    #[test]
    fn memo_aliasing_observes_mutation() {
        // A dict memoized empty, referenced via BINGET, then filled: the
        // memo reference must see the SETITEMS mutation.
        let mut stream = vec![0x80, 0x02];
        stream.push(0x7d); // EMPTY_DICT
        stream.push(0x71);
        stream.push(0); // BINPUT 0
        stream.push(0x28); // MARK
        stream.extend_from_slice(&[0x8c, 1, b'k']); // SHORT_BINUNICODE "k"
        stream.push(0x68);
        stream.push(0); // BINGET 0 (self-reference as the value)
        stream.push(0x75); // SETITEMS
        stream.push(0x2e); // STOP

        let value = load(&stream).unwrap();
        let inner = value.dict_get("k").unwrap();
        // The self-referencing value must contain the "k" entry too.
        assert!(inner.dict_get("k").is_some());
    }

    #[test]
    fn negative_long1_round_trips() {
        // pickle.dumps(-2, protocol=2) → LONG1 with 0xFE
        let stream = [0x80, 0x02, 0x8a, 0x01, 0xfe, 0x2e];
        assert_eq!(load(&stream).unwrap().as_int(), Some(-2));
        // LONG1 with zero bytes is 0
        let stream = [0x80, 0x02, 0x8a, 0x00, 0x2e];
        assert_eq!(load(&stream).unwrap().as_int(), Some(0));
    }

    #[test]
    fn rejects_protocol_0_text_opcodes() {
        // 'I1\n.' — protocol 0 INT
        let err = load(b"I1\n.").unwrap_err().to_string();
        assert!(err.contains("unsupported pickle opcode 0x49"), "{err}");
    }

    #[test]
    fn torch_shaped_stream_builds_tensor() {
        // Mimics the shape of a torch data.pkl entry:
        //   REDUCE(torch._utils._rebuild_tensor_v2,
        //          (BINPERSID(("storage", FloatStorage, "0", "cpu", 6)),
        //           0, (2, 3), (3, 1), False, OrderedDict()))
        let mut s = vec![0x80, 0x02];
        // torch._utils._rebuild_tensor_v2 via STACK_GLOBAL
        push_str(&mut s, "torch._utils");
        push_str(&mut s, "_rebuild_tensor_v2");
        s.push(0x93); // STACK_GLOBAL
        s.push(0x28); // MARK (outer args tuple built via TUPLE at the end)
                      // persistent id tuple ("storage", FloatStorage, "0", "cpu", 6)
        s.push(0x28); // MARK
        push_str(&mut s, "storage");
        push_str(&mut s, "torch");
        push_str(&mut s, "FloatStorage");
        s.push(0x93); // STACK_GLOBAL
        push_str(&mut s, "0");
        push_str(&mut s, "cpu");
        s.extend_from_slice(&[0x4b, 6]); // BININT1 6
        s.push(0x74); // TUPLE
        s.push(0x51); // BINPERSID
        s.extend_from_slice(&[0x4b, 0]); // offset 0
        s.extend_from_slice(&[0x4b, 2, 0x4b, 3, 0x86]); // (2, 3) TUPLE2
        s.extend_from_slice(&[0x4b, 3, 0x4b, 1, 0x86]); // (3, 1) TUPLE2
        s.push(0x89); // NEWFALSE (requires_grad)
                      // collections.OrderedDict backward hooks
        push_str(&mut s, "collections");
        push_str(&mut s, "OrderedDict");
        s.push(0x93);
        s.push(0x29); // EMPTY_TUPLE
        s.push(0x52); // REDUCE → empty dict
        s.push(0x74); // TUPLE (args)
        s.push(0x52); // REDUCE → tensor
        s.push(0x2e); // STOP

        let payload: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let resolver = |key: &str| {
            assert_eq!(key, "0");
            Ok(payload.clone())
        };
        let value = Unpickler::new(&s, &resolver).load().unwrap();
        let Value::Tensor(tensor) = value else {
            panic!("expected tensor, got {}", value.type_name());
        };
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(
            tensor.data,
            TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
    }

    fn push_str(stream: &mut Vec<u8>, s: &str) {
        assert!(s.len() < 256);
        stream.push(0x8c); // SHORT_BINUNICODE
        stream.push(s.len() as u8);
        stream.extend_from_slice(s.as_bytes());
    }
}
