use std::{
    env,
    ffi::CString,
    os::raw::{c_char, c_int},
    path::PathBuf,
};

unsafe extern "C" {
    fn trt_probe_build(
        onnx_path: *const c_char,
        engine_path: *const c_char,
        profile_shapes: *const c_char,
        message: *mut c_char,
        message_len: usize,
    ) -> c_int;
    fn trt_probe_engine(
        engine_path: *const c_char,
        frames: c_int,
        channels: c_int,
        message: *mut c_char,
        message_len: usize,
    ) -> c_int;
}

#[derive(Debug)]
struct Args {
    mode: Mode,
}

#[derive(Debug)]
enum Mode {
    Build {
        onnx: PathBuf,
        save_engine: PathBuf,
        profile: String,
    },
    Run {
        engine: PathBuf,
        frames: i32,
        channels: i32,
    },
}

fn cstring_path(path: &PathBuf, label: &str) -> CString {
    match CString::new(path.to_string_lossy().as_bytes()) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("{label} contains an interior NUL byte");
            std::process::exit(2);
        }
    }
}

fn cstring_text(value: &str, label: &str) -> CString {
    match CString::new(value) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("{label} contains an interior NUL byte");
            std::process::exit(2);
        }
    }
}

#[derive(Debug)]
struct RunArgs {
    engine: PathBuf,
    frames: i32,
    channels: i32,
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(err) => {
            eprintln!("{err}");
            print_usage();
            std::process::exit(2);
        }
    };

    let mut message = vec![0i8; 16 * 1024];
    let status = match args.mode {
        Mode::Build {
            onnx,
            save_engine,
            profile,
        } => {
            let onnx = cstring_path(&onnx, "onnx path");
            let save_engine = cstring_path(&save_engine, "engine path");
            let profile = cstring_text(&profile, "profile");
            unsafe {
                trt_probe_build(
                    onnx.as_ptr(),
                    save_engine.as_ptr(),
                    profile.as_ptr(),
                    message.as_mut_ptr(),
                    message.len(),
                )
            }
        }
        Mode::Run {
            engine,
            frames,
            channels,
        } => {
            let engine = cstring_path(&engine, "engine path");
            unsafe {
                trt_probe_engine(
                    engine.as_ptr(),
                    frames,
                    channels,
                    message.as_mut_ptr(),
                    message.len(),
                )
            }
        }
    };

    let nul = message
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(message.len());
    let bytes = message[..nul].iter().map(|&b| b as u8).collect::<Vec<_>>();
    println!("{}", String::from_utf8_lossy(&bytes));

    if status != 0 {
        std::process::exit(status);
    }
}

fn parse_args() -> Result<Args, String> {
    let mut engine = None;
    let mut onnx = None;
    let mut save_engine = None;
    let mut profile = None;
    let mut frames = 40;
    let mut channels = 768;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--engine" => {
                engine = Some(PathBuf::from(
                    iter.next().ok_or("--engine requires a path")?,
                ));
            }
            "--onnx" => {
                onnx = Some(PathBuf::from(iter.next().ok_or("--onnx requires a path")?));
            }
            "--save-engine" => {
                save_engine = Some(PathBuf::from(
                    iter.next().ok_or("--save-engine requires a path")?,
                ));
            }
            "--profile" => {
                profile = Some(iter.next().ok_or("--profile requires a value")?);
            }
            "--frames" => {
                frames = iter
                    .next()
                    .ok_or("--frames requires a value")?
                    .parse()
                    .map_err(|_| "--frames must be an integer")?;
            }
            "--channels" => {
                channels = iter
                    .next()
                    .ok_or("--channels requires a value")?
                    .parse()
                    .map_err(|_| "--channels must be an integer")?;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }

    let mode = if let Some(onnx) = onnx {
        Mode::Build {
            onnx,
            save_engine: save_engine.ok_or("--onnx requires --save-engine")?,
            profile: profile.ok_or("--onnx requires --profile")?,
        }
    } else {
        if save_engine.is_some() || profile.is_some() {
            return Err("--save-engine and --profile require --onnx".to_string());
        }
        let RunArgs {
            engine,
            frames,
            channels,
        } = RunArgs {
            engine: engine.ok_or("missing --engine or --onnx")?,
            frames,
            channels,
        };
        if frames <= 0 {
            return Err("--frames must be positive".to_string());
        }
        if channels <= 0 {
            return Err("--channels must be positive".to_string());
        }
        Mode::Run {
            engine,
            frames,
            channels,
        }
    };

    Ok(Args { mode })
}

fn print_usage() {
    eprintln!(
        "usage:\n  vc-tensorrt-builder --onnx <model.onnx> --save-engine <file.engine> --profile <name:1x...,...>\n  vc-tensorrt-builder --engine <file.engine> [--frames 40] [--channels 768]"
    );
}
