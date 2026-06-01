use std::{
    env,
    ffi::CString,
    os::raw::{c_char, c_int},
    path::PathBuf,
};

unsafe extern "C" {
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

    let engine = match CString::new(args.engine.to_string_lossy().as_bytes()) {
        Ok(path) => path,
        Err(_) => {
            eprintln!("engine path contains an interior NUL byte");
            std::process::exit(2);
        }
    };

    let mut message = vec![0i8; 16 * 1024];
    let status = unsafe {
        trt_probe_engine(
            engine.as_ptr(),
            args.frames,
            args.channels,
            message.as_mut_ptr(),
            message.len(),
        )
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

    let engine = engine.ok_or("missing --engine")?;
    if frames <= 0 {
        return Err("--frames must be positive".to_string());
    }
    if channels <= 0 {
        return Err("--channels must be positive".to_string());
    }

    Ok(Args {
        engine,
        frames,
        channels,
    })
}

fn print_usage() {
    eprintln!(
        "usage: cargo run --manifest-path tools/tensorrt_probe/Cargo.toml -- --engine <file.engine> [--frames 40] [--channels 768]"
    );
}
