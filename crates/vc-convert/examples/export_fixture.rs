//! Offline conversion helper for reference validation; does not run inference.
use anyhow::{Context, Result};
fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .context("usage: export_fixture input.pth output.onnx")?;
    let output = args.next().context("missing output.onnx")?;
    let bytes = std::fs::read(input)?;
    let conversion =
        vc_convert::pth_to_onnx(&bytes, &vc_convert::ConvertOptions::default(), &mut |_| {})?;
    std::fs::write(output, conversion.onnx_bytes)?;
    Ok(())
}
