# Optional helper script.
# This script downloads third-party reference ONNX models from:
# https://huggingface.co/wok000/weights_gpl
#
# The downloaded model files are NOT part of vc-rs and are NOT covered by this
# repository's MIT license. The upstream model repository is marked GPL-3.0.
# Review and comply with the upstream license before using, modifying, or
# redistributing the downloaded files.
#
# vc-rs does not redistribute pretrained model files. This script only downloads
# them directly from the upstream host at the user's request.


$ErrorActionPreference = "Stop"

$assetsDir = Join-Path $PSScriptRoot "assets"
New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null

$downloads = @(
    @{
        Name = "ContentVec ONNX"
        Url  = "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx"
        Path = Join-Path $assetsDir "content_vec_500.onnx"
    },
    @{
        Name = "RMVPE ONNX"
        Url  = "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"
        Path = Join-Path $assetsDir "rmvpe.onnx"
    }
)

foreach ($item in $downloads) {
    if (Test-Path $item.Path) {
        Write-Host "[skip] $($item.Name) already exists: $($item.Path)"
        continue
    }

    Write-Host "[download] $($item.Name)"
    Write-Host "  from: $($item.Url)"
    Write-Host "  to:   $($item.Path)"

    $tmpPath = "$($item.Path).download"

    try {
        Invoke-WebRequest `
            -Uri $item.Url `
            -OutFile $tmpPath `
            -UseBasicParsing

        Move-Item -Force $tmpPath $item.Path

        $sizeMB = [math]::Round((Get-Item $item.Path).Length / 1MB, 2)
        Write-Host "[done] $($item.Name) ($sizeMB MB)"
    }
    catch {
        if (Test-Path $tmpPath) {
            Remove-Item -Force $tmpPath
        }
        throw
    }
}

Write-Host ""
Write-Host "All requested models are ready."
Write-Host "ContentVec: $assetsDir\content_vec_500.onnx"
Write-Host "RMVPE:      $assetsDir\rmvpe.onnx"