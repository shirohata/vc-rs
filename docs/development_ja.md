# vc-rs 開発者向けガイド（ソースからビルド）

このドキュメントは **ソースからビルドする開発者向け** です。CLI の使い方や各
オプションの説明はルート [README](../README.md) にあります。

GPU ビルド/実行は **CUDA 13 / TensorRT 11** ラインを前提とします（CUDA 12 /
TensorRT 10 のサポートは終了しました）。

## 必要環境

- Rust stable と `cargo`
- Windows: `x86_64-pc-windows-msvc` ツールチェーン
- Windows: Visual Studio Build Tools（C++ workload）
  ― `cc` クレートがネイティブ TensorRT シムをコンパイルするために必要です。

CPU 実行だけで試す場合、CUDA / cuDNN / TensorRT は不要です。

## ビルド環境の自動セットアップ（Windows）

winget で入る範囲の導入、セッションごとの環境有効化、疎通確認は `scripts/` に
まとめています。詳細は [`scripts/README.md`](../scripts/README.md) を参照してください。

```powershell
pwsh -File scripts/bootstrap.ps1   # 初回のみ: Rustup / Git / VS BuildTools
# → CUDA 13.2 / cuDNN 9.x / TensorRT 11 を手動で配置（NVIDIA ログイン要）
. scripts/activate.ps1             # セッションごと: PATH と環境変数を設定
pwsh -File scripts/verify.ps1      # 疎通確認: cargo test + bundle
```

## ビルド

CLI（`vc-rs`）:

```powershell
cargo build --release
```

VST3 プラグイン（`vc-vst3`）:

```powershell
cargo xtask bundle vc-vst3 --release
# TensorRT 専用ビルド:
cargo xtask bundle vc-vst3 --release --no-default-features --features tensorrt
```

プラグインの詳細は [`crates/vc-vst3/README.md`](../crates/vc-vst3/README.md) を参照してください。

## GPU 実行用ランタイム

セッションごとに `. scripts/activate.ps1` を実行すると、配置済みの CUDA / cuDNN /
TensorRT を `PATH` 先頭へ追加し、`CUDA_PATH` / `TENSORRT_ROOT` / `ORT_CUDA_VERSION`
を設定します（手動で PATH を通す場合は下表のとおり）。

| ランタイム | バージョン | ダウンロードページ |
| --- | --- | --- |
| CUDA Toolkit | 13.2 | [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) |
| cuDNN | 9.x for CUDA 13 | [cuDNN Downloads](https://developer.nvidia.com/cudnn-downloads) |
| TensorRT | 11.x（CUDA 13.x 対応。開発環境では 11.0.0.114） | [TensorRT SDK](https://developer.nvidia.com/tensorrt) |

`--provider cuda` を使う場合は CUDA Toolkit と cuDNN を、`--provider tensorrt` を
使う場合はさらに TensorRT を配置します。ビルドはワークスペース直下にある最も新しい
TensorRT を自動検出し、対応する CUDA Toolkit を選択します（`TENSORRT_ROOT` /
`CUDA_PATH` で上書き可能）。

TensorRT は初回実行時やモデル・入力形状が変わったタイミングでエンジンを生成する
ため、コンパイルに非常に長い時間がかかることがあります。2 回目以降はエンジン
キャッシュが再利用できれば起動が短くなります。

> テスト実行ファイルはネイティブ TensorRT シムをリンクするため、TensorRT の `bin`
> が `PATH` にないと `STATUS_DLL_NOT_FOUND` で起動に失敗します。GPU スタックなしで
> テストだけ素早く回したいときは `VC_RS_ENABLE_NATIVE_TENSORRT=0` を設定してください。
