# vc-rs 開発者向けガイド（ソースからビルド）

このドキュメントは **ソースからビルドする開発者向け** です。CLI の使い方や各
オプションの説明はルート [README](../README.ja.md) にあります。

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
# → scripts/README.md の推奨バージョンと手順に従って NVIDIA SDK を配置
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
を設定します。推奨バージョン、ダウンロード先、cuDNN が必要になる条件は
[`scripts/README.md`](../scripts/README.md#first-time-setup) に集約しています。

`--provider cuda` を使う場合は CUDA Toolkit と cuDNN を、`--provider tensorrt` を
使う場合はさらに TensorRT を `external\nvidia\` 配下に配置します。ビルドは
`external\nvidia\` にある最も新しい TensorRT を自動検出し、対応する CUDA Toolkit を選択します（`TENSORRT_ROOT` /
`CUDA_PATH` で上書き可能）。

TensorRT は初回実行時やモデル・入力形状が変わったタイミングでエンジンを生成する
ため、コンパイルに非常に長い時間がかかることがあります。2 回目以降はエンジン
キャッシュが再利用できれば起動が短くなります。

> テスト実行ファイルはネイティブ TensorRT シムをリンクするため、TensorRT の `bin`
> が `PATH` にないと `STATUS_DLL_NOT_FOUND` で起動に失敗します。GPU スタックなしで
> テストだけ素早く回したいときは `VC_RS_ENABLE_NATIVE_TENSORRT=0` を設定してください。

<a id="windows-ml-troubleshooting"></a>

## Windows ML テストのトラブルシューティング

### サンドボックス内の bootstrap 失敗

2026-09-08 の検証では、必要なランタイムがインストールされていても Windows
サンドボックス内で `MddBootstrapInitialize2` が HRESULT `0x80670016` を返しました。
同じ配布パッケージの `vc-rs.exe doctor` と、後述の `vc_convert_ort` テスト3件は
サンドボックス外で成功しています。原因となる具体的な Windows 権限は未特定です。
この HRESULT だけでランタイム不足と判断しないでください。

2026-09-16 には bootstrap が成功しても、サンドボックス内で OpenVINO EP が
カタログに見つからず、同じ `openvino_installed_device_probe` が外では CPU/GPU を
検出するケースも確認しました。カタログ未検出だけで EP の未導入を断定せず、
同じ診断を以下の手順で外側でも確認してください。

承認済みの検証作業では、同じ診断・テストをツールの
`sandbox_permissions: require_escalated` 経由で再試行します。通常のツール承認を
使用し、再試行の前に重複する会話上の確認は求めません。昇格が使えない場合や
拒否された場合は検証の制約を報告し、独立して進められる作業を続けます。

サンドボックス外で成功した場合は環境の制約として記録し、リリース作業を続けます。
この症状への対処としてランタイムの再インストール、bootstrap コードの変更、
必要ランタイムの要件引き下げ、サンドボックスの全体的な無効化は行いません。
外でも失敗する場合は実際の未解決障害として調査します。

### 開発テストで bootstrap DLL が見つからない場合

`Microsoft.WindowsAppRuntime.Bootstrap.dll` が見つからない `os error 126` は、
上記の HRESULT とは別の問題です。現在の Windows ML パッケージ内にある同 DLL の
絶対パスを、そのシェルの `VC_RS_WINDOWSML_BOOTSTRAP_DLL` 環境変数に設定してください。
この設定はローカル限定とし、マシン固有のパスを追跡対象ファイルへ書き込まないでください。

設定後、リポジトリルートから workspace の feature 構成で実行します。

```powershell
. scripts/activate.ps1
. scripts/rustflags.ps1
cargo test --workspace model_rvc::tests::vc_convert_ort -- --nocapture
```

テスト3件が実際に実行されたことを確認してください。`cargo test -p vc-core` だけでは
feature 構成が変わり、該当テストが0件のまま成功することがあります。
CPU のみのテスト成功は Windows ML の検証を代替しません。
