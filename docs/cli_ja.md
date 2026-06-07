# vc-rs CLI リファレンス（`vc-rs.exe`）

> 日本語 | [English](cli.md)

`vc-rs.exe` は GUI + CLI パッケージに同梱される CLI です。通常の声の変換は
[`vc-gui.exe`](../README.md) で完結しますが、CLI は **GUI にない以下の用途**に
使えます。

- **WAV ファイルの一括変換**（`wav`） — GUI はリアルタイム変換専用。
- **診断・モデル調査**（`doctor` / `devices` / `inspect`）。
- **Windows ML 実行プロバイダ（EP）の確認とインストール**（`windowsml-eps`）。
- **エンジンキャッシュの管理**（`engine-cache`）。
- **自動化・スクリプト化**、および GUI が公開していない細かい DSP/オーディオ
  パラメータ（WASAPI 排他、`psola`、`--rms-mix-rate` ほか）の調整。

GUI と CLI は同じ推論パイプラインを共有するため、CLI で詰めた変換設定はそのまま
同じ音質で再現できます。

## 準備

1. GUI + CLI パッケージの zip を展開します（**DLL は `vc-gui.exe` /
   `vc-rs.exe` と同じフォルダに置いたまま**にしてください）。
2. そのフォルダで PowerShell を開きます。
3. 埋め込み・F0 モデルを取得します（下記）。RVC 音声変換モデル（`.onnx`）は
   別途自分で用意してください（`.pth` は非対応）。

```powershell
pwsh .\download-models.ps1
```

`.\assets\content_vec_500.onnx` と `.\assets\rmvpe.onnx` がダウンロードされます。

> 必要なもの（Windows App SDK ランタイム / NVIDIA ドライバ）とパッケージの
> 選び方は [`README.md`](../README.md) を参照してください。

## コマンド一覧

```powershell
.\vc-rs.exe --help
```

| コマンド | 用途 |
| --- | --- |
| `doctor` | 実行に必要なランタイム依存とデバイスの見え方を診断 |
| `devices` | オーディオ入出力デバイスを一覧表示 |
| `inspect` | ONNX モデルの入力・出力・メタデータを表示（バックエンド非依存） |
| `run` | マイク→スピーカーのリアルタイム変換 |
| `wav` | WAV ファイル→WAV ファイル変換（同一パイプラインで決定的にテスト可能） |
| `windowsml-eps` | Windows ML catalog EP の一覧・インストール（windowsml 版のみ） |
| `engine-cache` | GPU エンジンキャッシュの確認・消去 |

### 診断

```powershell
.\vc-rs.exe doctor
```

### デバイス確認

```powershell
.\vc-rs.exe devices
```

### モデル構造の確認

```powershell
.\vc-rs.exe inspect --model <あなたのRVCモデル>.onnx
```

`inspect` は実行バックエンドに依存せず、ONNX モデルの入力・出力・メタデータを
表示します。

### リアルタイム変換

```powershell
.\vc-rs.exe run --model <あなたのRVCモデル>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input "Microphone" --output "Speakers" `
    --chunk-ms 500 --extra-convert-ms 100 `
    --provider windowsml --speaker-id 0
```

`--input` と `--output` には `devices` で表示されるデバイス名の一部を指定します。
tensorrt 版では `--provider tensorrt` を指定してください。

### WAV ファイル変換

GUI にはない機能です。バッチ処理や、設定変更の決定的な検証に使えます。

```powershell
.\vc-rs.exe wav --model <あなたのRVCモデル>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input input.wav --output out.wav `
    --provider windowsml --speaker-id 0
```

## リアルタイム設定の調整

音切れ・遅延・CPU/GPU 負荷のバランスは `--chunk-ms` と `--extra-convert-ms`
で調整します。

- `--chunk-ms`: 1 回の処理でまとめる音声の長さ。音切れや負荷の張り付きが出る
  場合は大きくします（`500` → `750` → `1000`）。大きいほど安定しますが、入力から
  出力までの体感遅延も増えます。GPU 実行ではより小さい値を使えることがあります。
- `--extra-convert-ms`: 変換に渡す前後文脈の長さ。大きくすると安定することが
  ありますが負荷も増えます。まず `100` ms 付近から試してください。

設定を詰めるときは、**先に音切れしない値を見つけ、その後に `--chunk-ms` を
小さくして遅延を下げる**のが安全です。

## 主な変換パラメータ

- `--speaker-id 0`: マルチスピーカーモデルで使う話者 ID（デフォルト: 0）。
- `--pitch-shift 0.0`: F0 を半音単位で上下（デフォルト: 0.0）。`12.0` で 1
  オクターブ上、`-12.0` で 1 オクターブ下。
- `--input-gain 1.0` / `--output-gain 1.0`: 入力・出力にかけるゲイン
  （デフォルト: 1.0）。小さすぎる場合に上げます。上げすぎるとクリップします。
- `--silence-threshold 0.0001`: 無音とみなすしきい値。
- `--rms-mix-rate <0.0-1.0>`: 0.0 に近いほど入力音量の起伏を反映、1.0 に近いほど
  モデル出力の音量を保持（デフォルト: 0.0）。

その他に `--smoother sola|psola`、`--sola-search-ms`、`--crossfade-ms`、
`--rvc-output-tail-discard-ms`、`--gpu-priority normal|high`、WASAPI 関連
（`--audio-backend wasapi`、`--wasapi-exclusive*`、`--wasapi-buffer-ms`）など、
GUI が固定している項目も CLI から指定できます。各オプションの一覧と既定値は
`--help` で確認してください。

## Windows ML の実行プロバイダ（windowsml 版）

windowsml 版で `--provider windowsml` を指定すると、Windows ML の catalog EP を
優先し、使える EP がなければ DirectML、最後に CPU へ寄せます。特定の EP を
強制したい場合は `windowsml-nvtrtx` / `windowsml-qnn` / `windowsml-openvino` /
`windowsml-migraphx` / `windowsml-vitisai` を指定します（fallback せず、EP が
未導入・未準備ならエラー）。

catalog EP の状態確認とインストールは CLI から行えます。

```powershell
.\vc-rs.exe windowsml-eps list
.\vc-rs.exe windowsml-eps install            # 最適な EP を自動選択
.\vc-rs.exe windowsml-eps install --provider nvtrtx --yes
```

## TensorRT の実行（tensorrt 版）

tensorrt 版は GPU 実行を **同梱の TensorRT ランタイム** で行うため、NVIDIA
ドライバ以外の追加インストールは不要です。

> ⚠️ TensorRT は **初回実行時やモデル・入力形状が変わったとき**にエンジンを
> 生成するため、起動に非常に長い時間がかかることがあります。2 回目以降は
> エンジンキャッシュが再利用され、起動が短くなります。

TensorRT の詳しい性能特性は
[`tensorrt_performance_ja.md`](tensorrt_performance_ja.md) を参照してください。

## エンジンキャッシュの管理

TensorRT（tensorrt 版）と Windows ML の TensorRT-RTX（`windowsml-nvtrtx`）が
生成したエンジンは `%LOCALAPPDATA%\vc-rs\tensorrt-cache` に保存され、両バック
エンドで共有されます（`VC_RS_TENSORRT_CACHE_DIR` で場所を変更可）。場所・サイズ
の確認とキャッシュ消去は CLI から行えます。

```powershell
.\vc-rs.exe engine-cache info          # 場所・合計サイズ・モデル別の内訳を表示
.\vc-rs.exe engine-cache clear         # 確認のうえ全削除
.\vc-rs.exe engine-cache clear --yes   # 確認なしで全削除
```

キャッシュは再生成可能な派生データなので、削除しても次回のモデル読み込み時に
作り直されるだけです（その回だけ起動が長くなります）。
