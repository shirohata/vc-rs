# vc-rs

`vc-rs` は Rust 製の CLI ベース音声変換（RVC）実験アプリです。
WAV ファイルまたはリアルタイム入力（マイク）に対して、ONNX 形式の
RVC モデルで音声変換を実行します。

詳細な内部設計は [`docs/architecture_ja.md`](docs/architecture_ja.md) と
[`docs/architecture.md`](docs/architecture.md) を参照してください。

ソースからビルドする場合は、開発者向けの
[`docs/development_ja.md`](docs/development_ja.md)（必要環境・ビルド・GPU 実行用
ランタイム・環境構築スクリプト）を参照してください。

## 対応機能

- CLI での実行（GUI なし）
- オーディオデバイス列挙（`devices`）
- ONNX モデルの入力/出力/メタデータ確認（`inspect`）
- WAV ファイル変換（`wav`）
- リアルタイム変換またはパススルー（`run`）
- ONNX Runtime の CPU / CUDA / Windows ML / TensorRT Provider 利用
- CPAL / WASAPI による音声入出力

## 外部モデルについて

このリポジトリには学習済みモデルを同梱しません。利用者側で次の ONNX
ファイルを準備してください。

- RVC 音声変換モデル: `.\assets\voice.onnx`
- ContentVec などの埋め込み抽出モデル: `.\assets\content_vec_500.onnx`
- RMVPE などの F0 推定モデル: `.\assets\rmvpe.onnx`

RVC 音声変換モデルは **ONNX 形式（`.onnx`）のみ対応**です。`.pth` モデルは
直接読み込めません。`.pth` を使う場合は、RVC 系ツールや VCClient などを使って
事前に ONNX へ変換してください。

> 以下の使用例はソースから実行する形（`cargo run -- ...`）で記載しています。
> ビルド方法は [`docs/development_ja.md`](docs/development_ja.md) を参照してください。

## 基本的な使い方

### デバイス確認

```powershell
cargo run -- devices
```

### モデル構造確認

`inspect` は実行バックエンドに依存しない構造確認コマンドです。ONNX モデルを CPU で読み、
入力、出力、メタデータを表示します。

```powershell
cargo run -- inspect --model .\assets\voice.onnx
```

### WAV 変換

```powershell
cargo run -- wav --model .\assets\voice.onnx --embedder .\assets\content_vec_500.onnx --f0-model .\assets\rmvpe.onnx --input input.wav --output out.wav --provider cpu --speaker-id 0
```

### リアルタイム変換

```powershell
cargo run -- run --model .\assets\voice.onnx --embedder .\assets\content_vec_500.onnx --f0-model .\assets\rmvpe.onnx --input "Microphone" --output "Speakers" --chunk-ms 500 --extra-convert-ms 100 --provider cpu --speaker-id 0
```

`--input` と `--output` には `devices` で表示されるデバイス名の一部を指定します。

## リアルタイム設定の調整

CPU 実行では、デフォルトの `--chunk-ms` と `--extra-convert-ms` は速い CPU 向けの
値です。音切れ、遅延の増加、CPU 使用率の張り付きが出る場合は、まず
`--chunk-ms` を大きくしてください。`500` で不安定なら `750`、`1000` のように
上げると 1 回あたりの処理時間に余裕ができますが、そのぶん入力から出力までの
体感遅延も増えます。

`--extra-convert-ms` は変換に渡す前後文脈の長さをミリ秒で指定します。大きくすると
変換が安定することがありますが、推論するサンプル数が増えるため負荷も増えます。
CPU でリアルタイム性を優先する場合は、まず `100` ms 付近から試し、品質が足りない
場合だけ少しずつ増やしてください。

主な変換パラメータ:

- `--speaker-id 0`: マルチスピーカーモデルで使う話者 ID です（デフォルト: 0）。
- `--pitch-shift 0.0`: F0 を半音単位で上下させます（デフォルト: 0.0）。
  `12.0` で 1 オクターブ上、`-12.0` で 1 オクターブ下です。声質やモデルにより
  自然に聞こえる範囲は異なります。

主な音量関連オプション:

- `--input-gain 1.0`: モデルへ渡す前の入力音声にかけるゲインです（デフォルト: 1.0）。
  入力が小さすぎる場合に上げます。大きくしすぎるとモデル入力がクリップしやすくなります。
- `--output-gain 1.0`: 変換後の出力音声にかけるゲインです（デフォルト: 1.0）。
  変換結果が小さい場合に上げます。大きくしすぎると出力がクリップしやすくなります。
- `--silence-threshold 0.0001`: 入力音声を無音扱いするしきい値です。小さくすると小さい声や環境音にも反応しやすくなり、大きくすると無音判定されやすくなります。
- `--rms-mix-rate <0.0-1.0>`: 0.0 から 1.0 までの数値を指定します。(デフォルト: 0.0)
  0.0 に近いほど入力音量の起伏を強く反映し、1.0 に近いほどモデル出力の
  音量を保持します。たとえば `0.5` はその中間の補正量です。

## GPU / Windows ML / TensorRT 実行

GPU 実行は `wav` / `run` の `--provider` で指定します。

```powershell
cargo run -- wav --model .\assets\voice.onnx --embedder .\assets\content_vec_500.onnx --f0-model .\assets\rmvpe.onnx --input input.wav --output out.wav --provider cuda --speaker-id 0
```

```powershell
cargo run -- run --model .\assets\voice.onnx --embedder .\assets\content_vec_500.onnx --f0-model .\assets\rmvpe.onnx --input "Microphone" --output "Speakers" --chunk-ms 200 --extra-convert-ms 1000 --provider tensorrt --speaker-id 0
```

Windows ML build では `--provider windowsml` が catalog EP を優先し、使える EP
がなければ DirectML、最後に CPU へ寄せます。特定の catalog EP を強制したい
場合は `windowsml-nvtrtx` / `windowsml-qnn` / `windowsml-openvino` /
`windowsml-migraphx` / `windowsml-vitisai` を指定します。これらの明示指定は
fallback せず、EP が未導入または未準備ならエラーになります。

Windows ML catalog EP の状態確認とインストールは CLI から明示的に実行できます。
`install` で provider を省略すると、vc-rs の優先順位でそのマシンに compatible
な最上位 EP を選択します。

```powershell
cargo run -- windowsml-eps list
cargo run -- windowsml-eps install
cargo run -- windowsml-eps install --provider nvtrtx --yes
```

GPU 実行では、CPU より小さい `--chunk-ms` や大きい `--extra-convert-ms` を使えることが
あります。設定を詰めるときは、先に音切れしない値を見つけ、その後に遅延を下げる
方向で `--chunk-ms` を小さくしていくのが安全です。

GPU 実行に必要なランタイム（CUDA / cuDNN / TensorRT）の準備と `PATH` 設定は、
開発者向けの [`docs/development_ja.md`](docs/development_ja.md#gpu-実行用ランタイム)
を参照してください。

## 補助スクリプト

`download-models.ps1` は任意の補助スクリプトです。第三者の参照用 ONNX モデルを
[`wok000/weights_gpl`](https://huggingface.co/wok000/weights_gpl) からダウンロードします。

このスクリプトで取得されるモデルファイルは `vc-rs` 本体には含まれず、この
リポジトリの MIT License の対象でもありません。配布元リポジトリでは GPL-3.0 と
表示されています。利用、改変、再配布を行う場合は、配布元のライセンスを確認して
従ってください。

```powershell
.\download-models.ps1
```

## Acknowledgements

- 本実装は RVC 系 OSS 実装の知見を参考にしています。とくに Applio、VCClient、
  RVC WebUI の設計や実装上の工夫から多くを学んでいます。
- 関連する third-party notice は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
  にまとめています。

## License

MIT License（[`LICENSE`](LICENSE) を参照）。外部プロジェクトとモデルファイルに
関する注意事項は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照してください。
