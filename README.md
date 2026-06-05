# vc-rs

> 日本語 | [English](README.en.md)

`vc-rs` は Rust 製の **RVC 音声変換アプリ** です。マイク入力や WAV ファイルを、
ONNX 形式の RVC モデルで別の声に変換します。次の 2 つの使い方があります。

- **CLI 版（`vc-rs.exe`）** — コマンドラインで動かす本体。リアルタイム変換と
  WAV ファイル変換に対応します。
- **VST3 プラグイン版（`vc-vst3.vst3`）** — お使いの DAW に読み込んで使う
  プラグインです。

ビルド済みの Windows 用パッケージを配布しています。**ソースからビルドする必要は
ありません。** ダウンロードして展開し、モデルを用意すればすぐ使えます。

> ソースからビルドしたい開発者の方は [`docs/development_ja.md`](docs/development_ja.md)
> を参照してください。内部設計は [`docs/architecture_ja.md`](docs/architecture_ja.md)
> にあります。

## ダウンロード

最新版は GitHub の **[Releases](https://github.com/shirohata/vc-rs/releases)**
から入手できます。配布パッケージは Windows (x64) 向けで、用途と環境に合わせて
次の 4 種類があります。

| パッケージ | 形態 | バックエンド | 対象環境 | サイズ | 必要なもの |
| --- | --- | --- | --- | --- | --- |
| `vc-rs-cli-windowsml-…zip` | CLI | Windows ML | 多くの GPU（NVIDIA 以外も可） | 小（数 MB） | Windows App SDK ランタイム |
| `vc-rs-cli-tensorrt-…zip` | CLI | TensorRT | NVIDIA GPU | 大（約 1.9 GB） | 最新の NVIDIA ドライバ |
| `vc-vst3-windowsml-…zip` | VST3 プラグイン | Windows ML | 多くの GPU（NVIDIA 以外も可） | 小 | Windows App SDK ランタイム |
| `vc-vst3-tensorrt-…zip` | VST3 プラグイン | TensorRT | NVIDIA GPU | 大（約 1.9 GB） | 最新の NVIDIA ドライバ |

**どれを選べばよいか:**

- まず試すなら **windowsml 版**。ダウンロードが軽く、NVIDIA 以外の GPU でも
  DirectML 経由で動きます。
- **NVIDIA GPU を持っていて最速を狙う**なら **tensorrt 版**。ダウンロードは
  大きく、初回起動時にエンジン構築で時間がかかりますが、その後は高速です。
- DAW で歌や配信に使うなら **VST3 版**、単体・自動化・WAV 一括変換なら **CLI 版**。

## 必要なもの

### windowsml 版

- **Windows App SDK ランタイム（2.x 系）** をインストールしてください。ONNX
  Runtime と DirectML を提供します。Microsoft の
  [Windows App SDK ダウンロードページ](https://learn.microsoft.com/windows/apps/windows-app-sdk/downloads)
  から、最新安定版の **Runtime（ランタイム）インストーラ** を入れてください。

### tensorrt 版

- **最新の NVIDIA GPU ドライバ**。TensorRT 本体の DLL はパッケージに同梱して
  いるので、CUDA や TensorRT を別途インストールする必要はありません。

### 共通: モデルファイル

`vc-rs` はモデルを同梱しません。次の 3 つを自分で用意します。

1. **RVC 音声変換モデル**（`.onnx`） — 変換したい声のモデル。**ONNX 形式のみ
   対応**です。`.pth` は直接読み込めません（RVC 系ツールや VCClient などで
   事前に `.onnx` へ変換してください）。
2. **埋め込み抽出モデル**（ContentVec, `content_vec_500.onnx`）
3. **F0 推定モデル**（RMVPE, `rmvpe.onnx`）

2 と 3 は付属の `download-models.ps1` で取得できます（下記「モデルの準備」）。

## 使い方（CLI 版）

1. ダウンロードした zip を展開します（**DLL は `vc-rs.exe` と同じフォルダに
   置いたまま**にしてください）。
2. そのフォルダで PowerShell を開きます。

### モデルの準備

埋め込み・F0 モデルを取得します（このフォルダで実行）。

```powershell
pwsh .\download-models.ps1
```

`.\assets\content_vec_500.onnx` と `.\assets\rmvpe.onnx` がダウンロードされます。
RVC 音声変換モデル（`.onnx`）は別途自分で用意してください。

> これらのモデルは第三者配布（配布元では GPL-3.0 表示）で、`vc-rs` 本体の
> MIT License の対象外です。利用・改変・再配布の際は配布元のライセンスに
> 従ってください。詳細は `download-models.ps1` 内の注記を参照してください。

### デバイス確認

```powershell
.\vc-rs.exe devices
```

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

```powershell
.\vc-rs.exe wav --model <あなたのRVCモデル>.onnx `
    --embedder .\assets\content_vec_500.onnx `
    --f0-model .\assets\rmvpe.onnx `
    --input input.wav --output out.wav `
    --provider windowsml --speaker-id 0
```

### モデル構造の確認

```powershell
.\vc-rs.exe inspect --model <あなたのRVCモデル>.onnx
```

`inspect` は実行バックエンドに依存せず、ONNX モデルの入力・出力・メタデータを
表示します。

## 使い方（VST3 プラグイン版）

1. zip を展開し、`vc-vst3-windowsml.vst3` または
   `vc-vst3-tensorrt.vst3` を VST3 の標準フォルダにコピーします。
   - Windows: `%CommonProgramFiles%\VST3\`（例: `C:\Program Files\Common Files\VST3`）
2. 展開したフォルダで `pwsh .\download-models.ps1` を実行し、埋め込み・F0
   モデルを `.\assets\` に取得します（**インストール先ではなく、展開した
   フォルダで実行**してください）。
3. DAW でプラグインを読み込み、エディタ画面を開きます。
   - **Browse** から RVC モデル・埋め込み（ContentVec）・F0（RMVPE）の各 `.onnx`
     を指定します。
   - **バックエンド**を選びます（windowsml 版: `windowsml` / `windowsml-directml`
     / `cpu`、tensorrt 版: `tensorrt`）。
   - **chunk size**（ms）を設定します（大きいほど安定しますが遅延が増えます）。
   - **Load / Reload** を押して反映します。モデル・バックエンド・chunk の変更は
     このボタンを押すまで適用されません。
   - Pitch / Speaker / Input・Output ゲインはリアルタイムに反映され、DAW の
     パラメータとして自動化・保存できます。

モデルパスや設定はプロジェクト/プリセットごとに保存されます。詳細は
[`crates/vc-vst3/README.md`](crates/vc-vst3/README.md) を参照してください。

## リアルタイム設定の調整

音切れ・遅延・CPU/GPU 負荷のバランスは `--chunk-ms` と `--extra-convert-ms`
（VST3 では chunk size）で調整します。

- `--chunk-ms`: 1 回の処理でまとめる音声の長さ。音切れや負荷の張り付きが出る
  場合は大きくします（`500` → `750` → `1000`）。大きいほど安定しますが、入力から
  出力までの体感遅延も増えます。GPU 実行ではより小さい値を使えることがあります。
- `--extra-convert-ms`: 変換に渡す前後文脈の長さ。大きくすると安定することが
  ありますが負荷も増えます。まず `100` ms 付近から試してください。

設定を詰めるときは、**先に音切れしない値を見つけ、その後に `--chunk-ms` を
小さくして遅延を下げる**のが安全です。

主な変換パラメータ:

- `--speaker-id 0`: マルチスピーカーモデルで使う話者 ID（デフォルト: 0）。
- `--pitch-shift 0.0`: F0 を半音単位で上下（デフォルト: 0.0）。`12.0` で 1
  オクターブ上、`-12.0` で 1 オクターブ下。
- `--input-gain 1.0` / `--output-gain 1.0`: 入力・出力にかけるゲイン
  （デフォルト: 1.0）。小さすぎる場合に上げます。上げすぎるとクリップします。
- `--silence-threshold 0.0001`: 無音とみなすしきい値。
- `--rms-mix-rate <0.0-1.0>`: 0.0 に近いほど入力音量の起伏を反映、1.0 に近いほど
  モデル出力の音量を保持（デフォルト: 0.0）。

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
[`docs/tensorrt_performance_ja.md`](docs/tensorrt_performance_ja.md) を参照して
ください。

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

## トラブルシューティング / FAQ

**Q. windowsml 版が起動しない / モデル読み込みに失敗する**
A. **Windows App SDK ランタイム（2.x 系）** がインストールされているか確認して
ください（「必要なもの」参照）。

**Q. exe を実行すると SmartScreen の警告が出る**
A. 配布バイナリはコード署名していないため、Windows が警告を出すことがあります。
内容を確認のうえ「詳細情報」→「実行」で起動してください。

**Q. VST3 版が DAW でクラッシュする**
A. プラグインのフォルダに古い `onnxruntime_providers_cuda.dll` などの余分な
ONNX Runtime プロバイダ DLL が紛れ込んでいないか確認してください。windowsml 版の
バンドルには ONNX Runtime / DirectML / CUDA の DLL は含めません（システムの
Windows App SDK ランタイムが提供します）。配布 zip をそのまま展開した状態であれば
混入しませんが、過去のビルドからコピーした場合は削除してください。

**Q. `.pth` モデルが読み込めない**
A. RVC 音声変換モデルは **`.onnx` のみ対応**です。RVC 系ツールや VCClient などで
事前に ONNX へ変換してください。

**Q. リアルタイムで音が途切れる・遅延が大きい**
A. 「リアルタイム設定の調整」を参照してください。まず `--chunk-ms` を大きくして
音切れを止め、その後で遅延を詰めます。

## 補助スクリプト

`download-models.ps1` は任意の補助スクリプトです。第三者の参照用 ONNX モデル
（ContentVec / RMVPE）を [`wok000/weights_gpl`](https://huggingface.co/wok000/weights_gpl)
からダウンロードします。取得されるモデルは `vc-rs` 本体に含まれず、この
リポジトリの MIT License の対象でもありません（配布元は GPL-3.0 表示）。

## Acknowledgements

- 本実装は RVC 系 OSS 実装の知見を参考にしています。とくに Applio、VCClient、
  RVC WebUI の設計や実装上の工夫から多くを学んでいます。
- 関連する third-party notice は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
  にまとめています。

## License

MIT License（[`LICENSE`](LICENSE) を参照）。外部プロジェクトとモデルファイルに
関する注意事項は [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) を参照して
ください。
