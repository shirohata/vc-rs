# Applio生成モデル

v2・F0ありの推論用PTHについて、HiFi-GAN、MRF HiFi-GAN、RefineGANを
GUIの既存コンバーターでONNXへ変換できます。生成器はCLI・GUI・VST3・WAVで
共有する`vc-core`の同じパイプラインを使用します。

MRF／RefineGANは実験的対応です。合成された未学習モデルでの数値比較・状態検査と、
学習済みモデルによる音質・リアルタイム性能の検証は別です。後者が済んでいない
構成を、リリースで動作確認済みと表記しないでください。

## モデルの条件

- `weight`と18要素の`config`を含むZIP形式の推論用PTH。学習再開用G/Dは対象外。
- `version=v2`、`f0=true`、10 msあたりの出力サンプル数がupsample倍率の積と一致。
- `vocoder`は`HiFi-GAN`（省略時も同じ）、`MRF HiFi-GAN`、`RefineGAN`のいずれか。
- 学習時と同じembedder重み・特徴層・前処理を使用。標準Applioは16 kHz入力から
  最終層768次元を出すembedderを使用します。別プロジェクトvc-embeddersの
  `unit12`出力を利用でき、フレーム補間はvc-rs側が担当します。
- v1、F0なし、別フォークの独自Vocoderや標準と異なる倍音数は対象外。

参照実装は[Applio 5171e1dc](https://github.com/IAHispano/Applio/tree/5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1)
です。HiFi-GANの最終Conv直前のLeakyReLU係数は参照と同じ0.01です。
従来コンバーターの0.1から変わるため、再変換したONNXの音声は以前の生成物と
異なる可能性があります。既存ONNXを読み込み時に書き換えることはありません。

## ストリーミング契約

標準HiFi-GANは形式v1を維持します。MRF／RefineGANは形式v2を使用します。

| 値 | 形式v2 |
| --- | --- |
| `rvc.frame_hop` | 10 ms特徴フレームあたりの音声サンプル数 |
| `rvc.noise_frame_hop` | 特徴フレームあたりの全Gaussian乱数要素数 |
| `rvc.phase_channels` | MRFは9、RefineGANは1 |
| `nsf_noise` | float32 `[1, frames*noise_frame_hop, 1]` |
| `phase_in` | float32 `[1,1,phase_channels]`、窓開始直前の位相 |
| `streaming_nsf_phase` | float32 `[1,frames*frame_hop,phase_channels]` |

乱数は特徴フレーム順に格納します。各フレームは最初に励振の
`[samples_per_frame,harmonics]`、続いて各段・各残差分岐の前／後AdaINの
`[samples_per_stage_frame,channels]`を順に持ちます。各領域の乱数は独立です。
フレーム全体をcropしてから各時間解像度へ展開するため、同じ絶対時間の乱数が
重複窓で維持されます。学習済みAdaINの係数を無視したり、乱数をゼロにしたりしません。

MRFは基本波と8倍音の開始位相を保持します。初回の倍音位相は固定seedの一様乱数で、
reset時に再現します。次窓には位相traceの`advance_frames*frame_hop-1`を引き継ぎます。
RefineGANの位相は線形補間されたF0のサンプル軸で積算します。補間は窓端をclampするため、
次窓の最初のフレーム中心（`advance_frames*frame_hop+floor(frame_hop/2)-1`）を
基準に引き継ぎ、次窓のF0先頭値で半フレーム分の積算を引いて入力位相に戻します。
これにより、F0が変化する重複窓でも共通区間の励振位相を保持します。
traceが得られない場合のCPU積算も、この補間を反映します。

RefineGANのKaiser sincカーネルは変換時に計算してONNXへ固定します。
torchaudioと同じ幅・rolloff・beta・paddingを使用し、線形Resizeで代用しません。
両方式ともdecoder convolution cacheは持たず、context付き窓を再計算します。
特にRefineGANは多段乱数と広いフィルターにより転送量・メモリ・計算量が増えるため、
小さな合成モデルの速度から学習済みモデルの最小chunkを判断できません。

モデルの変換・読み込みと追加状態の管理はworker側です。audio callbackに
推論・乱数生成・ロック・追加のバッファ確保は導入していません。
GUI／VST3にVocoderやsmootherの選択操作を追加する必要はありません。

## 再現検証

チェックインした`tiny_applio_mrf.pth`と`tiny_applio_refine.pth`は未学習の
小規模fixtureです。`make_applio_pth.py`で、ネットワークやtorchなしに再生成できます。
RefineGANはpre_conv・pyramid幅とspeaker条件付けも縮小しています。
実際の標準幅モデルの性能・音質を表すものではありません。

通常のRustテストは変換・形式検査・重複乱数・多成分位相・resetを検査します。
`applio_cpu_streaming_and_pinned_bindings`は動的実行と固定入出力bindingを検査します。
Windows MLの開発用bootstrap設定は[開発手順](development_ja.md#windows-ml-troubleshooting)を参照してください。

公式ソースとの独立比較は、固定commitのApplioソースとtorch／torchaudio／onnx／
onnxruntimeのCPU環境を別途用意して実行します。ソースは
`scripts/applio-reference.json`のSHA-256（改行正規化・末尾空白除去後）を照合してからimportします。
スクリプトはモデルやソースをダウンロードせず、学習も行いません。

```powershell
cargo run -p vc-convert --example export_fixture -- crates/vc-convert/tests/fixtures/tiny_v2_f0.pth target/tiny_applio_hifi.onnx
cargo run -p vc-convert --example export_fixture -- crates/vc-convert/tests/fixtures/tiny_applio_mrf.pth target/tiny_applio_mrf.onnx
cargo run -p vc-convert --example export_fixture -- crates/vc-convert/tests/fixtures/tiny_applio_refine.pth target/tiny_applio_refine.onnx
python scripts/validate-applio.py --source <pinned-Applio-checkout> --exports target

# SDK環境とWindows ML bootstrapを、開発手順に従って準備して実行
cargo test -p vc-core --features windowsml,gtcrn applio_directml_streaming -- --ignored --nocapture
cargo test -p vc-core --features tensorrt,gtcrn applio_native_tensorrt_streaming -- --ignored --nocapture
```

比較は2／5／12フレーム、有声一定音程・滑走・無声切替で行います。
全乱数を対応する明示入力に固定し、音声と中間励振を比較します。
音声の合格条件は最大絶対誤差1e-4以下、平均絶対誤差1e-5以下、
励振は最大絶対誤差1e-4以下です。検証JSONは`target/`に保存します。
backendテストは`VC_RS_APPLIO_DUMP`を指定すると比較用float32音声を保存できます。

任意の推論用PTHと変換後ONNXを比較する場合は、`--checkpoint`と`--onnx`を指定します。
PTHのVocoderに応じた標準幅の公式生成器を使用します。

```powershell
python scripts/validate-applio.py --source <pinned-Applio-checkout> --checkpoint path/to/model.pth --onnx path/to/model.onnx --output target/applio-validation.json
```

数値基準を超えた場合も全ケースをJSONに記録し、終了コードを失敗にします。
