# 推論バックエンドの選び方

> [English](backends.md) | 日本語

vc-rsはGUI・CLI・VST3で共有する変換パイプラインに、複数の推論バックエンドを
組み合わせています。このガイドは現行ソースの説明です。ダウンロードする版の
対応機能は[リリース情報](https://github.com/shirohata/vc-rs/releases)も確認してください。

## まず何を選ぶか

| 目的 | 選択 |
| --- | --- |
| まず動かしたい、NVIDIA以外のGPUを使いたい | Windows ML版。既定の `windowsml` は自動選択。DirectMLを指定するなら `windowsml-directml` |
| NVIDIA GPUでネイティブTensorRTを使いたい | TensorRT版の `tensorrt`。初回構築とキャッシュ用の時間・容量を確保する |
| AMD向けMIGraphXを検証したい | Windows ML版の `windowsml-migraphx`（実験的） |
| Intel向けOpenVINOを検証したい | Windows ML版の `windowsml-openvino-cpu` / `windowsml-openvino-gpu` / `windowsml-openvino-npu`（実験的） |

単体利用にはGUI + CLI、DAW内ではVST3を選びます。Windows ML版はWindows App SDK
Runtime 2.1以上を利用し、TensorRT版は必要なランタイムを同梱します。
モデルは別途必要です。[導入手順](../README.ja.md)を参照してください。

## バックエンドを切り替える

GUIは「バックエンド詳細」のバックエンド選択を使い、変更後に再起動してモデルを
読み込み直します。VST3ではバックエンドを選び **Load / Reload** を押します。
CLIでは `run` / `wav` の `--provider` に上表の名前を指定します。
GUI/VST3ではOpenVINOのバックエンドとCPU/GPU/NPUのデバイス種類を別々に選びます。
EP未導入の場合は画面から取得操作を行えます。デバイスの検出結果はモデル互換性を保証しません。
Windows MLのカタログEPは、そのPCのカタログに応じてGUI/VST3の選択肢に追加されます。

Windows ML版に同梱されたCLIで、環境とEPの状態を確認できます。

```powershell
.\vc-rs.exe doctor
.\vc-rs.exe windowsml-eps list
.\vc-rs.exe windowsml-eps install --help
```

EPはONNX Runtimeに推論の実行を担当させる追加コンポーネントです。
インストール操作やCLIの使い方は[CLIガイド](cli_ja.md)を参照してください。
MIGraphX・OpenVINO専用ZIPはありません。

## ネイティブTensorRT

ContentVec、RMVPE、RVC生成器をネイティブTensorRTで実行します。ONNX Runtimeの
TensorRT EPを経由する構成ではありません。固定shapeのエンジンを構築し、キャッシュを
再利用します。NVIDIAドライバ以外にCUDAやTensorRTを別途インストールする必要はありません。

初回、モデルや入力shapeが変わったとき、同梱TensorRTのバージョンが変わったときは、
エンジンを構築するため待ち時間が生じます。チャンク・追加コンテキスト等は入力shapeに
影響します。まず使用する設定を決め、構築完了後の定常状態で音切れと音質を確認してください。

キャッシュの場所と容量は `.\vc-rs.exe engine-cache info` で確認できます。
ZIP、展開したランタイム、モデル、エンジンキャッシュは別々に容量を使います。
削除・場所変更は[キャッシュ管理](cli_ja.md#エンジンキャッシュの管理)を参照してください。

Windows ML版の `windowsml-nvtrtx` は別の **TensorRT-RTX EP** です。
ランタイム、構築方法、対応モデル、性能、キャッシュの扱いは同一ではありません。
現行実装ではstreaming NSF入力付きRVCのTensorRT-RTXセッションは、終了時の問題を
避けるためランタイムキャッシュを無効にしており、読み込みごとに再構築します。

## MIGraphX / OpenVINOの実験的な対応

両方ともWindows MLカタログEPを登録し、共有パイプラインのセッション作成・推論へ
接続する実装があります。MIGraphXは実機未検証です。OpenVINOにはIntel Iris Xe / Core
i7-1195G7でのモデル別計測とGUI試聴の記録がありますが、全機器・モデルの検証ではありません。
[検証記録](openvino-model-routing_ja.md)の設定変更履歴も参照してください。
EPの導入やセッション作成成功だけで加速成功とは判断できません。

- **MIGraphX:** 対応AMD GPUが必要です。個別GPUの指定や専用チューニング・キャッシュ管理は
  未実装で、EPの既定動作に依存します。動的shapeでは初回推論時にもコンパイルが発生し得ます。
- **OpenVINO:** CPU / GPU / NPUの種類を指定できます。指定した種類が見つからない場合は
  エラーとなり、別種類に自動変更しません。複数の同種デバイスから特定の1台を選ぶ機能は
  ありません。従来の `windowsml-openvino` は種類を制限しません。
  NPUは特にモデル・演算・shapeの互換性確認が必要です。

OpenVINO GPUを使う経路では、音質上の問題を避けるためRMVPEはOpenVINO CPUで実行し、
ContentVecは固定入力shape、RVCは動的shapeを使います。GPU全段実行ではありません。
現在はGPUの精度を個別に強制せずEPに委ねており、精度要求の指定だけで実効FP32を保証しません。

`windowsml` Autoはモデル読み込み時にカタログEP、DirectML、CPUへ再試行できます。
明示EPでは別EPへ自動再試行しません。ただし非対応演算を処理する **ORT CPU fallback** は
別の仕組みで、OpenVINO等のセッションでも有効です。Autoが初回推論以降のあらゆる失敗を
回復するわけでもありません。デバイス選択のログだけでは演算のGPU/NPU割り当ては分かりません。

## 速度と音質を比較する

同じモデル・入力音声・チャンク・追加コンテキスト・前後処理で比較し、初回構築を
定常時の処理時間と分けます。小さいチャンクは入力の待ち時間を減らせますが、
処理の余裕と音質も確認してください。処理時間、コンテンツ保持時間、実際の
マイクから出力までの遅延は異なる値です。

[TensorRT性能調査](tensorrt_performance_ja.md)はRTX 3060 Ti、TensorRT 11.0 / CUDA 13.2の
`trtexec`によるモデル単体測定です。現在の配布版の全体性能や他製品との優劣を示しません。
各段のp95の和をパイプライン全体の実測p95や入出力遅延として使わないでください。
処理と遅延の関係は[設計資料](architecture.md#latency-trade-offs)に記載しています。
