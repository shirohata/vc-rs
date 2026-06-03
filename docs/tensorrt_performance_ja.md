# TensorRT パフォーマンス調査

## 目的

このメモは、TensorRT 11 を使った RVC パイプラインの実測結果と、そこから見える
最適化方針をまとめます。対象は `ContentVec -> RMVPE -> RVC generator` の
3 段推論です。音声コールバックや SOLA/DSP を含むエンドツーエンド遅延ではなく、
`trtexec` によるモデル推論単体の比較です。

実測結果はワーカー側のモデル実行予算を判断するためのものです。リアルタイム音声
コールバックへ TensorRT ビルド、エンジンロード、ログ出力、ファイル I/O を移しては
いけません。

## 測定条件

- CPU: AMD Ryzen 5 5600X
- GPU: NVIDIA GeForce RTX 3060 Ti
- TensorRT: 11.0.0.114 for CUDA 13.2

`trtexec` の既定により、CUDA Graph は有効、H2D/D2H data transfer は測定から除外
されています。RVC generator については `--noCudaGraph` と `--includeDataTransfers`
の比較も取りました。

TensorRT 11 は strongly typed network が既定です。この調査では TensorRT 10 系の
`--fp16` / `--int8` のような精度フラグは使っていません。混合精度や量子化を比較する
場合は、事前に ONNX 側の tensor type / quantization を作ってから同じ手順で測ります。

## ベースライン

`builderOptimizationLevel=3`、`allocationStrategy=static`、固定 shape engine の結果です。
合計 p95 は、3 段を逐次実行した場合の単純和です。

| case | ContentVec p95 | RMVPE p95 | RVC p95 | 合計 p95 目安 |
| --- | ---: | ---: | ---: | ---: |
| `chunk_ms=100`, `extra_convert_ms=100` | 2.51 ms | 5.32 ms | 6.07 ms | 13.91 ms |
| `chunk_ms=200`, `extra_convert_ms=500` | 3.04 ms | 5.94 ms | 7.19 ms | 16.17 ms |
| `chunk_ms=500`, `extra_convert_ms=1000` | 4.32 ms | 8.02 ms | 11.54 ms | 23.88 ms |

推論単体では、100 ms チャンクでも十分な余裕があります。実運用で音切れや遅延が出る
場合は、TensorRT の純粋な kernel 実行時間よりも、音声 I/O、resampling、SOLA、
キュー詰まり、engine 初回ビルド/ロード、または CPU 側コピーの配置を疑うべきです。

役割別では RVC generator が最も重く、次に RMVPE、最後に ContentVec です。ただし
小さい shape では RMVPE と RVC の差は大きくありません。

## Builder Optimization Level

RVC generator のみ、`builderOptimizationLevel=0/3/5` を比較しました。

| case | level 0 p95 | level 3 p95 | level 5 p95 |
| --- | ---: | ---: | ---: |
| `100:100` | 80.89 ms | 5.91 ms | 5.29 ms |
| `200:500` | 118.10 ms | 6.53 ms | 6.52 ms |
| `500:1000` | 221.96 ms | 10.84 ms | 10.79 ms |

`level=0` はビルド時間は短いものの、RVC では実行性能が大きく落ちます。リアルタイム
用途では使わないでください。

`level=3` は実用上の標準値です。`level=5` は一部 shape で少し速くなりますが、
改善幅は小さく、ビルド時間は長くなります。最終配布用や、engine cache を長く再利用
できる用途でのみ検討するのが妥当です。

## CUDA Graph と Data Transfer

RVC generator で比較しました。

| case | 既定 p95 | `--noCudaGraph` p95 | `--includeDataTransfers` p95 |
| --- | ---: | ---: | ---: |
| `100:100` | 5.87 ms | 7.93 ms | 5.92 ms |
| `200:500` | 7.02 ms | 8.57 ms | 7.08 ms |
| `500:1000` | 11.54 ms | 13.76 ms | 11.60 ms |

CUDA Graph は有効のまま維持するべきです。無効化すると enqueue time が大きく増え、
p95 も悪化します。

RVC 単体では H2D/D2H の追加測定コストは小さく、`--includeDataTransfers` でも p95 は
ほぼ変わりません。ただしこれは `trtexec` のランダム入力での測定です。vc-rs 実装では
CPU 側の resampling、pitch preparation、IoBinding へのコピー、host/device tensor の
所有関係も含めて確認する必要があります。

## 固定 Shape と可変 Shape

同じ 3 つの runtime shape を、個別の固定 shape engine と、1 つの dynamic profile
engine で比較しました。dynamic profile は次の範囲です。

- min: `chunk_ms=100`, `extra_convert_ms=100`
- opt: `chunk_ms=200`, `extra_convert_ms=500`
- max: `chunk_ms=500`, `extra_convert_ms=1000`

| case | 固定 shape 合計 p95 | 可変 shape 合計 p95 | 差 |
| --- | ---: | ---: | ---: |
| `100:100` | 13.91 ms | 35.33 ms | 2.54x |
| `200:500` | 16.17 ms | 41.07 ms | 2.54x |
| `500:1000` | 23.88 ms | 59.61 ms | 2.50x |

役割別でも可変 shape はすべて遅くなりました。

| role | `100:100` 固定 -> 可変 | `200:500` 固定 -> 可変 | `500:1000` 固定 -> 可変 |
| --- | ---: | ---: | ---: |
| ContentVec | 2.51 -> 8.86 ms | 3.04 -> 9.12 ms | 4.32 -> 11.81 ms |
| RMVPE | 5.32 -> 14.63 ms | 5.94 -> 18.71 ms | 8.02 -> 25.56 ms |
| RVC | 6.07 -> 11.84 ms | 7.19 -> 13.24 ms | 11.54 -> 22.23 ms |

この構成では、可変 shape engine は実行時 p95 を約 2.5 倍悪化させます。1 engine で
複数設定を扱える利点はありますが、リアルタイム実行では固定 shape engine cache を
優先してください。

可変 shape の利点は、複数 shape の固定 engine をすべて作るより engine 数を減らせる
ことです。ただし、1 つの実運用設定だけなら固定 shape の方がビルドも実行も扱いやすく、
ランタイム性能も良好です。

## TensorRT-RTX との比較

同じ固定 shape 条件で TensorRT-RTX 1.5.0.114 も測定しました。TensorRT-RTX 1.5 は
data transfer 無効、RTX CUDA Graph wholeGraph 有効が既定です。

合計 p95 は 3 段推論を逐次実行した場合の単純和です。

| case | TensorRT 11 固定 p95 | TensorRT-RTX 1.5 既定 p95 | 差 |
| --- | ---: | ---: | ---: |
| `100:100` | 13.91 ms | 27.30 ms | 1.96x |
| `200:500` | 16.17 ms | 32.32 ms | 2.00x |
| `500:1000` | 23.88 ms | 47.25 ms | 1.98x |

この固定 shape の推論 p95 では、TensorRT-RTX 1.5 は TensorRT 11 より遅く、合計 p95 は
約 1.9-2.0 倍です。

TensorRT-RTX 1.5 のモード差は次の通りです。

| case | 1.5 既定 p95 | `--noCudaGraph` p95 | `--includeDataTransfers` p95 |
| --- | ---: | ---: | ---: |
| `100:100` | 27.30 ms | 33.88 ms | 27.27 ms |
| `200:500` | 32.32 ms | 37.82 ms | 31.97 ms |
| `500:1000` | 47.25 ms | 60.61 ms | 45.86 ms |

`--noCudaGraph` は明確に悪化します。TensorRT-RTX 1.5 では CUDA Graph を既定のまま
使うべきです。`--includeDataTransfers` は今回のランダム入力ベンチでは大きな悪化を
出していませんが、実アプリでは CPU 側の tensor 準備、host/device 間コピー、IoBinding
の配置が効くため、アプリ側計測と切り分けて判断してください。

TensorRT-RTX 1.5 の役割別 p95 は次の通りです。

| case | role | TensorRT 11 固定 p95 | TensorRT-RTX 1.5 既定 p95 | 差 |
| --- | --- | ---: | ---: | ---: |
| `100:100` | ContentVec | 2.51 ms | 6.27 ms | +3.75 ms |
| `100:100` | RMVPE | 5.32 ms | 10.09 ms | +4.77 ms |
| `100:100` | RVC | 6.07 ms | 10.95 ms | +4.88 ms |
| `200:500` | ContentVec | 3.04 ms | 9.00 ms | +5.96 ms |
| `200:500` | RMVPE | 5.94 ms | 11.81 ms | +5.87 ms |
| `200:500` | RVC | 7.19 ms | 11.52 ms | +4.33 ms |
| `500:1000` | ContentVec | 4.32 ms | 13.35 ms | +9.03 ms |
| `500:1000` | RMVPE | 8.02 ms | 16.90 ms | +8.88 ms |
| `500:1000` | RVC | 11.54 ms | 17.00 ms | +5.47 ms |

一方で TensorRT-RTX の engine build は短く、TensorRT-RTX 1.5 では 3 段合計で
約 15-16 秒でした。TensorRT 11 の約 86-93 秒に対してかなり短いため、TensorRT-RTX は
「ビルド時間を短くして検証を速く回す」用途では有効です。ただし、RTX 3060 Ti 上の
固定 shape リアルタイム推論では TensorRT 11 の方が有利です。

また TensorRT-RTX は engine load 後に JIT compilation が走るため、実運用では起動時や
設定変更時に全 shape を warm up し、リアルタイム音声経路へ初回 JIT を持ち込まない
設計が必要です。

## 推奨方針

- 本線は固定 shape engine cache とする。
- `builderOptimizationLevel=3` を標準にする。
- `builderOptimizationLevel=5` は、最終配布用または長期 cache 前提の追加比較に限定する。
- `builderOptimizationLevel=0` は RVC では避ける。
- CUDA Graph は有効のままにする。
- 固定 shape の本番推論は TensorRT 11 を優先し、TensorRT-RTX は短時間ビルドが必要な
  検証用途として扱う。
- 可変 shape engine は、chunk 設定を頻繁に変える検証用途に限定する。
- chunk / extra context を変更したら、固定 shape profile と engine cache key も一緒に
  見直す。

## 実装上の注意

TensorRT のビルドと engine cache miss は重い処理です。これらは必ずモデルワーカー側、
または起動・設定変更時の非リアルタイム経路に置きます。音声コールバックでは heap
allocation、blocking I/O、lock、logging、TensorRT engine build/load を行わないでください。

固定 shape の性能は、入力 tensor の形状とアドレスが安定している前提で成り立ちます。
CUDA Graph と IoBinding の都合上、ランタイム中に tensor を再確保したり、shape を暗黙に
変えたりする変更は慎重に扱う必要があります。`chunk_ms`、`extra_convert_ms`、
`crossfade_ms`、`sola_search_ms`、`rvc_output_tail_discard_ms` を変更する場合は、
`model_rvc::shape`、`model_rvc::tensorrt`、engine cache key、SOLA 出力長をまとめて
確認してください。
