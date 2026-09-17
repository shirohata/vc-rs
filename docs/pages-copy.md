# GitHub Pages introduction copy

Draft copy for the existing Pages site. Not published. Keep setup instructions in
the README and support details in the backend guide; check the target release
before publishing feature claims. These links should be published only after the
referenced documents have reached `main`.

## 日本語

### 見出し

GPUに合わせて選べる、RVCボイスチェンジャー

### 紹介文

NVIDIA向けネイティブTensorRTに対応。単体アプリでも、DAWのVST3プラグインでも、
RVCモデルで声を変換できます。Python環境の準備は不要。対応PTHモデルの変換と
補助モデルの取得も、アプリから進められます。

### 特長

- **GPUに合わせた推論:** ネイティブTensorRTとWindows ML / DirectMLを選択。
  Windows ML経由のMIGraphX・OpenVINOも実験的に対応しています。
- **単体でもDAWでも:** GUIでリアルタイム変換。VST3ではピッチやゲインを自動化し、
  設定をプロジェクトに保存できます。
- **既存モデルから始める:** ONNXモデルと、アプリ内変換に対応するRVC v2 / F0の
  PTHモデルを利用できます。同梱CLIではWAV変換や自動化も可能です。

### 利用条件とリンク

Windows x64向け。声モデルは別途必要です。TensorRTは初回にエンジン構築が必要です。
MIGraphX・OpenVINOは対応EP・デバイスが必要です。MIGraphXは実機未検証、
OpenVINOは一部Intel環境で検証・試聴済みで、NPUは未検証です。

[ダウンロード](https://github.com/shirohata/vc-rs/releases/latest) ·
[使い始め方](https://github.com/shirohata/vc-rs/blob/main/README.ja.md) ·
[バックエンドの対応範囲](https://github.com/shirohata/vc-rs/blob/main/docs/backends_ja.md)

## English

### Heading

An RVC voice changer with inference choices for your GPU

### Introduction

Native TensorRT for NVIDIA GPUs, in a standalone app or a VST3 plugin inside your
DAW. Convert voices with your RVC models without setting up Python. The app also
converts supported PTH models and downloads support models.

### Features

- **Inference for your hardware:** choose native TensorRT or Windows ML / DirectML.
  MIGraphX and OpenVINO through Windows ML are also available experimentally.
- **Standalone or in your DAW:** real-time conversion in the GUI; pitch/gain
  automation and project-saved settings in VST3.
- **Bring your models:** use ONNX or supported RVC v2 / F0 PTH models through the
  built-in converter. The bundled CLI adds WAV conversion and automation.

### Conditions and links

For Windows x64. Supply your own voice model. TensorRT requires an initial engine
build. MIGraphX / OpenVINO need compatible EPs and devices. MIGraphX remains
unverified on target hardware; OpenVINO has limited Intel hardware validation
and listening checks. NPU remains unverified.

[Download](https://github.com/shirohata/vc-rs/releases/latest) ·
[Get started](https://github.com/shirohata/vc-rs/blob/main/README.md) ·
[Backend support](https://github.com/shirohata/vc-rs/blob/main/docs/backends.md)
