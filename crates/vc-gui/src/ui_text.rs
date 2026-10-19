//! Presentation-only localization. English strings are stable keys; never
//! translate stored provider/host tokens, paths, or third-party diagnostics.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    #[serde(rename = "ja")]
    Japanese,
    #[default]
    #[serde(rename = "en", other)]
    English,
}

impl Language {
    pub fn text(self, english: &str) -> &str {
        if self == Self::English {
            return english;
        }
        TRANSLATIONS
            .iter()
            .find_map(|(key, ja)| (*key == english).then_some(*ja))
            .unwrap_or(english)
    }
}

// Per-egui-context state only: no global locale shared across app instances,
// and no dependency from engine or audio callbacks on the UI's language.
pub fn language(ui: &egui::Ui) -> Language {
    ui.ctx().data(|data| {
        data.get_temp::<Language>(egui::Id::new("vc-language"))
            .unwrap_or_default()
    })
}

pub fn set_language(ctx: &egui::Context, language: Language) {
    ctx.data_mut(|data| data.insert_temp(egui::Id::new("vc-language"), language));
}

pub fn engine_message(language: Language, message: &str) -> String {
    if language == Language::English {
        return message.to_string();
    }
    if let Some(role) = message
        .strip_prefix("Loading ")
        .and_then(|s| s.strip_suffix(" model"))
    {
        return format!("{role}モデルを読み込み中");
    }
    if let Some(role) = message
        .strip_prefix("Building ")
        .and_then(|s| s.strip_suffix(" TensorRT engine"))
    {
        return format!("{role}のTensorRTエンジンを構築中");
    }
    if let Some(devices) = message.strip_prefix("Running (in: ") {
        return format!(
            "実行中（入力: {}",
            devices.replace(" / out: ", " / 出力: ").replace(')', "）")
        );
    }
    language.text(message).to_string()
}

/// Translate application-owned error text only. Leave arbitrary provider/OS
/// details byte-for-byte intact so diagnostics remain useful for support.
pub fn diagnostic_message(language: Language, message: &str) -> String {
    if let Some((name, detail)) = message.split_once(": ") {
        return format!("{name}: {}", language.text(detail));
    }
    language.text(message).to_string()
}
pub const STEPS: [&str; 2] = ["1. Choose a voice", "2. Try your microphone"];
pub const ADVANCED: &str = "Advanced settings";
pub const SKIP: &str = "Skip tutorial";
pub const CHOOSE: &str = "Choose voice model…";
pub const VOICE_TITLE: &str = "Choose your voice model";
pub const VOICE_HELP: &str = "Choose an RVC voice model (.pth) or drop it here. The app will help you convert it into a usable format.";
pub const VOICE_COMPATIBILITY: &str = "For models trained with ContentVec, the next steps will help you prepare the required files. Models trained with a different feature extractor require you to provide its corresponding file separately.";
pub const VOICE_ONNX: &str = "Already converted? You can also use an .onnx file.";
pub const MODEL_HELP: &str = "Don't have a voice model?";
pub const MODEL_GUIDE: &str = "Get an RVC model from its creator and check its usage terms. ONNX models are checked before continuing. The built-in converter supports RVC v2 / F0 checkpoints; other .pth models must be converted with a compatible external tool.";
pub const CHECKING: &str = "Checking model structure…";
pub const CHECKED: &str =
    "RVC structure verified. Model loading and audio are checked when you start conversion.";
pub const NEXT: &str = "Continue";
pub const OPEN_MAIN: &str = "Open main screen";
pub const BACK: &str = "Back";
pub const PREPARE_HELP: &str =
    "Check the shared files and processing components needed for voice conversion.";
pub const DOWNLOAD: &str = "Agree and download required files";
pub const DOWNLOAD_INFO: &str = "ContentVec + RMVPE · up to 741 MB · GPL-3.0";
pub const SOURCE: &str = "Source and license";
pub const SUPPORT_READY: &str = "Support models are ready.";
pub const RUNTIME_HELP: &str = "Windows ML requires Windows App Runtime 2.x (version 2.1 or later). This error alone does not mean it is missing.";
pub const RUNTIME_INSTALL: &str = "If it is not installed, open Microsoft's download page, choose a stable 2.x runtime version 2.1 or later, download Installer (x64), and run it. The development SDK is not needed.";
pub const RUNTIME_LINK: &str = "Open installation instructions (Microsoft)";
pub const RUNTIME_RESTART: &str = "After installation, close vc-rs and start it again to check the runtime. Retry alone may keep showing the earlier initialization error.";
pub const RUNTIME_INSTALLED: &str = "Already installed? Check Runtime details. Permissions or missing app files can also cause failure; reinstalling may not help. You can retry the check without installing anything.";
pub const CUSTOM: &str = "Use existing support models";
pub const STOP: &str = "Stop";
pub const START: &str = "Start";
pub const APPLY: &str = "Apply changes / Restart";
pub const SETUP: &str = "Setup";
pub const REFRESH: &str = "Refresh devices";
pub const INPUT: &str = "Microphone";
pub const OUTPUT: &str = "Output / headphones";
pub const VOLUME: &str = "Output volume";
pub const PITCH: &str = "Voice pitch (semitones)";
pub const ROUTING: &str = "Use with Discord or another app";
pub const ROUTING_HELP: &str = "Install a virtual audio device if you do not already have one. Set this app's Output to its playback endpoint, then select the paired recording endpoint as the microphone in your call app. Endpoint names differ between drivers. Run the call app's microphone test. vc-rs does not install a virtual audio driver or change other apps automatically.";
pub const CANCEL: &str = "Cancel";
pub const DISMISS: &str = "Dismiss";
pub const RETRY: &str =
    "Check your connection and free disk space, then retry. Completed models are reused.";
pub const FINISH_ERROR: &str =
    "Setup could not be saved. Retry after fixing the settings directory permissions.";

pub const TRANSLATIONS: &[(&str, &str)] = &[
    ("Unspecified (legacy setting)", "デバイス指定なし（旧設定）"),
    ("Available", "利用可能"),
    ("Unavailable", "利用不可"),
    ("Unverified", "未確認"),
    ("Checking OpenVINO devices...", "OpenVINOデバイスを確認中..."),
    ("Could not verify OpenVINO devices. Hover for details.", "OpenVINOデバイスを確認できませんでした。ここにマウスを置くと詳細を表示します。"),
    ("Selected device is unavailable. Choose an available device.", "選択中のデバイスは利用できません。利用可能なデバイスを選んでください。"),
    ("Download OpenVINO to check available devices.", "利用可能なデバイスを確認するには、OpenVINOのダウンロードが必要です。"),
    ("Download and check", "ダウンロードして確認"),
    ("Downloading and preparing OpenVINO...", "OpenVINOをダウンロード・準備中..."),
    ("Retry device check", "デバイス確認を再試行"),
    ("Voice", "声"),
    ("Audio input", "入力"),
    ("Audio output", "出力"),
    ("Restart", "再起動"),
    ("Active backend", "適用中のバックエンド"),
    ("Unapplied changes — Restart to apply.", "未反映の変更があります。「再起動」で反映します。"),
    ("Send input audio to the output without voice conversion.", "声の変換をせず、入力音声を出力します。"),
    ("Preparing or stopping. Stop requests are handled after the current preparation step.", "準備・停止処理中です。停止要求は現在の準備処理が終わり次第反映されます。"),
    ("Model", "モデル"),
    ("Model information", "モデル情報"),
    ("Model name", "モデル名"),
    ("Model author", "モデル作者"),
    ("Model notes", "モデルの備考"),
    ("Model description", "モデルの説明"),
    ("Training epoch", "学習epoch"),
    ("Training step", "学習step"),
    ("Model creation date", "モデル作成日時"),
    ("Training embedder", "学習時の特徴抽出モデル"),
    ("Source vocoder", "元モデルのボコーダー"),
    ("Model license", "モデルのライセンス"),
    ("Model terms", "モデルの利用条件"),
    ("Source tool", "元モデルの作成ツール"),
    ("Source tool version", "元モデルの作成ツールのバージョン"),
    ("Export tool", "変換ツール"),
    ("Export tool version", "変換ツールのバージョン"),
    ("Export format", "エクスポート形式"),
    ("Selected file", "選択中のファイル"),
    ("No model selected.", "モデルが選択されていません。"),
    ("Running model", "動作中のモデル"),
    ("Model information is available after ONNX conversion.", "モデル情報はONNX変換後に確認できます。"),
    ("Refresh information", "情報を更新"),
    ("File size", "ファイルサイズ"),
    ("Sample rate", "サンプルレート"),
    ("Model version", "モデルのバージョン"),
    ("Speaker count", "話者数"),
    ("Active audio devices", "使用中のオーディオデバイス"),
    ("Device information is available while running.", "デバイス情報は実行中に確認できます。"),
    ("Reset to 0", "0に戻す"),
    ("Volume and pitch update live.", "音量と声の高さは即時反映されます。"),
    ("Audio devices and noise reduction", "音声デバイス・ノイズ低減"),
    ("Backend Details", "バックエンド詳細"),
    ("Sample rates", "入出力サンプルレート"),
    ("No engine error details.", "エンジンのエラー詳細はありません。"),
    ("Custom", "カスタム"),
    ("File path", "ファイルパス"),
    ("Agree and download", "同意して取得"),
    (OPEN_MAIN, "通常画面へ"),
    (RUNTIME_HELP, "Windows MLにはWindows App Runtime 2.x（2.1以上）が必要です。このエラーだけで未導入とは判断できません。"),
    (RUNTIME_INSTALL, "未導入の場合は、Microsoftのダウンロードページで安定版の2.x（2.1以上）を選び、Installer (x64) をダウンロードして実行してください。開発用SDKは不要です。"),
    (RUNTIME_LINK, "導入手順を開く（Microsoft公式）"),
    (RUNTIME_RESTART, "インストール後はvc-rsを閉じ、起動し直して確認してください。再確認ボタンだけでは、前の初期化エラーが残る場合があります。"),
    (RUNTIME_INSTALLED, "導入済みの場合は「実行環境の詳細」を確認してください。権限やアプリのファイル不足でも失敗するため、再インストールで解決するとは限りません。インストールせずに再確認することもできます。"),
    ("1. Language", "1. 言語"), ("2. Terms", "2. 利用条件"),
    ("3. Audio devices", "3. マイクと出力先"), ("4. Voice model", "4. 声のモデル"),
    ("5. Preparation", "5. 必要ファイルの準備"),
    ("You can change this later in settings.", "言語はあとから設定で変更できます。"),
    ("Next", "次へ"), ("Terms of use", "利用条件"),
    ("Agree and continue", "同意して続ける"), ("Exit", "終了"),
    ("Review the terms for this app and its processing components. Original license texts are authoritative.", "アプリと処理部品の利用条件をご確認ください。規約は原文が適用されます。"),
    ("Processing component terms (NVIDIA / Intel / AMD / Qualcomm)", "処理部品の利用条件（NVIDIA / Intel / AMD / Qualcomm）"),
    ("Windows ML may obtain and update hardware-specific components. Their vendor terms also apply. Microsoft components may collect diagnostic data; see the privacy statement.", "Windows MLは、お使いの機器に対応する部品を取得・更新する場合があります。各メーカーの利用条件も適用されます。Microsoftの部品による診断情報の収集については、プライバシーステートメントをご確認ください。"),
    ("Microsoft privacy statement", "Microsoft プライバシーステートメント"),
    ("Component information", "部品の情報（原文）"),
    ("Check your microphone and headphones", "マイクとヘッドホンを確認しましょう"),
    ("Speak normally. Adjust the microphone volume so the meter does not frequently turn red.", "普段の声で話して、メーターが頻繁に赤くならないようマイク音量を調整してください。"),
    ("Microphone volume", "マイク音量"), ("Peak", "ピーク"),
    ("Checking microphone…", "マイクを確認中…"),
    ("Input is too loud. Lower the microphone volume.", "入力が大きすぎます。マイク音量を下げてください。"),
    ("Play test sound", "テスト音を鳴らす"), ("Hear my voice", "自分の声を聞く"),
    ("Use headphones to hear your voice without feedback. Sound starts only when you press a button.", "自分の声を聞くときは、ハウリングを防ぐためヘッドホンを使ってください。ボタンを押すまで音は出ません。"),
    ("Reconnect microphone", "マイクに再接続"),
    ("GTCRN noise reduction model is included: 352 KB, MIT license. Noise reduction will not be enabled automatically.", "ノイズ低減モデルGTCRNも取得します（352 KB・MITライセンス）。ノイズ低減は自動では有効になりません。"),
    ("Processing method", "処理方式"),
    ("Windows ML (automatic, recommended)", "Windows ML（自動・推奨）"),
    ("TensorRT (NVIDIA GPU)", "TensorRT（NVIDIA GPU）"),
    ("CUDA (NVIDIA GPU)", "CUDA（NVIDIA GPU）"),
    ("DirectML (GPU)", "DirectML（GPU）"),
    ("Windows ML selects an available accelerator automatically. Some processing may use the CPU.", "Windows MLが利用可能なアクセラレータを自動選択します。一部の処理ではCPUを使う場合があります。"),
    ("If conversion does not work well", "うまく動かない場合"),
    ("Try another processing method if conversion fails or audio breaks up. For missing models or device errors, go back to check those settings.", "変換に失敗したり音が途切れたりする場合は、別の処理方式を試せます。モデルの不足やデバイスのエラーは、戻って該当の設定を確認してください。"),
    ("This build does not include another processing method.", "このビルドには別の処理方式が含まれていません。"),
    ("CPU processing may be too slow for real-time conversion.", "CPUではリアルタイム変換に処理が間に合わない場合があります。"),
    ("Processing time / available time", "処理時間／使える時間"),
    ("There is little processing headroom. Audio may break up; try another processing method if needed.", "処理の余裕が少ない状態です。音が途切れる場合は、別の処理方式を試してください。"),
    ("If background noise bothers you", "ノイズが気になる場合"),
    ("Reduce background noise", "ノイズを低減する"),
    ("Reduces environmental noise. It can also change how your voice sounds.", "環境音を抑えます。声の聞こえ方も変わる場合があります。"),
    ("Your saved denoiser is preserved for conversion. This device test offers only noise reduction without extra downloads.", "保存済みのノイズ処理は音声変換で使用します。このテストでは追加ダウンロード不要のノイズ低減のみ試せます。"),
    ("Continue with these devices", "この入出力で次へ"),
    ("Prepare voice conversion", "音声変換に必要なファイルを準備します"),
    ("Details and existing files", "詳細・手持ちのファイルを指定"),
    ("Prepare the processing components for this PC. Windows ML may download vendor components; their size depends on your device.", "このPCに必要な処理部品を準備します。Windows MLがメーカーの部品を取得する場合があります。容量は機器によって異なります。"),
    ("Prepare processing components", "処理部品を準備する"),
    ("Downloading uses the GPL-3.0 model license. Completed files are reused.", "取得するモデルにはGPL-3.0が適用されます。取得済みのファイルは再利用します。"),
    ("Choose a voice later", "モデルはあとで選ぶ"), ("Try conversion later", "あとで試す"),
    (STEPS[0], "1. 声を選んで準備"), (STEPS[1], "2. マイクで試す"),
    (ADVANCED, "詳細設定"), (CHOOSE, "声のモデルを選ぶ…"),
    (SKIP, "スキップして通常画面へ"),
    (VOICE_TITLE, "声のモデルを選びましょう"),
    (VOICE_HELP, "RVCの声モデル（.pth）を選択するか、ここにドロップしてください。アプリで使える形式への変換をご案内します。"),
    (VOICE_COMPATIBILITY, "ContentVecで学習されたモデルは、このあとの案内で必要なファイルを準備できます。それ以外の特徴抽出モデルで学習された場合は、対応するファイルを別途用意する必要があります。"),
    (VOICE_ONNX, "変換済みの.onnxファイルも使用できます。"),
    ("File found", "選択済み（ファイルあり）"),
    ("File not found", "ファイルが見つかりません"),
    ("Choose .pth file…", ".pthファイルを選択…"),
    (MODEL_HELP, "声のモデルを持っていない場合"),
    (MODEL_GUIDE, "モデルの作者からRVCモデルを入手し、利用条件を確認してください。ONNXは次へ進む前に構造を確認します。内蔵コンバータはRVC v2 / F0のチェックポイントに対応しています。それ以外の .pth は対応する外部ツールで変換してください。"),
    (CHECKING, "モデルの構造を確認中…"),
    (CHECKED, "RVCの構造を確認しました。読み込みと音声出力は音声変換を開始したときに確認します。"),
    (NEXT, "次へ進む"), (BACK, "戻る"),
    (PREPARE_HELP, "音声変換に必要な共通ファイルと実行部品を確認します。"),
    (DOWNLOAD, "同意して必要なファイルを取得"),
    (DOWNLOAD_INFO, "ContentVec + RMVPE · 最大741 MB · GPL-3.0"),
    (SOURCE, "配布元・ライセンス"), (SUPPORT_READY, "補助モデルの準備ができました。"),
    ("Ready", "準備済み"),
    ("Download required", "ダウンロードが必要"),
    ("Setup is ready. Open the main screen to start voice conversion.", "準備できました。通常画面で音声変換を開始できます。"),
    (CUSTOM, "手持ちの補助モデルを使う"),
    ("Choose a valid voice model to continue.", "声のモデルを選び、確認が終わると先へ進めます。"),
    ("Download the required files above to continue.", "上のボタンから必要なファイルをダウンロードすると先へ進めます。"),
    (STOP, "停止"), (START, "開始"), (APPLY, "変更を反映 / 再起動"),
    (SETUP, "セットアップ"), (REFRESH, "デバイスを再取得"),
    (INPUT, "入力マイク"), (OUTPUT, "出力 / ヘッドホン"),
    (VOLUME, "出力音量"), (PITCH, "声の高さ（半音）"),
    (ROUTING, "Discordなどのアプリで使う"),
    (ROUTING_HELP, "仮想オーディオデバイスを未導入の場合はインストールしてください。本アプリの出力を仮想デバイスの再生側に、通話アプリのマイクを対応する録音側に設定します。名前はドライバによって異なります。通話アプリ側でもマイクテストを行ってください。vc-rsはドライバの導入や他のアプリの設定変更を自動では行いません。"),
    (CANCEL, "キャンセル"), (DISMISS, "閉じる"),
    (RETRY, "接続と空き容量を確認して再試行してください。取得済みのモデルは再利用します。"),
    (FINISH_ERROR, "設定を保存できませんでした。設定フォルダのアクセス権を確認して再試行してください。"),
    ("Checking the runtime…", "実行環境を確認中…"),
    ("Runtime preflight passed. Starting conversion will verify model loading and audio.", "実行環境の事前確認が完了しました。モデルの読み込みと音声は音声変換を開始したときに確認します。"),
    ("Runtime check could not complete.", "実行環境の確認を完了できませんでした。"),
    ("Runtime details", "実行環境の詳細"), ("Retry runtime check", "実行環境を再確認"),
    ("No voice model selected", "声のモデルが選択されていません"),
    ("Convert to ONNX…", "ONNXへ変換…"),
    ("Choose an RVC .onnx or .pth file.", "RVCの .onnx / .pth ファイルを選んでください。"),
    ("Microphone level", "マイク入力レベル"), ("Converted output level", "変換後の出力レベル"),
    ("Audio levels are available while preview or conversion is running.", "試聴・音声変換中に音声レベルを表示します。"),
    ("Use Next or Back to move through setup.", "「次へ」「戻る」でセットアップを進めます。"),
    ("Processing backend", "処理方式"),
    ("Windows App SDK runtime installation", "Windows App SDKランタイムのインストール"),
    ("Choose ContentVec…", "ContentVecを選ぶ…"), ("Choose RMVPE…", "RMVPEを選ぶ…"),
    ("Voice conversion", "音声変換"),
    ("Models are missing. Open Setup to select or download them.", "モデルが不足しています。セットアップで選択または取得してください。"),
    ("Convert .pth to ONNX", ".pthをONNXへ変換"),
    ("Source", "変換元 / 配布元"), ("Output", "出力先"),
    ("The existing file will be overwritten.", "既存のファイルを上書きします。"),
    ("Export mode", "出力形式"), ("Streaming (recommended)", "ストリーミング（推奨）"),
    ("WebUI-compatible", "WebUI互換"),
    ("Streaming exports carry NSF phase across chunks for the realtime engine.", "ストリーミング形式では、リアルタイム変換のチャンク間でNSF位相を引き継ぎます。"),
    ("Convert", "変換"), ("Retry", "再試行"), ("Close", "閉じる"),
    ("RVC model", "RVCモデル"), ("ONNX model", "ONNXモデル"), ("PyTorch checkpoint", "PyTorchチェックポイント"),
    ("Choose a voice model and prepare ContentVec / RMVPE in Setup.", "セットアップで声のモデルを選び、ContentVec / RMVPEを用意してください。"),
    ("Status", "動作状況"), ("Stopped", "停止中"), ("Starting", "準備中"), ("Running", "実行中"), ("Stopping", "停止処理中"), ("Error", "エラー"),
    ("Error details", "エラー詳細（技術情報は原文）"),
    ("Apply / Start", "開始 / 再起動"), ("Passthrough", "元の声をそのまま出力"),
    ("Live passthrough switching requires all three models; Apply / Start after selecting them.", "実行中のパススルー切り替えには3種類のモデルが必要です。モデルを選んで「開始」または「再起動」を押してください。"),
    ("Models", "モデル"), ("PyTorch checkpoints must be converted to ONNX first.", "PyTorchチェックポイントは先にONNXへ変換してください。"),
    ("Embedder", "埋め込みモデル"), ("F0 model", "F0推定モデル"),
    ("Provider", "処理方式"), ("OpenVINO Device", "OpenVINOデバイス"), ("Default", "既定"), ("GPU Priority", "GPU優先度"), ("high", "高"), ("normal", "通常"),
    ("Audio", "音声入出力"), ("Input backend", "入力方式"), ("Output backend", "出力方式"),
    ("ASIO uses one driver for both directions; pick the same device for input and output.", "ASIOは入出力で1つのドライバを使います。同じデバイスを選んでください。"),
    ("Input device", "入力デバイス"), ("Output device", "出力デバイス"),
    ("Engine configuration (Apply to restart)", "エンジン設定（反映時に再起動）"),
    ("Smoother", "チャンク接続方式"),
    ("Chunk ms", "チャンク長（ms）"),
    ("RVC: 10 ms steps, with integer samples at both device rates.", "RVCは10 ms刻みです。入出力のサンプル数が整数になる設定を使います。"),
    ("Extra convert ms", "追加変換時間（ms）"), ("Live parameters", "リアルタイム調整"),
    ("Pitch shift", "声の高さ"), ("Speaker ID", "話者ID"), ("Input gain", "入力音量"), ("Output gain", "出力音量"),
    ("Input denoiser", "入力ノイズ抑制"), ("off", "無効"), ("noise-gate", "ノイズゲート"),
    ("Gate threshold", "ゲートしきい値"), ("Gate attack (ms)", "ゲートアタック（ms）"),
    ("Gate release (ms)", "ゲートリリース（ms）"), ("Gate floor", "ゲート下限"),
    ("GTCRN model dir", "GTCRNモデルフォルダ"), ("Browse…", "参照…"), ("Browse", "参照"),
    ("GTCRN: ready (Apply / Start to activate)", "GTCRN: 準備完了（「開始」または「再起動」で反映）"),
    ("License", "ライセンス"), ("Download GTCRN", "GTCRNをダウンロード"),
    ("Download progress and cancellation are shown above.", "ダウンロードの進捗とキャンセルは上部に表示されます。"),
    ("Telemetry", "動作モニター"), ("Processing (total)", "全体処理時間"),
    ("Content delay (nominal)", "モデル由来の遅延（公称）"), ("Inference", "推論時間"),
    ("Input RMS", "入力RMS"), ("Output RMS", "出力RMS"), ("Input overruns", "入力オーバーラン"),
    ("Output underruns", "出力アンダーラン"), ("Dropped output samples", "破棄した出力サンプル数"),
    ("Output buffered samples", "出力バッファ内のサンプル数"),
    ("Content delay excludes devices, queues, chunk accumulation and processing time.", "モデル由来の遅延には、デバイス・キュー・チャンク蓄積・処理時間は含まれません。"),
    ("Unknown", "不明"), ("GPU Device", "GPUデバイス"), ("GPU Device ID: ", "GPUデバイスID: "),
    ("GPU enumeration failed", "GPU一覧の取得に失敗しました"), ("Detecting CUDA devices...", "CUDAデバイスを検出中…"),
    ("Unavailable: device", "利用できないデバイス"), ("System default", "システムの既定"),
    ("Preparing", "準備中"), ("Download cancelled. You can retry.", "ダウンロードを中止しました。再試行できます。"),
    ("Unexpected model size.", "モデルのサイズが一致しません。"),
    ("Model verification failed. Please retry the download.", "モデルの検証に失敗しました。もう一度ダウンロードしてください。"),
    ("Stopping previous session", "前のセッションを停止中"), ("Validating configuration", "設定を確認中"),
    ("Opening audio devices", "音声デバイスを開いています"), ("Loading RVC model", "RVCモデルを読み込み中"),
    ("Reading checkpoint…", "チェックポイントを読み込み中…"),
    ("Parsing checkpoint…", "チェックポイントを解析中…"),
    ("Building ONNX graph…", "ONNXグラフを構築中…"),
    ("Writing model…", "モデルを書き込み中…"),
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn catalog_has_unique_nonempty_keys_and_translations() {
        let mut keys = std::collections::HashSet::new();
        for &(en, ja) in TRANSLATIONS {
            assert!(!en.is_empty() && !ja.is_empty());
            assert!(keys.insert(en), "Duplicate translation: {en}");
            assert_eq!(Language::English.text(en), en);
            assert_eq!(Language::Japanese.text(en), ja);
        }
    }
    #[test]
    fn locale_roundtrips_and_unknown_locale_falls_back_to_english() {
        #[derive(Serialize, Deserialize)]
        struct Config {
            language: Language,
        }
        for language in [Language::English, Language::Japanese] {
            let restored: Config =
                toml::from_str(&toml::to_string(&Config { language }).unwrap()).unwrap();
            assert_eq!(restored.language, language);
        }
        let unknown: Config = toml::from_str("language = 'future-locale'").unwrap();
        assert_eq!(unknown.language, Language::English);
    }
    #[test]
    fn diagnostics_and_names_are_not_rewritten_as_ui_tokens() {
        assert_eq!(
            Language::Japanese.text("C:/models/voice.onnx"),
            "C:/models/voice.onnx"
        );
        assert_eq!(
            engine_message(Language::Japanese, "Loading RMVPE model"),
            "RMVPEモデルを読み込み中"
        );
        assert_eq!(
            engine_message(Language::Japanese, "Driver error: 0x80670016"),
            "Driver error: 0x80670016"
        );
    }
}
