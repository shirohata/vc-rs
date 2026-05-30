# Third-Party Notices

`vc-rs` is an independently written Rust implementation for RVC-compatible
ONNX runtime behavior. The implementation and compatibility checks were
informed by public RVC ecosystem projects, but this repository does not vendor
their source files or pretrained model weights.

If a future change copies, translates, or includes substantial portions of code
from these or other third-party projects, keep the corresponding upstream
copyright and license notices with that code.

## RVC WebUI

- Repository: <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
- License: MIT
- Upstream license copyright notices include liujing04, 源文雨, and Ftps.

## VCClient / w-okada voice-changer

- Repository: <https://github.com/w-okada/voice-changer>
- License: MIT
- Upstream license copyright notices include Wataru Okada, Isle Tennos,
  Jaehyeon Kim, liujing04, 源文雨, and yxlllc.

## Applio

- Repository: <https://github.com/IAHispano/Applio>
- License: MIT
- Upstream license copyright notices include AI Hispano.

## External Model Weights

Pretrained model weights are not included in this repository. The optional
`download-models.ps1` helper downloads third-party model files from
<https://huggingface.co/wok000/weights_gpl>. Those downloaded files are outside
the scope of this repository's MIT License; review the upstream model license
before using, modifying, or redistributing them.
