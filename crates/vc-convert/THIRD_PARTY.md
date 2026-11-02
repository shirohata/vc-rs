# Third-party source notices

`src/graph/applio.rs` implements Applio's MRF HiFi-GAN and RefineGAN decoder
algorithms from [IAHispano/Applio](https://github.com/IAHispano/Applio/tree/5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1),
Copyright (c) 2026 AI Hispano, MIT. See [LICENSE-Applio.txt](LICENSE-Applio.txt).
The integer Kaiser sinc kernel follows torchaudio 2.7.1's resampling algorithm,
Copyright (c) 2017 Facebook Inc. (Soumith Chintala), BSD-2-Clause; see
[LICENSE-torchaudio.txt](LICENSE-torchaudio.txt). These source notices must
accompany distributions containing the converter; no Python runtime is shipped.

## rvc-onnx-web

The `vc-convert` crate is a Rust port of
[rvc-onnx-web](https://github.com/visgotti/rvc-onnx-web) by Joseph Viscardi
(visgotti), used under the MIT License. Each source module's header names the
TypeScript file it ports.

> Note: the upstream `LICENSE` file carries a placeholder copyright line
> ("Your Name"); the project README states "MIT © Joseph Viscardi".

```
MIT License

Copyright (c) 2024 Joseph Viscardi (visgotti)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
