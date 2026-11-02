"""Compare decoder exports with a local pinned Applio source tree.

No model download or training. Requires torch, torchaudio and onnxruntime CPU.
Export the three fixtures with vc-convert's export_fixture example first.
Reference source: IAHispano/Applio 5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1.
The small RefineGAN fixture scales constructor widths and speaker conditioning;
the layers, sinc parameters and all stochastic operations remain upstream code.
"""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import onnx
import torch


def compare(source, exports, output, checkpoint_path=None, onnx_path=None):
    manifest = json.loads(Path(__file__).with_name('applio-reference.json').read_text())
    for path, expected_hash in manifest['normalized_source_sha256'].items():
        data = (source / path).read_text(encoding='utf-8').rstrip().encode()
        assert hashlib.sha256(data).hexdigest() == expected_hash, f'Wrong Applio reference source: {path}'
    sys.path.insert(0, str(source.resolve()))
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.lib.algorithm.generators.refinegan import RefineGANGenerator

    torch.set_num_threads(2)
    results = []
    fixtures = Path(__file__).resolve().parents[1] / 'crates/vc-convert/tests/fixtures'
    cases = [('hifi', 'tiny_v2_f0.pth'), ('mrf', 'tiny_applio_mrf.pth'), ('refine', 'tiny_applio_refine.pth')]
    if checkpoint_path is not None:
        assert onnx_path is not None, '--onnx required with --checkpoint'
        cases = [(None, checkpoint_path)]
    for name, checkpoint in cases:
        # The existing dependency-free fixture writer omits Torch's archive
        # version record. Add that bookkeeping in memory for the reference read.
        input_path = checkpoint if checkpoint_path is not None else fixtures / checkpoint
        archive = io.BytesIO(input_path.read_bytes())
        with zipfile.ZipFile(archive, 'a') as z:
            if not any(n.endswith('/version') for n in z.namelist()):
                z.writestr('archive/version', '3\n')
        archive.seek(0)
        c = torch.load(archive, weights_only=True, map_location='cpu')
        # Public inference exports use the full upstream constructor. Only the
        # checked-in RefineGAN fixture needs the deliberately reduced widths.
        vocoder = c.get('vocoder', 'HiFi-GAN')
        name = {'HiFi-GAN': 'hifi', 'MRF HiFi-GAN': 'mrf', 'RefineGAN': 'refine'}[vocoder]
        c['config'][15] = c['weight']['emb_g.weight'].shape[0]
        model = Synthesizer(*c['config'], use_f0=True, vocoder=vocoder)
        if name == 'refine' and checkpoint_path is None:
            model.dec = RefineGANGenerator(sample_rate=c['config'][17], upsample_rates=c['config'][12],
                num_mels=c['config'][2], start_channels=1, upsample_initial_channel=32)
            model.dec.pre_conv = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(1, 1, 7, padding=3))
            model.dec.cond = torch.nn.Conv1d(c['config'][16], 16, 1)
        missing, unexpected = model.load_state_dict(c['weight'], strict=False)
        assert all(key.startswith('enc_q.') for key in missing), missing
        assert not unexpected, unexpected
        model.eval()
        graph = onnx.load(onnx_path if checkpoint_path is not None else exports / f'tiny_applio_{name}.onnx')
        source_tensor = next(n.output[0] for n in graph.graph.node if n.op_type == 'Transpose' and
            n.output[0].startswith(('nsf_source_', 'source_')))
        graph.graph.output.append(onnx.helper.make_tensor_value_info(source_tensor, onnx.TensorProto.FLOAT, [1, 1, None]))
        session = ort.InferenceSession(graph.SerializeToString(), providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        phases = int(metadata.get('rvc.phase_channels', 1))
        hop = int(metadata['rvc.frame_hop'])
        noise_hop = int(metadata.get('rvc.noise_frame_hop', hop))
        for frames in (2, 5, 12):
            for contour in ('voiced', 'glide', 'unvoiced'):
                rng = np.random.default_rng(1234 + frames)
                phone = rng.normal(0, .2, (1, frames, 768)).astype('float32')
                rnd = rng.normal(size=(1, c['config'][2], frames)).astype('float32')
                noise = rng.normal(size=(frames, noise_hop)).astype('float32')
                phase = np.linspace(0, .8, phases, dtype='float32').reshape(1, 1, phases)
                f0 = np.full((1, frames), 113., dtype='float32')
                if contour == 'glide':
                    f0[0] = np.linspace(75, 450, frames)
                if contour == 'unvoiced':
                    f0[0, ::2] = 0
                pitch = np.full((1, frames), 100, dtype='int64')
                feeds = dict(phone=phone, phone_lengths=np.array([frames], dtype='int64'),
                             pitch=pitch, pitchf=f0, ds=np.array([0], dtype='int64'), rnd=rnd,
                             nsf_noise=noise.reshape(1, -1, 1), phase_in=phase)
                state = dict(offset=0, latent=False)

                def normal_like(x, **kwargs):
                    if not state['latent']:
                        state['latent'] = True
                        assert tuple(x.shape) == rnd.shape
                        return torch.from_numpy(rnd.copy()).to(x)
                    if x.shape[-1] == phases and x.shape[1] == frames * hop and state['offset'] == 0:
                        width = hop * phases
                        values = noise[:, :width].reshape(1, frames * hop, phases)
                    else:
                        channels, rate = x.shape[1], x.shape[2] // frames
                        width = channels * rate
                        start = state['offset']
                        values = noise[:, start:start+width].reshape(1, frames * rate, channels).transpose(0, 2, 1)
                    state['offset'] += width
                    return torch.from_numpy(values.copy()).to(x)

                def uniform(*shape, **kwargs):
                    if isinstance(shape[0], (list, tuple)):
                        shape = shape[0]
                    value = np.zeros(shape, dtype='float32')
                    if name == 'mrf':
                        value[:] = phase.reshape(1, phases)
                    return torch.from_numpy(value).to(device=kwargs.get('device', 'cpu'))

                source_values = []
                def source_hook(module, inputs, result):
                    value = result if name == 'refine' else result[0]
                    source_values.append(value.transpose(1, 2).detach().numpy())
                hook = model.dec.m_source.register_forward_hook(source_hook)
                with torch.inference_mode(), patch('torch.randn_like', normal_like), patch('torch.rand', uniform):
                    expected = model.infer(torch.from_numpy(phone), torch.tensor([frames]),
                        torch.from_numpy(pitch), torch.from_numpy(f0), torch.tensor([0]))[0].numpy()
                hook.remove()
                assert state['offset'] == noise_hop, (name, state, noise_hop)
                actual, actual_source = session.run(['audio', source_tensor], feeds)
                source_error = float(np.max(np.abs(source_values[0] - actual_source)))
                error = float(np.max(np.abs(expected - actual)))
                mean = float(np.mean(np.abs(expected - actual)))
                assert np.isfinite(actual).all()
                assert actual.shape == expected.shape
                results.append(dict(vocoder=name, frames=frames, contour=contour, max_abs=error, mean_abs=mean,
                                    source_max_abs=source_error, samples=int(actual.size), peak=float(np.max(np.abs(actual))),
                                    passed=source_error <= 1e-4 and error <= 1e-4 and mean <= 1e-5))
    report = dict(source_revision='5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1', torch=torch.__version__,
                  onnxruntime=ort.__version__, results=results)
    if checkpoint_path is not None:
        report['checkpoint_sha256'] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        report['vocoder'] = vocoder
        report['sample_rate'] = c['config'][17]
        report['embedder_model'] = c.get('embedder_model')
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    # Retain every case's diagnostics even when a trained model exceeds the
    # fixed numerical budget. Do not silently loosen fixture tolerances.
    assert all(row['passed'] for row in results), f'Numerical comparison failed; see {output}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--exports', type=Path, default=Path('target'))
    parser.add_argument('--output', type=Path, default=Path('target/applio-validation.json'))
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--onnx', type=Path)
    args = parser.parse_args()
    compare(args.source, args.exports, args.output, args.checkpoint, args.onnx)
