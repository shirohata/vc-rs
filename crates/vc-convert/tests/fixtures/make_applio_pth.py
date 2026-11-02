#!/usr/bin/env python3
"""Make synthetic (untrained) Applio topology fixtures, without torch.

Uses the same ZIP/pickle writer and encoder/flow fixture as make_tiny_pth.py.
Decoder widths are scaled down, but all four stages, residual branches,
harmonics, learned noise gains and weight-normalization layouts are retained.
Applio reference: IAHispano/Applio 5171e1dcf31a33d1afeb14ef7c6aac6dadae5dc1.
"""
from pathlib import Path
import make_tiny_pth as f


def conv(w, base, out_ch, in_ch, kernel, normalized=False, bias=True):
    weight = f.t([out_ch, in_ch, kernel])
    if normalized:
        w.update(f.wn(base, weight))
    else:
        w[base + '.weight'] = weight
    if bias:
        w[base + '.bias'] = f.t([out_ch])


def make_mrf(w):
    conv(w, 'dec.conv_pre', f.UPS_INIT, f.INTER, 7, True)
    conv(w, 'dec.cond', f.UPS_INIT, f.GIN, 1)
    w['dec.m_source.l_linear.weight'] = f.t([1, 9])
    w['dec.m_source.l_linear.bias'] = f.t([1])
    for i, (rate, kernel) in enumerate(zip(f.UPS, f.UPS_K)):
        ch = f.UPS_INIT // 2**(i+1)
        w.update(f.wn(f'dec.upsamples.{i}', f.t([ch*2, ch, kernel])))
        w[f'dec.upsamples.{i}.bias'] = f.t([ch])
        stride = 1
        for r in f.UPS[i+1:]:
            stride *= r
        conv(w, f'dec.noise_convs.{i}', ch, 1, 1 if stride == 1 else 2*stride-stride%2)
        for j, (k, ds) in enumerate(zip(f.RES_K, f.RES_D)):
            for l, _ in enumerate(ds):
                for n in (1, 2):
                    conv(w, f'dec.mrfs.{i}.{j}.layers.{l}.conv{n}', ch, ch, k, True)
    conv(w, 'dec.conv_post', 1, ch, 7, True)


def make_refine(w):
    conv(w, 'dec.pre_conv', 1, 1, 7, True)
    w['dec.m_source.merge.0.weight'] = f.t([1, 1], values=[0.7])
    ch = 1
    for i in range(4):
        conv(w, f'dec.downsample_blocks.{i}', ch*2, ch, 7, True)
        ch *= 2
    conv(w, 'dec.mel_conv', 16, f.INTER, 7, True)
    conv(w, 'dec.cond', 16, f.GIN, 1)
    ch = 32
    for i in range(4):
        base = f'dec.upsample_conv_blocks.{i}'
        out = ch//2
        conv(w, base+'.input_conv', out, ch+ch//4, 7)
        for j, k in enumerate((3, 7, 11)):
            for n in (0, 2):
                w[f'{base}.blocks.{j}.{n}.weight'] = f.t([out], values=[0.025]*out)
            for l in range(3):
                for n in (1, 2):
                    conv(w, f'{base}.blocks.{j}.1.convs{n}.{l}', out, out, k, True)
        ch = out
    conv(w, 'dec.conv_post', 1, ch, 7, True, False)


def main():
    for vocoder, name, build in [('MRF HiFi-GAN', 'mrf', make_mrf), ('RefineGAN', 'refine', make_refine)]:
        f._seed = 0x12345678
        f.FakeTensor._next_key = 0
        w = {k: v for k, v in f.build_weights().items() if not k.startswith('dec.')}
        build(w)
        config = list(f.CONFIG)
        if name == 'refine':
            # Match the official UI's primary RefineGAN rate and exercise sinc
            # ratios different from the 40 kHz MRF fixture.
            config[12] = [10, 8, 2, 2]
            config[14] = [20, 16, 4, 4]
            config[17] = 32000
        path = Path(__file__).parent / f'tiny_applio_{name}.pth'
        f.write_pth(path, dict(config=config, weight=w, f0=True, version='v2', vocoder=vocoder,
                              info='Synthetic untrained scaled Applio decoder fixture'))
        print(path.name, path.stat().st_size)


if __name__ == '__main__':
    main()
