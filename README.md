# Fourier Series Coder

This repository contains the official minimal release of **Fourier Series
Coder (FSC)** for oriented object detection.

FSC is a lightweight angle coder designed to alleviate the angle boundary
discontinuity and cyclic ambiguity problems in rotated object detection. It
maps the angle into Fourier components and decodes the final orientation with a
continuous inverse mapping. The released code keeps only the core method and a
reproducible DOTA training configuration based on MMRotate.

Paper: **Fourier Series Coder: A Novel Perspective on Angle Boundary
Discontinuity Problem for Oriented Object Detection**

arXiv: https://arxiv.org/abs/2604.20281

## News

- The minimal MMRotate implementation of FSC is released.
- DOTA and HRSC pretrained weights will be released soon.

## Method

For an oriented bounding box angle `theta`, FSC encodes the angle as Fourier
components:

```text
FSC(theta) = [a0, cos(omega * theta), sin(omega * theta),
              cos(2 * omega * theta), sin(2 * omega * theta), ...]
```

For the common `le90` angle definition, `omega=2` maps the pi-periodic box
angle into a full phase cycle. During training, FSC supervises the Fourier
components directly and adds a unit-circle manifold constraint on each
`cos/sin` pair:

```text
cos^2 + sin^2 ~= 1
```

During inference, the Fourier components are decoded back to the final angle
with `atan2`, and the standard rotated bounding box post-processing pipeline is
kept unchanged.

## Repository Structure

This release is intentionally small. It contains only the method files needed
to run FSC on top of MMRotate.

```text
configs/
  fsc-retinanet-r50_fpn_3x_dota.py
mmrotate/
  models/
    dense_heads/
      fourier_series_retina_head.py
    task_modules/
      coders/
        fst_coder.py
```

The DOTA config uses `custom_imports`, so users do not need to edit MMRotate
registry files manually.

## Installation

First install MMRotate 1.x and its dependencies following the official
instructions:

```bash
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -v -e .
```

Then copy this repository's `configs/` and `mmrotate/` folders into the
MMRotate root directory:

```bash
cp -r /path/to/FSC/configs ./
cp -r /path/to/FSC/mmrotate ./
```

On Windows, copy the same folders into the MMRotate root directory with File
Explorer or PowerShell.

## Data Preparation

Prepare DOTA in the standard MMRotate format. The default config expects the
same dataset layout as MMRotate's official DOTA configs.

Please refer to the MMRotate documentation for dataset conversion and image
splitting details.

## Training

Train FSC-RetinaNet on DOTA:

```bash
python tools/train.py configs/fsc-retinanet-r50_fpn_3x_dota.py
```

The released configuration uses:

- Detector: RetinaNet
- Backbone: ResNet-50
- Neck: FPN
- Dataset: DOTA-v1.0
- Angle version: `le90`
- Fourier size: `2`
- Base frequency: `omega=2`

## Evaluation

After training, evaluate with the standard MMRotate test script:

```bash
python tools/test.py \
  configs/fsc-retinanet-r50_fpn_3x_dota.py \
  /path/to/checkpoint.pth
```

## Model Zoo

Pretrained weights will be released soon.

| Model | Dataset | Backbone | Config | Weights |
| --- | --- | --- | --- | --- |
| FSC-RetinaNet | DOTA-v1.0 | R-50-FPN | `configs/fsc-retinanet-r50_fpn_3x_dota.py` | Coming soon |
| FSC-RetinaNet | HRSC-2016 | R-50-FPN | Coming soon | Coming soon |

## Notes

- This repository is a minimal method release rather than a full fork of
  MMRotate.
- No private paths, local checkpoints, visualization scripts, or intermediate
  experiment files are included.
- If you need to reproduce the exact paper tables, please use the released
  checkpoints after they become available.

## Citation

If this code helps your research, please cite:

```bibtex
@article{wei2026fourier,
  title={Fourier Series Coder: A Novel Perspective on Angle Boundary Discontinuity Problem for Oriented Object Detection},
  author={Wei, Minghong and Cao, Pu and Chen, Zhihao and Zang, Zhiyuan and Yang, Lu and Song, Qing},
  journal={arXiv preprint arXiv:2604.20281},
  year={2026}
}
```

## License

This project is released under the Apache License 2.0.
