# Fourier Series Coder (FSC)

This repository provides a minimal MMRotate add-on for the Fourier Series
Coder (FSC), a lightweight angle representation for oriented object detection.

The release intentionally contains only the core method files and one DOTA
training configuration. It is designed to be copied on top of an existing
MMRotate 1.x checkout.

## Files

```text
configs/fsc-retinanet-r50_fpn_3x_dota.py
mmrotate/models/dense_heads/fourier_series_retina_head.py
mmrotate/models/task_modules/coders/fst_coder.py
```

## Usage

1. Install MMRotate 1.x following the official instructions.
2. Copy the `configs/` and `mmrotate/` folders in this repository into the
   MMRotate root directory.
3. Prepare DOTA in the standard MMRotate format.
4. Train:

```bash
python tools/train.py configs/fsc-retinanet-r50_fpn_3x_dota.py
```

## Citation

If this code helps your research, please cite our paper:

```bibtex
@article{wei2026fsc,
  title={Fourier Series Coder: A Novel Perspective on Angle Boundary Discontinuity Problem for Oriented Object Detection},
  author={Wei, Minghong and Cao, Pu and Chen, Zhihao and Zang, Zhiyuan and Yang, Lu and Song, Qing},
  year={2026}
}
```
