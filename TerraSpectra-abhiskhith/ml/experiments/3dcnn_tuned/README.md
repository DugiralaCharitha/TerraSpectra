# Week 3 3D-CNN tuning experiment

This experiment preserves the Week 2 baseline in `../3dcnn/` and tests a
larger memory-safe model independently. Cubes remain spatially pooled from
128x128 to 32x32 while retaining all 125 spectral bands.

## Configuration

- 1,990 usable samples, 101 classes, stratified 80/20 split (seed 42)
- 16/32/64 3D-convolution channels (76,453 parameters)
- Random horizontal and vertical spatial flips on training data
- AdamW optimizer with `ReduceLROnPlateau` scheduling and early stopping
- CPU training, batch size 4

## Initial comparison

| Experiment | Epochs | Best validation accuracy |
| --- | ---: | ---: |
| Week 2 baseline | 10 | 2.015% |
| Week 3 tuned trial | 3 | 1.511% |

The tuned architecture did not improve on the baseline in its initial run.
A 32-sample overfit diagnostic did rise from 0% to 21.9% training accuracy in
20 epochs, confirming that the data loader, labels, loss, gradients, and model
updates work. The limiting issue is held-out generalization with roughly
16–32 samples per class, rather than a broken training pipeline.

## Reproduce a longer tuned run

```powershell
py -3.12 ml/training/train_3dcnn.py --epochs 30 --batch-size 4
```

This writes `best_model.pt` and `metrics.json` in this directory without
overwriting the Week 2 baseline artifacts.
