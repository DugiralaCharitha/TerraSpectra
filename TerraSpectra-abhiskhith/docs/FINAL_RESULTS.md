# Final Model Results

## Best Model
Day 2 Regression V3 — 3D CNN

- Validation MAE: 23.664
- Validation RMSE: 27.641
- Validation R²: 0.0031
- Spatial size: 32 × 32
- Spectral bands: 125
- Training samples: 1592
- Validation samples: 398
- Random seed: 42
- Device: CPU

## Dataset
- Usable labelled samples: 1,990
- Classes/labels: 101
- Cube shape: 128 × 128 × 125
- Missing/invalid samples after cleaning: 0

## Experiments
- Original 3D CNN: ~26 MAE
- ExtraTrees spectral baseline: ~25 MAE
- Day 2 Regression V2: 23.677 MAE
- Day 2 Regression V3: 23.664 MAE
- 64×64 CNN: stopped after epoch 1 because CPU training was impractical
- Hybrid spectral ExtraTrees: 24.985 MAE

## Conclusion
The 32×32 3D CNN produced the best validated regression result obtained in the current dataset experiments. The near-zero R² indicates that the available anonymous numeric labels have weak predictive relationship with the extracted hyperspectral information. Therefore, further accuracy claims should not be made without recovering authoritative dataset provenance and label meaning.
