# TerraSpectra ML — Week 2 Completion Report

## Scope

This report records safe validation of the existing hyperspectral ML pipeline.
No raw cube, cleaned CSV, checkpoint, existing experiment result, Git state, or
Python installation was changed while producing it.

## Verified pipeline state

- `train_final.csv`: 1,990 unique labelled rows.
- Labels: integer IDs 0 through 100, with 101 classes. Their semantic meanings
  remain undocumented and were not inferred.
- Each referenced cube exists, has shape `(128, 128, 125)`, and has no blank or
  non-finite sample remaining.
- The memory-safe 3D CNN loads `(batch, 1, 125, 32, 32)` tensors and produces
  `(batch, 101)` logits.
- A finite loss and finite, nonzero gradients were verified.
- A 16-example, four-class, in-memory overfit test reached 100% training
  accuracy at epoch 107. This shows the current data loading, tensor layout,
  101-class loss, and optimisation path can learn a controlled subset.

## Environment

- Python 3.12: PyTorch 2.13.0 CPU works; pandas and scikit-learn are absent.
- Python 3.14: PyTorch 2.13.0 CPU, NumPy, pandas, and scikit-learn all import
  successfully.
- No GPU is available.
- The repository has no project-specific VS Code interpreter configuration.

## Evaluation

Mean-spectrum logistic regression was evaluated with stratified five-fold
cross-validation on the 1,990 usable samples. The scaler was fit within each
training fold to avoid validation leakage.

| Metric | Mean | Sample standard deviation |
| --- | ---: | ---: |
| Accuracy | 1.156% | 0.381% |
| Balanced accuracy | 0.918% | 0.230% |
| Macro F1 | 0.460% | 0.171% |
| Weighted F1 | 0.549% | 0.244% |

Reference baselines over the same folds:

| Baseline | Accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Majority class | 1.508% | 0.990% | 0.029% |
| Uniform random class | 1.055% | 1.003% | 0.964% |

The spectral logistic baseline is not meaningfully better than chance or the
majority baseline. Earlier exploratory single-split results are likewise weak:
3D CNN best validation accuracy 2.02%, tuned 3D CNN 1.51%, and ViT 1.51%.
Those exploratory results are not cross-validated and must not be represented
as reliable model performance.

## Normalisation finding

The current CNN divides raw values by 4,095. Across all usable cubes, 1,222 of
1,990 cubes have a maximum above 4,095; per-cube maxima have median 4,367.5,
95th percentile 8,593.75, and maximum 28,906. This creates no non-finite
values, and the overfit test passed, but the fixed divisor is not demonstrably
well matched to the dataset. Future CNN evaluation should fit a robust scale
using only each training fold, then apply it unchanged to that fold's
validation data.

## Week 2 conclusion

Week 2 can be treated as complete as an implementation and validation
milestone: preprocessing, traditional and spatial baselines, pipeline sanity
checks, controlled overfitting, and stratified cross-validation have been
performed. The evidence does not support claiming a useful 101-class model.

The appropriate Week 3 focus is data/label provenance and a carefully
normalised, reproducible evaluation protocol—not a larger neural network or
invented label meanings.
