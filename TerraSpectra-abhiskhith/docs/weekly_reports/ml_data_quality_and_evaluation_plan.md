# ML data-quality findings and evaluation plan

## Current evidence

The checked-in labels and the local, ignored training subset have the following
lineage:

| File | Labelled rows | Classes |
| --- | ---: | ---: |
| `train.csv` | 2,177 | 101 |
| `train_clean.csv` | 2,176 | 101 |
| `train_final.csv` (local only) | 1,990 | 101 |

The 1,990 rows in `train_final.csv` have unique, non-empty identifiers; every
referenced file exists. A sequential, memory-safe audit found that all 1,990
cubes have shape `(128, 128, 125)`, no cube is blank, and no cube contains a
non-finite value. Foreground coverage nevertheless varies widely, from 59 to
16,384 spatial pixels, so foreground coverage should be retained as an audit
field for future data additions.

The retained label distribution is nearly balanced but too small for the
number of classes: 9--32 examples per class (median 19). An 80/20 split gives
roughly 2--6 validation examples per class. This makes a single validation
accuracy noisy and is the most credible current explanation for poor
generalization.

`train_final.csv` is intentionally local and is now ignored explicitly. It
must not be committed or used as an exchange mechanism for labels.

## Baseline status

All reported models are close to the 101-class chance accuracy of about 0.99%:

| Model | Best observed validation accuracy |
| --- | ---: |
| Mean-spectrum logistic regression | 1.01% |
| Statistical-feature logistic regression | 1.01% |
| Memory-safe 3D CNN | 2.02% |
| Tuned 3D CNN (initial 3 epochs) | 1.51% |
| Compact hyperspectral ViT (3 epochs) | 1.51% |

The predictor module is an integration handoff for the saved 3D-CNN baseline,
not evidence of production-quality predictions. Its reported confidence should
not be presented as calibrated confidence.

## Next milestone: evaluate before scaling a model

1. Obtain additional labelled cubes, prioritising classes with fewer than 20
   retained examples and recording source, collection conditions, and label
   provenance. Do not mix uncertain labels into the benchmark without a
   review flag.
2. Freeze a versioned label manifest outside Git if it is sensitive or too
   large, and record its row count, class counts, and file checksum in an
   experiment note. Keep raw cubes and checkpoints unchanged.
3. Run a stratified 5-fold evaluation with fixed folds for the mean-spectrum
   baseline, statistical baseline, and a memory-safe spatial model. Report
   mean and standard deviation for accuracy, macro F1, weighted F1, and
   per-class recall; retain confusion matrices.
4. Use a majority-class and a random-class baseline alongside each model.
   Promote a model only if its cross-validation macro F1 clearly exceeds both
   baselines and is stable across folds. Do not select models on a single
   80/20 split.
5. Before a longer CPU training run, perform a small overfit check on a fixed
   subset and inspect the label/cube pairs for classes with consistently poor
   recall. Continue to spatially pool or patch cubes before batching; never
   batch original `128x128x125` cubes directly.

## Reproducibility safeguards

- Use the existing fixed seed and store the split/fold identifiers with each
  experiment.
- Load and inspect cubes sequentially during audits to keep CPU memory bounded.
- Write new results to a new experiment directory; never overwrite the
  existing `3dcnn`, `3dcnn_tuned`, or `vit` artifacts.
- Backend code should import `HyperspectralPredictor` rather than duplicate
  preprocessing. There is no backend implementation on this branch to change.
