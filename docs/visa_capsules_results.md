# VisA Capsules Training Results

Dataset:

- Source: VisA `capsules`
- Split file: `datasets/raw/visa/split_csv/1cls.csv`
- Training: 542 normal images
- Test: 60 normal images, 100 anomaly images

Runs completed:

| Run | Calibration | Threshold | AUROC | F1 | Precision | Recall | Specificity | TP | TN | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visa_capsules_patchcore_lite` | conservative, normal holdout | 1.628164 | 0.8385 | 0.373984 | 1.0 | 0.23 | 1.0 | 23 | 60 | 0 | 77 |
| `visa_capsules_patchcore_lite_sensitive` | 75th percentile normal holdout | 1.365184 | 0.8385 | 0.77095 | 0.873418 | 0.69 | 0.833333 | 69 | 50 | 10 | 31 |
| `visa_capsules_patchcore_lite_hires_sensitive` | high-res, 75th percentile normal holdout | 1.361949 | 0.877333 | 0.808989 | 0.923077 | 0.72 | 0.9 | 72 | 54 | 6 | 28 |

Deep-feature Anomalib run:

| Run | Model | Backbone | Image size | Coreset | Image AUROC | Image F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `visa_capsules_anomalib_patchcore_resnet18` | PatchCore | `resnet18`, `layer2+layer3` | 192 | 0.02 | 0.814 | 0.78125 |

Current best baseline:

- `visa_capsules_patchcore_lite_hires_sensitive`
- Model: `data/training_runs/visa_capsules_patchcore_lite_hires_sensitive/patchcore_lite.npz`
- Heatmaps: `data/training_runs/visa_capsules_patchcore_lite_hires_sensitive/evaluation/heatmaps/`
- Deep-feature checkpoint: `data/anomalib_runs/visa_capsules_anomalib_patchcore_resnet18/Patchcore/visa_capsules/v0/weights/lightning/model.ckpt`

Interpretation:

- The high-resolution PatchCore-lite baseline is useful enough for the platform prototype and catches most visible capsule anomalies in VisA.
- The Anomalib Docker path is now working and should be tuned further on GPU or real blister-line images.
- The operating threshold was calibrated from normal holdout images, not from test labels.
