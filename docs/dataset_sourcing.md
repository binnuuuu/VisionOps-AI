# Dataset Sourcing Notes

This project starts with normal-only anomaly detection because real blister defect data is scarce and the reference machine workflow learns from a small set of good blisters.

## Recommended Order

1. **VisA - Visual Anomaly**
   - Source: https://registry.opendata.aws/visa/
   - Project docs: https://github.com/amazon-science/spot-diff
   - License: CC BY 4.0
   - Why it matters: Includes a `capsules` subset, normal/anomaly images, and pixel-level masks. It is the best first public dataset for this project because it supports anomaly detection and segmentation.
   - Command:
     ```bash
     python scripts/source_datasets.py download visa --extract
     ```

2. **MVTec AD**
   - Source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
   - License: CC BY-NC-SA 4.0
   - Why it matters: Industrial anomaly benchmark with `capsule`, `pill`, and `bottle` categories. Very useful for PatchCore/PaDiM baselines, but not suitable for commercial use without checking license constraints.
   - Command after manual download:
     ```bash
     python scripts/source_datasets.py download mvtec_ad --archive /path/to/mvtec_anomaly_detection.tar.xz --extract
     ```

3. **Roboflow Larger Blister Pack Defect**
   - Source: https://universe.roboflow.com/my-workspace-d5mot/larger-blister-pack-defect
   - Why it matters: Blister-pack-specific object detection labels. Use for YOLO after setting a Roboflow API key and checking export/license terms.
   - Command:
     ```bash
     export ROBOFLOW_API_KEY=...
     python scripts/source_datasets.py download roboflow_larger_blister_pack_defect --extract
     ```

4. **Roboflow Blister Strips' defect detection**
   - Source: https://universe.roboflow.com/blister-strips/blister-strips-defect-detection
   - Why it matters: Additional blister strip defect labels for supervised detection experiments.

5. **Roboflow Blister Pack Defects**
   - Source: https://universe.roboflow.com/my-workspace-d5mot/blister-pack-defects
   - Why it matters: Small but quick YOLO smoke-test dataset with blister defect labels.

## Model Path

- Start with the current normal-sample model for the MVP loop.
- Add PatchCore or PaDiM using VisA/MVTec normal samples.
- Add YOLO using Roboflow blister datasets.
- Replace or fine-tune thresholds once real line images arrive.

