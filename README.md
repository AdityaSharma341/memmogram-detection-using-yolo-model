# Mammography YOLO — Dataset & Training

## Project Overview
This repository contains dataset utilities, example Yolo models, and scripts used to train and run object detection on mammography images.

**Key goals:**
- **Prepare** the DDSM-style dataset and convert DICOM to JPEG/labels
- **Train** a YOLOv8 model on the resulting dataset
- **Run** inference and visualize detections

## Repository Structure
- **File:** [data.yaml](data.yaml) — dataset config used by training scripts
- **File:** [ddsm_yolo.py](ddsm_yolo.py) — helper / experiment script
- **File:** [train2.py](train2.py) — training entrypoint used in experiments
- **Directory:** [data/](data) — raw and converted dataset, plus csv metadata
  - `data/jpeg/` — converted images (nested case folders)
  - `data/csv/` — CSV metadata and case descriptions
- **Directory:** [runs/](runs) — inference outputs and detection visualizations
- **Directory:** [yolo_dataset/](yolo_dataset) — images/labels prepared for YOLO (train/val)

## Quickstart
1. Prepare environment (recommended virtualenv / conda).

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt  # create if missing; see notes
```

2. Inspect dataset and config:
- Check dataset config: [data.yaml](data.yaml)
- Preview CSVs: [data/csv/meta.csv](data/csv/meta.csv)

3. To train (example):

```powershell
python train2.py --data data.yaml --weights yolov8n.pt --epochs 50
```

4. To run detection on images:

```powershell
python ddsm_yolo.py detect --weights runs/train/exp/weights/best.pt --source data/jpeg
```

## Dataset Layout & Conventions
- Images are stored under `data/jpeg/<case_id>/...`.
- CSV files in `data/csv/` map case IDs to labels and metadata.
- The `yolo_dataset/` directory contains the final YOLO-style `images/` and `labels/` for `train/` and `val/` splits.

## Workflow Diagram


```
flowchart TD
  A[Raw DICOMs & CSVs] --> B[Conversion]
  B --> C[JPEG Images & Case Folders]
  C --> D[Label Extraction & CSV -> YOLO labels]
  D --> E[yolo_dataset (images/ + labels/)]
  E --> F[Training (train2.py, YOLOv8)]
  F --> G[Trained model weights]
  G --> H[Inference (ddsm_yolo.py)]
  H --> I[Visualization & runs/detect outputs]
  style A fill:#f9f,stroke:#333,stroke-width:1px
  style I fill:#bfe,stroke:#333,stroke-width:1px
```



```powershell
npm install -g @mermaid-js/mermaid-cli
mmdc -i workflow.mmd -o docs/workflow.png
```

Create a simple `workflow.mmd` file with just the Mermaid code block contents above.

## Images & Visual Examples
Add example images to `docs/` (recommended):
- `docs/workflow.png` — exported workflow diagram
- `docs/example-input.jpg` — sample input mammogram
- `docs/example-output.jpg` — sample detection visualization (from `runs/detect/`)




If you want me to pull a representative image from the repo (for example, the first detection under `runs/detect/mammography_yolo/`) and add it to `docs/` and into this README, tell me and I will add it.

## Reproducible Workflow (detailed steps)
1. Collect raw DICOMs and CSVs under `data/`.
2. Run `data/conversion.py` (or your conversion script) to produce JPEGs in `data/jpeg/`.
3. Run `data/check.py` to validate CSVs and image/label consistency.
4. Create YOLO-formatted labels in `yolo_dataset/labels/` and split images into `images/train` and `images/val`.
5. Tweak `data.yaml` to point to the `yolo_dataset` paths and class names.
6. Launch training using `train2.py`.
7. Evaluate and run inference using `ddsm_yolo.py` (or your detect script) and review outputs in `runs/detect/`.

## Tips & Notes
- If you don't have `requirements.txt`, create one listing key libs: `ultralytics`, `opencv-python`, `pandas`, `pydicom`, and `matplotlib`.
- Use smaller `--weights` (`yolov8n.pt`) for quick experiments, larger for higher accuracy.
- Keep a `docs/` folder for diagrams and example outputs; it keeps the README lean and visual.
----