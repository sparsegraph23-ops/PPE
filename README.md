# PPE Compliance Detection

A two-stage YOLO cascade that checks whether people in construction imagery are wearing
their required personal protective equipment, plus a face-recognition module that maps a
detected person back to an employee ID.

**Stage 1 — person detection** (`half_person`, `person`) finds and crops each worker.
**Stage 2 — PPE detection** (`boot`, `helmet`, `jacket`) runs on each crop.
A `person` needs helmet + jacket + boot to be compliant; a `half_person` (upper body only)
needs helmet + jacket.

## Setup

Everything is already installed in the `ppe` conda environment:

```bash
conda activate ppe
```

To recreate it elsewhere: `conda create -n ppe python=3.12 && pip install -r requirements.txt`.

All notebook paths are **relative to the notebook's own directory**, so the project can be
moved anywhere as long as notebooks are run with their own folder as the working directory
(the default in VS Code and Jupyter).

## Running

Open a notebook in VS Code and select the **`ppe`** kernel, then Run All. There is no
Jupyter server installed in the env; for a browser UI install one first
(`pip install jupyterlab`, then `jupyter lab`).

### 1. Image pipeline — `model_pipeline/pipeline.ipynb`

The main end-to-end demo. Reads the 16 images in `quality images for inferencing/`,
resizes to 640×640, detects people, de-duplicates boxes with IoU-based NMS, crops each
person, resizes crops to 171×455, and runs PPE detection.

Results land in:
- `runs/detect/predict*/` — stage-1 annotated images
- `extracted_humans/` — raw person crops
- `resized_human_images/` — crops resized for stage 2
- `ppe_results/` — **final annotated PPE detections**

### 2. Video pipeline — `model_pipeline/video_inf_pipeline2.ipynb`

The better of the two video notebooks: tracks people across frames with persistent IDs and
labels each with `COMPLIANT` / `PARTIAL` / `NON-COMPLIANT` (green / orange / red) plus the
list of missing items. Set `video_path` in cell 5 to pick a clip from `inf_vid/`, or use `0`
for a live webcam.

- A live OpenCV window opens during the run — press **`q`** to stop early.
- Annotated output is written to `out_vid/out_<input name>.mp4`.
- Always run the final cleanup cell; it releases the writer and closes the window.

`video_inf_pipeline.ipynb` is the earlier version — it draws boxes but does not evaluate
compliance, and writes to `inf_vid/output_ppe_detected.mp4`.

### 3. Validation — `model_pipeline/new_pipeline.ipynb`

Scores the cascade against the 49-image Roboflow test set in `test dataset/`. Prints
stage-1 mAP, end-to-end PPE mAP@0.5 / 0.75 / 0.5:0.95, a confusion matrix and per-class
P/R/F1 at conf 0.50. Writes `comparison_results/` (side-by-side prediction vs. ground truth
images and `summary.txt`).

Skip cell 0 (`%pip install`) — the env already has pinned versions and reinstalling may
change them.

### 4. Face recognition — `model_pipeline/face_detect.ipynb`

Extracts 128-d `face_recognition` embeddings and stores them in a ChromaDB collection at
`face_vector_db/` (cosine distance). Four employees are already enrolled
(`emp001`–`emp004`). `enroll_employee(image, id)` adds or updates one; `identify_person(image)`
queries and reports a match below a 0.4 distance threshold.

## Model weights

Weights live under `runs/<timestamp> <epochs> <size>/weights/best.pt`. The pipelines use:

| Purpose | Weights |
| --- | --- |
| Person (images, validation) | `model_finetune/person/runs/2026-06-01 15:31:23 200 l/` |
| Person (video) | `model_finetune/person/runs/2026-06-04 16:54:10 150 l/` |
| PPE (all pipelines) | `model_finetune/ppe_detect/runs/2026-06-02 18:26:19 200 l/` |

The `.pt` files are gitignored (too large for GitHub), so a fresh clone has the metrics CSVs
but not the weights — copy them across manually or retrain.

## Training notebooks

`model_finetune/person/person_detection_model.ipynb`, `model_finetune/ppe_detect/ppe_detection_model.ipynb`
and `yolo 26/ppe_detection_model.ipynb` fine-tune the detectors. They need a Roboflow API key
and download their datasets (`PPE-Detection-4/`, `ppe_objects-5/`) into the notebook's own
directory — those datasets are gitignored and are not present in this checkout, so these
notebooks will not run as-is until the download cell is executed. Base weights
(`yolo11m.pt`, `yolo26l.pt`, …) are fetched automatically by Ultralytics.
