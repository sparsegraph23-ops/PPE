# Vision-Based Real-Time PPE Monitoring for Industrial Environments

A two-stage detection cascade that finds people in an image or video stream and
checks whether each one is wearing the required personal protective equipment
(helmet, hi-vis jacket, boots), plus an optional face-recognition stage for
identifying who the non-compliant worker is.

```
frame ──▶ Stage 1: person detector (YOLO, 640×640)
              │  classes: person, half_person
              ▼  crop each person, resize to 171×455
          Stage 2: PPE detector (YOLO, 455×171)
              │  classes: boot, helmet, jacket
              ▼  reproject boxes to frame coords, NMS
          compliance verdict per person
              │
              └──▶ (optional) face ID against the employee vector DB
```

Compliance rule: a `person` (full body visible) needs helmet + jacket + boot; a
`half_person` (upper body only) needs helmet + jacket.

## Results

End-to-end, on the 49-image benchmark in `data/benchmarks/cascade_testset_v1`:

| Metric | Value |
|---|---|
| mAP@0.5 | 0.678 |
| mAP@0.75 | 0.431 |
| mAP@0.5:0.95 | 0.411 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| helmet | 0.785 | 0.886 | 0.832 |
| jacket | 0.662 | 0.895 | 0.761 |
| boot | 0.467 | 0.824 | 0.596 |

Component models score higher in isolation (person 0.864 mAP@50, PPE 0.879) —
the gap is stage-1 misses propagating through the cascade. Full leaderboard for
all 34 training runs: [`experiments/RESULTS.md`](experiments/RESULTS.md).

## Quickstart

```bash
conda activate ppe
jupyter lab
```

Notebooks are numbered by pipeline stage and can be run top-to-bottom. Each one
begins with a bootstrap cell that locates the repo root, so paths work from any
checkout location.

| Notebook | Purpose | Needs |
|---|---|---|
| `10_train_person.ipynb` | Fine-tune the person detector | Roboflow key + GPU |
| `11_train_ppe.ipynb` | Fine-tune the PPE detector | Roboflow key + GPU |
| `12_ablation_yolo26.ipynb` | YOLO26-vs-YOLO11 sweep for stage 2 | Roboflow key + GPU |
| `20_infer_image.ipynb` | Run the cascade over a folder of stills | — |
| `21_infer_video.ipynb` | Tracked, compliance-annotated video | display for the preview window |
| `22_face_enroll_identify.ipynb` | Enroll / identify employees | — |
| `30_eval_cascade.ipynb` | End-to-end mAP + confusion matrix | — |

`notebooks/exploratory/` holds superseded and scratch notebooks. They are kept
for reference and are not maintained.

## Layout

| Path | Contents |
|---|---|
| `configs/` | Paths and hyperparameters — no path literals in notebooks |
| `data/` | Datasets, demo media, face DB. Gitignored; see `data/README.md` |
| `models/` | Released weights + `registry.yaml` mapping model → metrics → source run |
| `notebooks/` | Numbered by stage: 10s train, 20s infer, 30s evaluate |
| `src/ppe/` | Skeleton for the shared library the notebooks will migrate into |
| `scripts/` | CLI entry points (to come) |
| `experiments/` | All training runs + `RESULTS.md` leaderboard |
| `outputs/` | Everything generated. Gitignored, safe to delete |
| `docs/` | Architecture, dataset/model cards, evaluation protocol |

## Models in use

Defined in [`models/registry.yaml`](models/registry.yaml):

| Stage | Weights | mAP@50-95 |
|---|---|---|
| Person (image + eval) | `models/person/yolo11l_ep200/best.pt` | 0.568 |
| Person (video) | `models/person/yolo26l_ep150/best.pt` | 0.591 |
| PPE (all pipelines) | `models/ppe/yolo11l_ep200/best.pt` | 0.540 |

The two pipelines currently load **different** person detectors. The video one is
the stronger model; the reported end-to-end numbers use the weaker one.
Unifying them is a pending decision.

## Known limitations

- **Cascade error propagation.** A person missed at stage 1 is never checked for
  PPE — invisible recall loss, and the main cause of the 0.879 → 0.678 drop.
- **Compliance is inferred from absence of detection.** There are no explicit
  `no_helmet` / `no_vest` classes, so a missed detection is indistinguishable
  from a genuine violation and raises a false alarm.
- **Boot precision is weak** (0.467) — small objects at the bottom of a stretched
  crop, frequently occluded.
- **No temporal smoothing.** Track IDs are available but compliance is recomputed
  independently per frame, so verdicts flicker.
- **Not yet real-time.** One PPE forward pass per person per frame, run
  sequentially, no batching.
- **Benchmark is 49 images** from the same distribution as training. Metrics are
  indicative, not tight, and there is no cross-site validation.

## Security / privacy

- The Roboflow API key is still hardcoded in the training notebooks and present
  in git history. **It must be rotated** and moved to `.env`.
- `data/external/faces/` and `data/processed/face_db/` hold photographs and
  biometric embeddings of identifiable people. The `.gitignore` covers them for
  **new** files, but they were already tracked before the restructure and remain
  tracked — an ignore rule cannot untrack existing files. To stop tracking them
  (files stay on disk):

  ```bash
  git rm -r --cached data/external/faces data/processed/face_db
  ```

  They will still exist in history until it is rewritten. Do not redistribute.
