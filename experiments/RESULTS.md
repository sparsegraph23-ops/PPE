# Experiment leaderboard

Metrics are the **best epoch** of each run on its own validation split, read from
`results.csv`. Weights and batch previews are gitignored; the CSVs and
`args.yaml` are kept, so this table is reproducible from the repo alone.

Run directories were renamed from timestamps to `{backbone}_ep{epochs}` during the
Phase 1 restructure. `_stoppedN` marks a run that ended early (early stopping or
interruption) — N is the epoch it actually reached.

## Person detector — `experiments/person/`

Dataset: `ppe-detection-farqh` v4 · imgsz 640 · classes `half_person, person`

| Run | Backbone | Epochs | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **yolo26l_ep150** | yolo26l | 150 | 0.900 | 0.783 | 0.8642 | **0.5905** |
| yolo26m_ep220 | yolo26m | 220 | 0.874 | 0.791 | 0.8501 | 0.5827 |
| yolo26m_ep150 | yolo26m | 150 | 0.876 | 0.778 | 0.8546 | 0.5801 |
| yolo26m_ep170 | yolo26m | 170 | 0.910 | 0.789 | 0.8529 | 0.5782 |
| yolo11l_ep170 | yolo11l | 170 | 0.860 | 0.755 | 0.8423 | 0.5725 |
| yolo26l_ep200 | yolo26l | 200 | 0.880 | 0.763 | 0.8523 | 0.5717 |
| yolo26l_ep170 | yolo26l | 170 | 0.862 | 0.802 | 0.8491 | 0.5695 |
| **yolo11l_ep200** | yolo11l | 200 | 0.810 | 0.766 | 0.8257 | 0.5684 |
| yolo11m_ep200 | yolo11m | 200 | 0.852 | 0.746 | 0.8286 | 0.5678 |
| yolo11m_ep150 | yolo11m | 150 | 0.890 | 0.749 | 0.8496 | 0.5675 |
| yolo11l_ep150 | yolo11l | 150 | 0.857 | 0.728 | 0.8291 | 0.5644 |
| yolo26m_ep200 | yolo26m | 200 | 0.851 | 0.790 | 0.8470 | 0.5615 |
| yolo11m_ep170 | yolo11m | 170 | 0.854 | 0.748 | 0.8413 | 0.5614 |
| yolo11l_ep220_stopped163 | yolo11l | 220→163 | 0.791 | 0.773 | 0.8287 | 0.5544 |

**Finding:** YOLO26 beats YOLO11 on the person stage — best-vs-best is
0.5905 → 0.5725 mAP@50-95, and 4 of the top 5 runs are YOLO26.

**Inconsistency to resolve:** `yolo26l_ep150` is loaded by the video notebook,
but `yolo11l_ep200` (8th place) is loaded by the image and evaluation notebooks.
The published end-to-end numbers therefore understate what the cascade can do.

## PPE detector — `experiments/ppe/`

Dataset: `ppe_objects` v5 · imgsz [455, 171] · classes `boot, helmet, jacket`

| Run | Backbone | Epochs | P | R | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|---|
| **yolo11l_ep200** | yolo11l | 200 | 0.899 | 0.832 | 0.8787 | **0.5400** |
| yolo11m_ep200 | yolo11m | 200 | 0.924 | 0.744 | 0.8516 | 0.5354 |
| yolo11l_ep220_stopped174 | yolo11l | 220→174 | 0.885 | 0.810 | 0.8623 | 0.5280 |
| yolo11l_ep150 | yolo11l | 150 | 0.888 | 0.848 | 0.8873 | 0.5254 |
| yolo26l_ep170_stopped151 | yolo26l | 170→151 | 0.866 | 0.846 | 0.8780 | 0.5250 |
| yolo11m_ep150 | yolo11m | 150 | 0.901 | 0.787 | 0.8643 | 0.5198 |
| yolo11m_ep200_stopped101 | yolo11m | 200→101 | 0.884 | 0.809 | 0.8734 | 0.5194 |
| yolo26m_ep200_stopped153 | yolo26m | 200→153 | 0.855 | 0.830 | 0.8822 | 0.5189 |
| yolo26m_ep170_stopped138 | yolo26m | 170→138 | 0.834 | 0.831 | 0.8744 | 0.5163 |
| yolo26m_ep150 | yolo26m | 150 | 0.859 | 0.858 | 0.8868 | 0.5158 |
| yolo26m_ep220_stopped159 | yolo26m | 220→159 | 0.891 | 0.840 | 0.8793 | 0.5129 |
| yolo26l_ep220_stopped154 | yolo26l | 220→154 | 0.880 | 0.795 | 0.8563 | 0.5080 |
| yolo26l_ep150 | yolo26l | 150 | 0.893 | 0.863 | 0.8889 | 0.5067 |
| yolo26l_ep200_stopped151 | yolo26l | 200→151 | 0.826 | 0.836 | 0.8472 | 0.5041 |

**Finding:** the opposite result — YOLO11 wins the PPE stage
(0.5400 vs 0.5250). Note that YOLO26 runs score competitively on mAP@50 but fall
behind at stricter IoU, i.e. they find the objects but localise them less tightly
on these small, thin crops.

## End-to-end cascade

Measured on `data/benchmarks/cascade_testset_v1` (49 images) by
`notebooks/30_eval_cascade.ipynb`, using `person/yolo11l_ep200` +
`ppe/yolo11l_ep200`.

```
mAP@0.5      : 0.6782
mAP@0.75     : 0.4305
mAP@0.5:0.95 : 0.4109

boot    P=0.467  R=0.824  F1=0.596
helmet  P=0.785  R=0.886  F1=0.832
jacket  P=0.662  R=0.895  F1=0.761
```

The gap between the PPE component (0.879 mAP@50) and the cascade (0.678) is the
cost of stage-1 misses propagating. Boot precision is the weakest link.
