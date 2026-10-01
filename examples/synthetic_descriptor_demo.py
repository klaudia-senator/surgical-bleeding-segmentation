"""Data-free smoke demo for mask post-processing and temporal descriptors."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from surgical_bleeding.temporal import ROITracker, bbox_from_mask, frame_descriptor, remove_small_components


def main() -> None:
    output = Path("results/synthetic_demo")
    output.mkdir(parents=True, exist_ok=True)
    tracker = ROITracker(margin=8, empty_patience=2)
    previous_mask = None
    previous_area = None
    rows = []
    for frame in range(30):
        mask = np.zeros((128, 192), dtype=np.uint8)
        radius = 5 + frame // 3
        cv2.circle(mask, (55 + frame, 64), radius, 255, -1)
        mask = remove_small_components(mask, min_area=20)
        roi = tracker.update(bbox_from_mask(mask), mask.shape[1], mask.shape[0])
        values = frame_descriptor(mask, roi, previous_mask, previous_area)
        gated = values.pop("gated_mask")
        rows.append({"frame_idx": frame, "time_sec": frame / 10.0, **values})
        previous_mask, previous_area = gated, int(values["blood_area_px"])
    table = pd.DataFrame(rows)
    table["dA_dt_px_per_sec"] = (table.blood_area_px.diff() / table.time_sec.diff()).fillna(0.0)
    table.to_csv(output / "synthetic_temporal_metrics.csv", index=False)
    print(table.tail().to_string(index=False))
    print(f"Saved: {output / 'synthetic_temporal_metrics.csv'}")


if __name__ == "__main__":
    main()

