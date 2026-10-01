from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            output[labels == label] = 255
    return output


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> Box | None:
    ys, xs = np.where(mask > 0)
    return None if not len(xs) else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_iou(a: Box, b: Box, eps: float = 1e-6) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return float((intersection + eps) / (area_a + area_b - intersection + eps))


def binary_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    aa, bb = a > 0, b > 0
    return float((np.logical_and(aa, bb).sum() + eps) / (np.logical_or(aa, bb).sum() + eps))


@dataclass
class ROITracker:
    margin: int = 60
    keep_iou: float = 0.3
    smoothing: float = 0.7
    empty_patience: int = 5
    box: Box | None = None
    empty_count: int = 0

    def update(self, detected: Box | None, width: int, height: int) -> Box | None:
        if detected is None:
            self.empty_count += 1
            if self.empty_count > self.empty_patience:
                self.box = None
            return self.box
        expanded = (max(0, detected[0] - self.margin), max(0, detected[1] - self.margin), min(width, detected[2] + self.margin), min(height, detected[3] + self.margin))
        if self.box is not None and bbox_iou(self.box, expanded) >= self.keep_iou:
            self.box = tuple(int(round(self.smoothing * old + (1 - self.smoothing) * new)) for old, new in zip(self.box, expanded))  # type: ignore[assignment]
        else:
            self.box = expanded
        self.empty_count = 0
        return self.box


def stabilize_ecc(previous_gray: np.ndarray, current_gray: np.ndarray, current_bgr: np.ndarray, motion_model: int = cv2.MOTION_EUCLIDEAN, iterations: int = 50, eps: float = 1e-5) -> tuple[np.ndarray, bool]:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        _, warp = cv2.findTransformECC(previous_gray, current_gray, warp, motion_model, criteria, None, 1)
        aligned = cv2.warpAffine(current_bgr, warp, (previous_gray.shape[1], previous_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return aligned, True
    except cv2.error:
        return current_bgr, False


def farneback_flow(previous_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def flow_descriptors(flow: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    valid = mask > 0
    if not valid.any():
        return 0.0, 0.0, 0.0
    fx, fy = flow[..., 0][valid], flow[..., 1][valid]
    magnitude = np.hypot(fx, fy)
    unit_x, unit_y = fx / np.maximum(magnitude, 1e-6), fy / np.maximum(magnitude, 1e-6)
    consistency = np.hypot(unit_x.mean(), unit_y.mean())
    return float(magnitude.mean()), float(np.percentile(magnitude, 95)), float(consistency)


def frame_descriptor(mask: np.ndarray, roi: Box | None, previous_mask: np.ndarray | None, previous_area: int | None) -> dict[str, float | int]:
    height, width = mask.shape
    if roi is None:
        roi = (0, 0, width, height)
    gated = np.zeros_like(mask)
    gated[roi[1]:roi[3], roi[0]:roi[2]] = mask[roi[1]:roi[3], roi[0]:roi[2]]
    area = int((gated > 0).sum())
    roi_area = max(1, (roi[2] - roi[0]) * (roi[3] - roi[1]))
    change = 0 if previous_area is None else area - previous_area
    return {"blood_area_px": area, "roi_area_px": roi_area, "occupancy_ratio": area / roi_area, "frame_to_frame_area_change_px": change, "frame_to_frame_area_change_ratio": change / previous_area if previous_area else 0.0, "temporal_iou": 0.0 if previous_mask is None and area else (1.0 if previous_mask is None else binary_iou(previous_mask, gated)), "gated_mask": gated}

