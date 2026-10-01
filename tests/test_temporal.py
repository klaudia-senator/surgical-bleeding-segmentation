import cv2
import numpy as np

from surgical_bleeding.temporal import ROITracker, bbox_from_mask, flow_descriptors, frame_descriptor, keep_largest_component


def test_largest_component_filter():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    mask[8:15, 8:15] = 255
    result = keep_largest_component(mask)
    assert int((result > 0).sum()) == 49


def test_roi_tracker_expires_after_empty_patience():
    tracker = ROITracker(margin=0, empty_patience=1)
    assert tracker.update((1, 1, 4, 4), 10, 10) is not None
    assert tracker.update(None, 10, 10) is not None
    assert tracker.update(None, 10, 10) is None


def test_descriptor_area_and_temporal_iou():
    mask = np.zeros((10, 10), dtype=np.uint8); mask[2:5, 2:5] = 255
    first = frame_descriptor(mask, (0, 0, 10, 10), None, None)
    second = frame_descriptor(mask, (0, 0, 10, 10), first["gated_mask"], first["blood_area_px"])
    assert first["blood_area_px"] == 9
    assert second["temporal_iou"] == 1.0


def test_flow_directional_consistency():
    flow = np.zeros((4, 4, 2), dtype=np.float32); flow[..., 0] = 2.0
    mask = np.full((4, 4), 255, dtype=np.uint8)
    mean, p95, consistency = flow_descriptors(flow, mask)
    assert mean == p95 == 2.0
    assert np.isclose(consistency, 1.0)

