from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms.functional as TF

from surgical_bleeding.models import load_owned_unet_class
from surgical_bleeding.temporal import ROITracker, bbox_from_mask, farneback_flow, flow_descriptors, frame_descriptor, keep_largest_component, remove_small_components, stabilize_ecc


def load_model(path: Path, import_path: str, device: str, args):
    model_class = load_owned_unet_class(import_path)
    model = model_class(base_ch=args.base_ch, dropout=args.dropout, use_batchnorm=True, upsample_mode=args.upsample_mode).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in state: state = state["model_state_dict"]
    if "state_dict" in state: state = state["state_dict"]
    cleaned = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch. Missing={missing}; unexpected={unexpected}")
    return model.eval()


@torch.no_grad()
def predict(model, frame: np.ndarray, size: int, threshold: float, device: str) -> np.ndarray:
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = TF.to_tensor(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)).unsqueeze(0).to(device)
    logits = model(tensor); logits = logits[0] if isinstance(logits, (list, tuple)) else logits
    small = (torch.sigmoid(logits)[0, 0].cpu().numpy() >= threshold).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(small).resize((width, height), Image.Resampling.NEAREST), dtype=np.uint8)


def make_plot(frame: pd.DataFrame, key: str, ylabel: str, output: Path) -> None:
    plt.figure(figsize=(9, 4.5)); plt.plot(frame.time_sec, frame[key]); plt.xlabel("Time [s]"); plt.ylabel(ylabel); plt.grid(alpha=.25); plt.tight_layout(); plt.savefig(output, dpi=180); plt.close()


def process_case(model, video: Path, output: Path, case_id: str, domain: str, label: str, args, device: str) -> dict:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError(f"Cannot open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps): fps = args.default_fps
    tracker = ROITracker(args.roi_margin, args.roi_iou_keep, args.roi_smoothing, args.empty_patience)
    previous_gray = previous_mask = None; previous_area = None; rows = []; frame_idx = 0
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    while True:
        ok, frame = capture.read()
        if not ok: break
        frame_idx += 1
        if (frame_idx - 1) % args.frame_stride: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stabilized, ecc_ok = (frame, True) if previous_gray is None or not args.use_ecc else stabilize_ecc(previous_gray, gray, frame)
        stable_gray = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        mask = remove_small_components(predict(model, stabilized, args.img_size, args.threshold, device), args.min_component_area)
        if args.use_largest_component: mask = keep_largest_component(mask)
        roi = tracker.update(bbox_from_mask(mask), frame.shape[1], frame.shape[0])
        descriptor = frame_descriptor(mask, roi, previous_mask, previous_area); gated = descriptor.pop("gated_mask")
        if previous_gray is None:
            flow_mean = flow_p95 = flow_consistency = 0.0
        else:
            flow_mask = gated if args.flow_inside_blood_only else np.full_like(gated, 255, dtype=np.uint8)
            if not args.flow_inside_blood_only and roi is not None:
                flow_mask[:] = 0; flow_mask[roi[1]:roi[3], roi[0]:roi[2]] = 255
            flow_mean, flow_p95, flow_consistency = flow_descriptors(farneback_flow(previous_gray, stable_gray), flow_mask)
        rows.append({"case_id": case_id, "domain": domain, "case_label": label, "frame_idx": frame_idx, "time_sec": frame_idx / fps, **descriptor, "flow_mean_magnitude": flow_mean, "flow_p95_magnitude": flow_p95, "flow_directional_consistency": flow_consistency, "ecc_ok": int(ecc_ok)})
        previous_gray, previous_mask, previous_area = stable_gray, gated.copy(), int(descriptor["blood_area_px"])
    capture.release()
    data = pd.DataFrame(rows)
    if data.empty: raise RuntimeError(f"No frames processed for {video}")
    data["dA_dt_px_per_sec"] = (data.blood_area_px.diff() / data.time_sec.diff()).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data.to_csv(output / "temporal_metrics_per_frame.csv", index=False)
    summary = {"case_id": case_id, "domain": domain, "case_label": label, "num_frames": len(data), "duration_sec": float(data.time_sec.max()), "peak_blood_area_px": int(data.blood_area_px.max()), "peak_occupancy_ratio": float(data.occupancy_ratio.max()), "peak_dA_dt_px_per_sec": float(data.dA_dt_px_per_sec.max()), "mean_temporal_iou": float(data.temporal_iou.mean()), "mean_abs_frame_to_frame_area_change_ratio": float(data.frame_to_frame_area_change_ratio.abs().mean()), "mean_flow_magnitude": float(data.flow_mean_magnitude.mean()), "p95_flow_magnitude": float(np.percentile(data.flow_p95_magnitude, 95)), "mean_flow_directional_consistency": float(data.flow_directional_consistency.mean()), "ecc_success_rate": float(data.ecc_ok.mean())}
    (output / "temporal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for key, label_text in [("blood_area_px", "Blood area [px]"), ("occupancy_ratio", "Occupancy ratio"), ("dA_dt_px_per_sec", "dA/dt [px/s]"), ("temporal_iou", "Temporal IoU"), ("flow_mean_magnitude", "Mean flow magnitude")]: make_plot(data, key, label_text, output / f"plot_{key}.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract temporal bleeding descriptors from video.")
    parser.add_argument("--model-path", type=Path, required=True); parser.add_argument("--model-import", default="src.models.unet:UNetBinary")
    source = parser.add_mutually_exclusive_group(required=True); source.add_argument("--video-path", type=Path); source.add_argument("--manifest", type=Path)
    parser.add_argument("--case-id"); parser.add_argument("--domain", default="unknown"); parser.add_argument("--case-label", default="temporal sequence")
    parser.add_argument("--out-root", type=Path, default=Path("results/temporal_descriptors")); parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=.5); parser.add_argument("--base-ch", type=int, default=64); parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--upsample-mode", choices=["transpose", "bilinear"], default="transpose"); parser.add_argument("--default-fps", type=float, default=25)
    parser.add_argument("--frame-stride", type=int, default=1); parser.add_argument("--min-component-area", type=int, default=30); parser.add_argument("--roi-margin", type=int, default=60)
    parser.add_argument("--roi-iou-keep", type=float, default=.3); parser.add_argument("--roi-smoothing", type=float, default=.7); parser.add_argument("--empty-patience", type=int, default=5)
    parser.add_argument("--no-largest-component", dest="use_largest_component", action="store_false"); parser.set_defaults(use_largest_component=True)
    parser.add_argument("--no-ecc", dest="use_ecc", action="store_false"); parser.set_defaults(use_ecc=True)
    parser.add_argument("--flow-inside-roi", dest="flow_inside_blood_only", action="store_false"); parser.set_defaults(flow_inside_blood_only=True)
    args = parser.parse_args(); args.out_root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"; model = load_model(args.model_path, args.model_import, device, args)
    cases = pd.read_csv(args.manifest).to_dict("records") if args.manifest else [{"video_path": str(args.video_path), "case_id": args.case_id or args.video_path.stem, "domain": args.domain, "case_label": args.case_label}]
    summaries = [process_case(model, Path(row["video_path"]), args.out_root / str(row["case_id"]), str(row["case_id"]), str(row["domain"]), str(row["case_label"]), args, device) for row in cases]
    pd.DataFrame(summaries).to_csv(args.out_root / "temporal_summary_all.csv", index=False)


if __name__ == "__main__":
    main()

