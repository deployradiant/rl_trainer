"""Rollout JSON and preview image artifacts for trajectory training."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from training.mcts import run_trajectory_mcts
from training.metrics import ade, fde


def save_prediction_artifacts(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    preview_path: str,
    rollouts_path: str,
    split: str,
    input_frames: int,
    horizon: int,
    mcts_simulations: int,
    mcts_top_k: int,
    mcts_c_puct: float,
) -> None:
    """Save visual artifacts for held-out eval rollouts."""
    if len(dataset) == 0:
        return

    samples = []
    seen_sequences = set()
    for idx in range(len(dataset)):
        sample = dataset.full_sample(idx) if hasattr(dataset, "full_sample") else dataset[idx]
        sequence = sample["sequence"]
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        samples.append(sample)
    if not samples:
        return

    rollouts = []
    preview_image = None
    for sample in samples:
        rollout, image = build_rollout_artifact(
            model=model,
            sample=sample,
            device=device,
            split=split,
            input_frames=input_frames,
            horizon=horizon,
            mcts_simulations=mcts_simulations,
            mcts_top_k=mcts_top_k,
            mcts_c_puct=mcts_c_puct,
        )
        rollouts.append(rollout)
        if preview_image is None:
            preview_image = image

    preview_out = Path(preview_path)
    preview_out.parent.mkdir(parents=True, exist_ok=True)
    preview_image.save(preview_out)
    print(f"Preview saved to {preview_path}")

    rollouts_out = Path(rollouts_path)
    rollouts_out.parent.mkdir(parents=True, exist_ok=True)
    rollouts_out.write_text(json.dumps({"rollouts": rollouts}, indent=2))
    print(f"Rollout list saved to {rollouts_out}")


def build_rollout_artifact(
    model: nn.Module,
    sample: dict,
    device: str,
    split: str,
    input_frames: int,
    horizon: int,
    mcts_simulations: int,
    mcts_top_k: int,
    mcts_c_puct: float,
) -> tuple[dict, Image.Image]:
    """Build a rollout JSON payload and preview image for one sample."""
    bbox = sample["player_bbox"].unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if "visual_features" in sample:
                visual_features = sample["visual_features"].unsqueeze(0).to(device)
            else:
                frames = sample["frames"].unsqueeze(0).to(device)
                visual_features = model.extract_visual_features(frames)
            outputs = model.predict_from_visual_features(
                visual_features,
                bbox,
                return_outputs=True,
                greedy=True,
            )
            pred = outputs["future_xy"][0].detach().cpu().numpy()
            action_logits = outputs["action_logits"][0].detach().cpu()
            mcts_result = None
            if mcts_simulations > 0:
                mcts_result = run_trajectory_mcts(
                    model=model,
                    visual_features=visual_features,
                    player_bbox=bbox,
                    simulations=mcts_simulations,
                    top_k=mcts_top_k,
                    c_puct=mcts_c_puct,
                )
            teacher_action_logits = (
                model.teacher_forced_action_logits(
                    visual_features,
                    sample["player_bbox"].unsqueeze(0).to(device),
                    sample["future_xy"].unsqueeze(0).to(device),
                )[0]
                .detach()
                .cpu()
            )
            teacher_action_values = (
                model.teacher_forced_action_values(
                    visual_features,
                    sample["player_bbox"].unsqueeze(0).to(device),
                    sample["future_xy"].unsqueeze(0).to(device),
                )[0]
                .detach()
                .cpu()
            )
    finally:
        if was_training:
            model.train()

    target = sample["future_xy"].numpy()
    input_bboxes = sample["player_bbox"].numpy()
    sample_ade = ade(pred, target)
    sample_fde = fde(pred, target)
    action_labels = (
        model.action_labels(
            sample["player_bbox"].unsqueeze(0).to(device),
            sample["future_xy"].unsqueeze(0).to(device),
        )[0]
        .detach()
        .cpu()
    )
    action_pred = action_logits.argmax(dim=-1)
    teacher_action_pred = teacher_action_logits.argmax(dim=-1)
    value_action_pred = teacher_action_values.argmax(dim=-1)
    sample_action_accuracy = float((action_pred == action_labels).float().mean().item())
    sample_teacher_action_accuracy = float(
        (teacher_action_pred == action_labels).float().mean().item()
    )
    sample_value_action_accuracy = float((value_action_pred == action_labels).float().mean().item())

    frame = sample["frames"][-1].permute(1, 2, 0).numpy()
    image = Image.fromarray((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    current = sample["player_bbox"][-1].numpy()
    cx, cy, bw, bh = current
    x0 = (cx - bw / 2) * width
    y0 = (cy - bh / 2) * height
    x1 = (cx + bw / 2) * width
    y1 = (cy + bh / 2) * height
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)

    current_xy = np.array([[cx, cy]], dtype=np.float32)
    actual_points = np.concatenate([current_xy, target], axis=0)
    pred_points = np.concatenate([current_xy, pred], axis=0)
    mcts = _mcts_artifact(mcts_result, current_xy, target, action_labels)

    def to_pixels(points: np.ndarray) -> list[tuple[float, float]]:
        return [(float(x * width), float(y * height)) for x, y in points]

    actual_pixels = to_pixels(actual_points)
    pred_pixels = to_pixels(pred_points)
    draw.line(actual_pixels, fill=(80, 220, 120), width=3)
    draw.line(pred_pixels, fill=(255, 80, 80), width=3)
    if mcts is not None:
        mcts_pixels = to_pixels(mcts["points"])
        draw.line(mcts_pixels, fill=(80, 180, 255), width=3)
    for point in actual_pixels[1::4]:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(80, 220, 120))
    for point in pred_pixels[1::4]:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(255, 80, 80))
    if mcts is not None:
        for point in mcts_pixels[1::4]:
            draw.ellipse(
                [point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(80, 180, 255)
            )
    draw.text((8, 8), "green=actual red=pred blue=mcts", fill=(255, 255, 255))

    start_frame = int(sample["start_frame"])
    input_frame_numbers = list(range(start_frame, start_frame + input_frames))
    future_frame_numbers = list(
        range(start_frame + input_frames, start_frame + input_frames + horizon)
    )
    input_centers = input_bboxes[:, :2]
    actual_path = np.concatenate([input_centers, target], axis=0)
    pred_path = np.concatenate([current_xy, pred], axis=0)

    rollout = {
        "split": split,
        "sequence": sample["sequence"],
        "track_id": int(sample["track_id"]),
        "start_frame": start_frame,
        "input_frames": input_frames,
        "horizon": horizon,
        "action_stride": getattr(model, "action_stride", 1),
        "action_count": int(getattr(model, "action_codebook", torch.empty(0)).shape[0]),
        "fps": 25,
        "current_index": input_frames - 1,
        "frame_numbers": input_frame_numbers + future_frame_numbers,
        "input_frame_numbers": input_frame_numbers,
        "future_frame_numbers": future_frame_numbers,
        "input_bboxes": input_bboxes.tolist(),
        "actual_xy": actual_path.tolist(),
        "pred_xy": pred_path.tolist(),
        "metrics": {
            "ADE": sample_ade,
            "FDE": sample_fde,
            "actual_displacement": float(np.linalg.norm(target[-1] - input_centers[-1])),
            "pred_displacement": float(np.linalg.norm(pred[-1] - input_centers[-1])),
            "action_accuracy": sample_action_accuracy,
            "teacher_action_accuracy": sample_teacher_action_accuracy,
            "value_action_accuracy": sample_value_action_accuracy,
        },
    }
    if mcts is not None:
        rollout["mcts_xy"] = mcts["points"].tolist()
        rollout["mcts_action_indices"] = mcts["actions"]
        rollout["mcts"] = {
            "simulations": mcts_simulations,
            "top_k": mcts_top_k,
            "c_puct": mcts_c_puct,
            "learned_value": mcts["learned_value"],
            "root_visits": mcts["root_visits"],
            "root_children": mcts["root_children"],
        }
        rollout["metrics"].update(
            {
                "mcts_ADE": mcts["ADE"],
                "mcts_FDE": mcts["FDE"],
                "mcts_displacement": float(
                    np.linalg.norm(mcts["future_xy"][-1] - input_centers[-1])
                ),
                "mcts_action_accuracy": mcts["action_accuracy"],
            }
        )
    return rollout, image


def _mcts_artifact(
    mcts_result: dict | None,
    current_xy: np.ndarray,
    target: np.ndarray,
    action_labels: torch.Tensor,
) -> dict | None:
    if mcts_result is None:
        return None

    mcts_future = mcts_result["future_xy"].detach().cpu().numpy()
    mcts_points = np.concatenate([current_xy, mcts_future], axis=0)
    mcts_action_pred = torch.tensor(mcts_result["actions"], dtype=torch.long)
    mcts_action_accuracy = float((mcts_action_pred == action_labels).float().mean().item())
    return {
        "future_xy": mcts_future,
        "points": mcts_points,
        "actions": mcts_result["actions"],
        "action_accuracy": mcts_action_accuracy,
        "learned_value": float(mcts_result["learned_value"]),
        "root_visits": int(mcts_result["root_visits"]),
        "root_children": mcts_result["root_children"],
        "ADE": ade(mcts_future, target),
        "FDE": fde(mcts_future, target),
    }
