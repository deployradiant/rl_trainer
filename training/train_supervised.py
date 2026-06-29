"""Supervised training loop for trajectory prediction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, Subset

from data.tracking_dataset import (
    MultiSequenceSoccerNetTrackingDataset,
    choose_sequence_subsets,
    choose_sequence_subsets_by_count,
    list_sequences,
)
from models.qwen_football import QwenFootballModel


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    """Average Displacement Error over the future window."""
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    """Final Displacement Error at the last timestep."""
    return float(np.linalg.norm(pred[-1] - target[-1]))


def outputs_from_batch(
    model: nn.Module,
    batch: dict,
    device: str,
    greedy: bool = False,
) -> dict[str, torch.Tensor]:
    bbox = batch["player_bbox"].to(device)
    visual_features = visual_features_from_batch(model, batch, device)
    return model.predict_from_visual_features(
        visual_features,
        bbox,
        return_outputs=True,
        greedy=greedy,
    )


def visual_features_from_batch(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    if "visual_features" in batch:
        return batch["visual_features"].to(device)
    frames = batch["frames"].to(device)
    return model.extract_visual_features(frames)


def teacher_action_logits_from_batch(
    model: nn.Module,
    batch: dict,
    device: str,
) -> torch.Tensor:
    bbox = batch["player_bbox"].to(device)
    target = batch["future_xy"].to(device)
    visual_features = visual_features_from_batch(model, batch, device)
    return model.teacher_forced_action_logits(visual_features, bbox, target)


def predict_from_batch(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    return outputs_from_batch(model, batch, device, greedy=True)["future_xy"]


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ades, fdes = [], []
    action_correct = 0
    action_total = 0
    teacher_action_correct = 0
    teacher_action_total = 0
    with torch.no_grad():
        for batch in loader:
            bbox = batch["player_bbox"].to(device)
            target = batch["future_xy"].to(device)
            outputs = outputs_from_batch(model, batch, device, greedy=True)
            pred = outputs["future_xy"]
            labels = model.action_labels(bbox, target)
            action_pred = outputs["action_logits"].argmax(dim=-1)
            action_correct += int((action_pred == labels).sum().item())
            action_total += int(labels.numel())
            teacher_action_pred = teacher_action_logits_from_batch(model, batch, device).argmax(dim=-1)
            teacher_action_correct += int((teacher_action_pred == labels).sum().item())
            teacher_action_total += int(labels.numel())
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            for i in range(len(pred_np)):
                ades.append(ade(pred_np[i], target_np[i]))
                fdes.append(fde(pred_np[i], target_np[i]))
    return {
        "ADE": float(np.mean(ades)),
        "FDE": float(np.mean(fdes)),
        "action_accuracy": action_correct / max(1, action_total),
        "teacher_action_accuracy": teacher_action_correct / max(1, teacher_action_total),
    }


def select_sample_indices(
    dataset: Dataset,
    count: int,
    selection: str,
) -> tuple[list[int], list[dict]]:
    """Select a tiny deterministic sample subset for overfit demos."""
    if count <= 0:
        return [], []
    count = min(count, len(dataset))
    if selection == "first":
        indices = list(range(count))
    elif selection == "max_motion":
        scored = [(dataset.motion_score(idx), idx) for idx in range(len(dataset))]
        indices = [idx for _, idx in sorted(scored, reverse=True)[:count]]
    else:
        raise ValueError(f"Unknown sample_selection={selection!r}")
    metadata = [dataset.sample_metadata(idx) for idx in indices]
    return indices, metadata


class CachedFeatureDataset(Dataset):
    """Dataset wrapper that attaches frozen Qwen visual features to each sample."""

    def __init__(self, dataset: Dataset, visual_features: torch.Tensor, samples: list[dict]):
        self.dataset = dataset
        self.visual_features = visual_features
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = dict(self.samples[idx])
        sample["visual_features"] = self.visual_features[idx]
        return sample

    def full_sample(self, idx: int) -> dict:
        sample = dict(self.dataset[idx])
        sample["visual_features"] = self.visual_features[idx]
        return sample


def cache_dataset_visual_features(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    batch_size: int,
) -> CachedFeatureDataset:
    """Precompute frozen Qwen features for small deterministic overfit runs."""
    features = []
    cached_samples = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            frames = batch["frames"].to(device)
            visual_features = model.extract_visual_features(frames).cpu()
            features.append(visual_features)
            for i in range(visual_features.shape[0]):
                cached_samples.append({
                    "player_bbox": batch["player_bbox"][i].cpu(),
                    "future_xy": batch["future_xy"][i].cpu(),
                    "sequence": batch["sequence"][i],
                    "track_id": int(batch["track_id"][i]),
                    "start_frame": int(batch["start_frame"][i]),
                })
            if batch_idx == 1 or batch_idx == len(loader) or batch_idx % 10 == 0:
                print(f"Cached visual feature batch {batch_idx}/{len(loader)}")
    if was_training:
        model.train()
    return CachedFeatureDataset(dataset, torch.cat(features, dim=0), cached_samples)


def save_prediction_artifacts(
    model: nn.Module,
    dataset: Dataset,
    device: str,
    preview_path: str,
    rollouts_path: str,
    split: str,
    input_frames: int,
    horizon: int,
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
) -> tuple[dict, Image.Image]:
    """Build a rollout JSON payload and preview image for one sample."""
    bbox = sample["player_bbox"].unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if "visual_features" in sample:
            visual_features = sample["visual_features"].unsqueeze(0).to(device)
            outputs = model.predict_from_visual_features(
                visual_features,
                bbox,
                return_outputs=True,
                greedy=True,
            )
        else:
            frames = sample["frames"].unsqueeze(0).to(device)
            outputs = model.forward_outputs(frames, bbox, greedy=True)
        pred = outputs["future_xy"][0].detach().cpu().numpy()
        action_logits = outputs["action_logits"][0].detach().cpu()
    if was_training:
        model.train()
    target = sample["future_xy"].numpy()
    input_bboxes = sample["player_bbox"].numpy()
    sample_ade = ade(pred, target)
    sample_fde = fde(pred, target)
    action_labels = model.action_labels(
        sample["player_bbox"].unsqueeze(0).to(device),
        sample["future_xy"].unsqueeze(0).to(device),
    )[0].detach().cpu()
    action_pred = action_logits.argmax(dim=-1)
    if "visual_features" in sample:
        teacher_visual_features = sample["visual_features"].unsqueeze(0).to(device)
    else:
        teacher_frames = sample["frames"].unsqueeze(0).to(device)
        teacher_visual_features = model.extract_visual_features(teacher_frames)
    teacher_action_logits = model.teacher_forced_action_logits(
        teacher_visual_features,
        sample["player_bbox"].unsqueeze(0).to(device),
        sample["future_xy"].unsqueeze(0).to(device),
    )[0].detach().cpu()
    teacher_action_pred = teacher_action_logits.argmax(dim=-1)
    sample_action_accuracy = float((action_pred == action_labels).float().mean().item())
    sample_teacher_action_accuracy = float((teacher_action_pred == action_labels).float().mean().item())

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

    def to_pixels(points: np.ndarray) -> list[tuple[float, float]]:
        return [(float(x * width), float(y * height)) for x, y in points]

    actual_pixels = to_pixels(actual_points)
    pred_pixels = to_pixels(pred_points)
    draw.line(actual_pixels, fill=(80, 220, 120), width=3)
    draw.line(pred_pixels, fill=(255, 80, 80), width=3)
    for point in actual_pixels[1::4]:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(80, 220, 120))
    for point in pred_pixels[1::4]:
        draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], fill=(255, 80, 80))
    draw.text((8, 8), "green=actual red=pred", fill=(255, 255, 255))

    start_frame = int(sample["start_frame"])
    input_frame_numbers = list(range(start_frame, start_frame + input_frames))
    future_frame_numbers = list(range(start_frame + input_frames, start_frame + input_frames + horizon))
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
        },
    }
    return rollout, image


def build_dataset(
    data_root: str,
    split: str,
    sequences: list[str],
    input_frames: int,
    horizon: int,
    stride: int,
    image_size: int,
    max_frames: int | None,
) -> MultiSequenceSoccerNetTrackingDataset:
    dataset = MultiSequenceSoccerNetTrackingDataset(
        data_root=data_root,
        sequences=sequences,
        split=split,
        input_frames=input_frames,
        horizon=horizon,
        stride=stride,
        image_size=image_size,
        max_frames=max_frames,
    )
    if len(dataset) == 0:
        raise ValueError(
            "No samples found "
            f"(data_root={data_root}, split={split}, sequences={sequences}, "
            f"input_frames={input_frames}, horizon={horizon}, max_frames={max_frames})"
        )
    return dataset


def train(
    data_root: str,
    sequence: str | None = None,
    split: str = "train",
    input_frames: int = 5,
    horizon: int = 25,
    action_stride: int = 5,
    stride: int = 50,
    image_size: int = 224,
    max_frames: int = 250,
    batch_size: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    train_sequence_count: int = 0,
    eval_sequence_count: int = 0,
    seed: int = 7,
    overfit_samples: int = 0,
    sample_selection: str = "first",
    cache_visual_features: bool = False,
    feature_cache_batch_size: int = 4,
    policy_loss_weight: float = 0.05,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    log_every: int = 10,
    preview_path: str = "trajectory_preview.png",
    rollouts_path: str = "trajectory_rollouts.json",
):
    """Run supervised trajectory training on fixed train/eval sequence subsets."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if sequence:
        train_sequences = [sequence]
        eval_sequences = [sequence]
        all_sequences = [sequence]
    else:
        all_sequences = list_sequences(data_root, split)
        if train_sequence_count or eval_sequence_count:
            train_sequences, eval_sequences = choose_sequence_subsets_by_count(
                all_sequences,
                train_count=train_sequence_count or 1,
                eval_count=eval_sequence_count or 1,
                seed=seed,
            )
        else:
            train_sequences, eval_sequences = choose_sequence_subsets(
                all_sequences,
                train_fraction=train_fraction,
                eval_fraction=eval_fraction,
                seed=seed,
            )

    print(
        f"Sequence split: {len(train_sequences)} train / {len(eval_sequences)} eval "
        f"from {len(all_sequences)} available sequences"
    )

    train_dataset = build_dataset(
        data_root=data_root,
        split=split,
        sequences=train_sequences,
        input_frames=input_frames,
        horizon=horizon,
        stride=stride,
        image_size=image_size,
        max_frames=max_frames,
    )
    print(f"Train dataset: {len(train_dataset)} samples")

    selected_samples: list[dict] = []
    if overfit_samples:
        base_train_dataset = train_dataset
        selected_indices, selected_samples = select_sample_indices(
            base_train_dataset,
            count=overfit_samples,
            selection=sample_selection,
        )
        train_dataset = Subset(base_train_dataset, selected_indices)
        eval_dataset = Subset(base_train_dataset, selected_indices)
        print(
            f"Overfit subset: {len(train_dataset)} {sample_selection} sample(s): "
            f"{selected_samples}"
        )
    else:
        eval_dataset = build_dataset(
            data_root=data_root,
            split=split,
            sequences=eval_sequences,
            input_frames=input_frames,
            horizon=horizon,
            stride=stride,
            image_size=image_size,
            max_frames=max_frames,
        )
    print(f"Eval dataset: {len(eval_dataset)} samples")

    resolved_cache_visual_features = cache_visual_features or overfit_samples > 0
    manifest = {
        "seed": seed,
        "split": split,
        "available_sequences": len(all_sequences),
        "train_fraction": train_fraction,
        "eval_fraction": eval_fraction,
        "train_sequences": train_sequences,
        "eval_sequences": eval_sequences,
        "input_frames": input_frames,
        "horizon": horizon,
        "action_stride": action_stride,
        "stride": stride,
        "image_size": image_size,
        "max_frames": max_frames,
        "train_sequence_count": train_sequence_count,
        "eval_sequence_count": eval_sequence_count,
        "overfit_samples": overfit_samples,
        "sample_selection": sample_selection,
        "cache_visual_features": resolved_cache_visual_features,
        "feature_cache_batch_size": feature_cache_batch_size,
        "policy_loss_weight": policy_loss_weight,
        "selected_samples": selected_samples,
    }
    Path("sequence_manifest.json").write_text(json.dumps(manifest, indent=2))

    model = QwenFootballModel(
        model_name=model_name,
        horizon=horizon,
        input_frames=input_frames,
        action_stride=action_stride,
        freeze_backbone=True,
    )
    model.to(device)

    # Only train the projection + trajectory head
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    if resolved_cache_visual_features:
        print("Caching frozen Qwen visual features for this small training set...")
        train_dataset = cache_dataset_visual_features(
            model,
            train_dataset,
            device,
            batch_size=feature_cache_batch_size,
        )
        eval_dataset = cache_dataset_visual_features(
            model,
            eval_dataset,
            device,
            batch_size=feature_cache_batch_size,
        )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=loader_generator,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    trajectory_loss_fn = nn.SmoothL1Loss()

    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_policy_loss = 0.0
        n_batches = 0
        for batch in loader:
            bbox = batch["player_bbox"].to(device)
            target = batch["future_xy"].to(device)
            outputs = outputs_from_batch(model, batch, device)
            pred = outputs["future_xy"]
            action_labels = model.action_labels(bbox, target)
            teacher_action_logits = teacher_action_logits_from_batch(model, batch, device)
            trajectory_loss = trajectory_loss_fn(pred, target)
            policy_loss = F.cross_entropy(
                teacher_action_logits.reshape(-1, teacher_action_logits.shape[-1]),
                action_labels.reshape(-1),
            )
            loss = trajectory_loss + policy_loss_weight * policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_traj_loss += trajectory_loss.item()
            epoch_policy_loss += policy_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_traj_loss = epoch_traj_loss / max(n_batches, 1)
        avg_policy_loss = epoch_policy_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            train_metrics = evaluate(model, loader, device)
            eval_metrics = evaluate(model, eval_loader, device)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:4d} | loss={avg_loss:.6f} | "
                f"traj={avg_traj_loss:.6f} policy={avg_policy_loss:.6f} | "
                f"train_ADE={train_metrics['ADE']:.6f} train_FDE={train_metrics['FDE']:.6f} | "
                f"train_act={train_metrics['action_accuracy']:.3f} "
                f"train_teacher_act={train_metrics['teacher_action_accuracy']:.3f} | "
                f"eval_ADE={eval_metrics['ADE']:.6f} eval_FDE={eval_metrics['FDE']:.6f} | "
                f"eval_act={eval_metrics['action_accuracy']:.3f} "
                f"eval_teacher_act={eval_metrics['teacher_action_accuracy']:.3f} | "
                f"{elapsed:.1f}s"
            )
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "trajectory_loss": avg_traj_loss,
                "policy_loss": avg_policy_loss,
                "train_ADE": train_metrics["ADE"],
                "train_FDE": train_metrics["FDE"],
                "train_action_accuracy": train_metrics["action_accuracy"],
                "train_teacher_action_accuracy": train_metrics["teacher_action_accuracy"],
                "eval_ADE": eval_metrics["ADE"],
                "eval_FDE": eval_metrics["FDE"],
                "eval_action_accuracy": eval_metrics["action_accuracy"],
                "eval_teacher_action_accuracy": eval_metrics["teacher_action_accuracy"],
            })

    # Final eval
    final_train = evaluate(model, loader, device)
    final_eval = evaluate(model, eval_loader, device)
    final = {
        "train_ADE": final_train["ADE"],
        "train_FDE": final_train["FDE"],
        "eval_ADE": final_eval["ADE"],
        "eval_FDE": final_eval["FDE"],
        "train_action_accuracy": final_train["action_accuracy"],
        "eval_action_accuracy": final_eval["action_accuracy"],
        "train_teacher_action_accuracy": final_train["teacher_action_accuracy"],
        "eval_teacher_action_accuracy": final_eval["teacher_action_accuracy"],
    }
    print(
        "\nFinal: "
        f"train_ADE={final['train_ADE']:.6f} train_FDE={final['train_FDE']:.6f} | "
        f"train_act={final['train_action_accuracy']:.3f} "
        f"train_teacher_act={final['train_teacher_action_accuracy']:.3f} | "
        f"eval_ADE={final['eval_ADE']:.6f} eval_FDE={final['eval_FDE']:.6f} | "
        f"eval_act={final['eval_action_accuracy']:.3f} "
        f"eval_teacher_act={final['eval_teacher_action_accuracy']:.3f}"
    )
    save_prediction_artifacts(
        model=model,
        dataset=eval_dataset,
        device=device,
        preview_path=preview_path,
        rollouts_path=rollouts_path,
        split=split,
        input_frames=input_frames,
        horizon=horizon,
    )

    # Save history
    out_path = Path("training_history.json")
    out_path.write_text(json.dumps(history, indent=2))
    print(f"History saved to {out_path}")

    # Save model head
    torch.save(
        {"visual_proj": model.visual_proj.state_dict(),
         "step_embedding": model.step_embedding.state_dict(),
         "head": model.head.state_dict(),
         "action_codebook": model.action_codebook.cpu(),
         "action_stride": model.action_stride},
        "trajectory_head.pt",
    )
    print("Head weights saved to trajectory_head.pt")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-frames", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--action-stride", type=int, default=5)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--train-sequence-count", type=int, default=0)
    parser.add_argument("--eval-sequence-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--sample-selection", default="first", choices=["first", "max_motion"])
    parser.add_argument("--cache-visual-features", action="store_true")
    parser.add_argument("--feature-cache-batch-size", type=int, default=4)
    parser.add_argument("--policy-loss-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        sequence=args.sequence,
        split=args.split,
        input_frames=args.input_frames,
        horizon=args.horizon,
        action_stride=args.action_stride,
        stride=args.stride,
        image_size=args.image_size,
        max_frames=args.max_frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_fraction=args.train_fraction,
        eval_fraction=args.eval_fraction,
        train_sequence_count=args.train_sequence_count,
        eval_sequence_count=args.eval_sequence_count,
        seed=args.seed,
        overfit_samples=args.overfit_samples,
        sample_selection=args.sample_selection,
        cache_visual_features=args.cache_visual_features,
        feature_cache_batch_size=args.feature_cache_batch_size,
        policy_loss_weight=args.policy_loss_weight,
        device=args.device,
    )
