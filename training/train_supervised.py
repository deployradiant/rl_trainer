"""Supervised training loop for trajectory prediction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from data.tracking_dataset import (
    MultiSequenceSoccerNetTrackingDataset,
    choose_sequence_subsets,
    list_sequences,
)
from models.qwen_football import QwenFootballModel


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    """Average Displacement Error over the future window."""
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    """Final Displacement Error at the last timestep."""
    return float(np.linalg.norm(pred[-1] - target[-1]))


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    ades, fdes = [], []
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            bbox = batch["player_bbox"].to(device)
            target = batch["future_xy"].to(device)
            pred = model(frames, bbox)
            pred_np = pred.cpu().numpy()
            target_np = target.cpu().numpy()
            for i in range(len(pred_np)):
                ades.append(ade(pred_np[i], target_np[i]))
                fdes.append(fde(pred_np[i], target_np[i]))
    return {"ADE": float(np.mean(ades)), "FDE": float(np.mean(fdes))}


def save_prediction_artifacts(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str,
    preview_path: str,
    rollout_path: str,
    split: str,
    input_frames: int,
    horizon: int,
) -> None:
    """Save visual artifacts for one held-out eval rollout."""
    if len(dataset) == 0:
        return

    sample = dataset[0]
    frames = sample["frames"].unsqueeze(0).to(device)
    bbox = sample["player_bbox"].unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pred = model(frames, bbox)[0].detach().cpu().numpy()
    if was_training:
        model.train()
    target = sample["future_xy"].numpy()
    input_bboxes = sample["player_bbox"].numpy()

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

    preview_out = Path(preview_path)
    preview_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_out)
    print(f"Preview saved to {preview_path}")

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
        "fps": 25,
        "current_index": input_frames - 1,
        "frame_numbers": input_frame_numbers + future_frame_numbers,
        "input_frame_numbers": input_frame_numbers,
        "future_frame_numbers": future_frame_numbers,
        "input_bboxes": input_bboxes.tolist(),
        "actual_xy": actual_path.tolist(),
        "pred_xy": pred_path.tolist(),
    }
    rollout_out = Path(rollout_path)
    rollout_out.parent.mkdir(parents=True, exist_ok=True)
    rollout_out.write_text(json.dumps(rollout, indent=2))
    print(f"Rollout saved to {rollout_path}")


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
    stride: int = 50,
    image_size: int = 224,
    max_frames: int = 250,
    batch_size: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    seed: int = 7,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    log_every: int = 10,
    preview_path: str = "trajectory_preview.png",
    rollout_path: str = "trajectory_rollout.json",
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
        train_sequences, eval_sequences = choose_sequence_subsets(
            all_sequences,
            train_fraction=train_fraction,
            eval_fraction=eval_fraction,
            seed=seed,
        )

    manifest = {
        "seed": seed,
        "split": split,
        "available_sequences": len(all_sequences),
        "train_fraction": train_fraction,
        "eval_fraction": eval_fraction,
        "train_sequences": train_sequences,
        "eval_sequences": eval_sequences,
    }
    Path("sequence_manifest.json").write_text(json.dumps(manifest, indent=2))
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
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Eval dataset: {len(eval_dataset)} samples")

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

    model = QwenFootballModel(
        model_name=model_name,
        horizon=horizon,
        input_frames=input_frames,
        freeze_backbone=True,
    )
    model.to(device)

    # Only train the projection + trajectory head
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = nn.SmoothL1Loss()

    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            bbox = batch["player_bbox"].to(device)
            target = batch["future_xy"].to(device)

            pred = model(frames, bbox)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            train_metrics = evaluate(model, loader, device)
            eval_metrics = evaluate(model, eval_loader, device)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:4d} | loss={avg_loss:.6f} | "
                f"train_ADE={train_metrics['ADE']:.6f} train_FDE={train_metrics['FDE']:.6f} | "
                f"eval_ADE={eval_metrics['ADE']:.6f} eval_FDE={eval_metrics['FDE']:.6f} | "
                f"{elapsed:.1f}s"
            )
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "train_ADE": train_metrics["ADE"],
                "train_FDE": train_metrics["FDE"],
                "eval_ADE": eval_metrics["ADE"],
                "eval_FDE": eval_metrics["FDE"],
            })

    # Final eval
    final_train = evaluate(model, loader, device)
    final_eval = evaluate(model, eval_loader, device)
    final = {
        "train_ADE": final_train["ADE"],
        "train_FDE": final_train["FDE"],
        "eval_ADE": final_eval["ADE"],
        "eval_FDE": final_eval["FDE"],
    }
    print(
        "\nFinal: "
        f"train_ADE={final['train_ADE']:.6f} train_FDE={final['train_FDE']:.6f} | "
        f"eval_ADE={final['eval_ADE']:.6f} eval_FDE={final['eval_FDE']:.6f}"
    )
    save_prediction_artifacts(
        model=model,
        dataset=eval_dataset,
        device=device,
        preview_path=preview_path,
        rollout_path=rollout_path,
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
         "head": model.head.state_dict()},
        "trajectory_head.pt",
    )
    print("Head weights saved to trajectory_head.pt")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-fraction", type=float, default=0.10)
    parser.add_argument("--eval-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        sequence=args.sequence,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_fraction=args.train_fraction,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        device=args.device,
    )
