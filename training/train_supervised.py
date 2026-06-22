"""Supervised training loop for trajectory prediction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.tracking_dataset import SoccerNetTrackingDataset
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


def train(
    data_root: str,
    sequence: str = "SNMOT-116",
    split: str = "test",
    input_frames: int = 5,
    horizon: int = 25,
    stride: int = 50,
    image_size: int = 224,
    max_frames: int = 250,
    batch_size: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    log_every: int = 10,
):
    """Run supervised training to overfit on a small clip."""

    dataset = SoccerNetTrackingDataset(
        data_root=data_root,
        sequence=sequence,
        split=split,
        input_frames=input_frames,
        horizon=horizon,
        stride=stride,
        image_size=image_size,
        max_frames=max_frames,
    )
    print(f"Dataset: {len(dataset)} samples from {sequence}")
    if len(dataset) == 0:
        raise ValueError(
            "No training samples found "
            f"(data_root={data_root}, split={split}, sequence={sequence}, "
            f"input_frames={input_frames}, horizon={horizon}, max_frames={max_frames})"
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
            metrics = evaluate(model, loader, device)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:4d} | loss={avg_loss:.6f} | "
                f"ADE={metrics['ADE']:.6f} FDE={metrics['FDE']:.6f} | "
                f"{elapsed:.1f}s"
            )
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "ADE": metrics["ADE"],
                "FDE": metrics["FDE"],
            })

    # Final eval
    final = evaluate(model, loader, device)
    print(f"\nFinal: ADE={final['ADE']:.6f} FDE={final['FDE']:.6f}")

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
    parser.add_argument("--sequence", default="SNMOT-116")
    parser.add_argument("--split", default="test")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        sequence=args.sequence,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
