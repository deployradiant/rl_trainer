"""Supervised training loop for trajectory prediction."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from data.tracking_dataset import (
    MultiSequenceSoccerNetTrackingDataset,
    choose_sequence_subsets,
    choose_sequence_subsets_by_count,
    list_sequences,
)
from models.qwen_football import QwenFootballModel
from models.tracking_policy import TrackingChunkBaseline
from training.artifacts import save_prediction_artifacts
from training.baselines import batch_ade, batch_fde, constant_position, constant_velocity

QWEN_LR = 1e-5
TRACKING_LR = 1e-3
ADAM_EPS = 1e-4
GRAD_CLIP_NORM = 1.0
ENDPOINT_LOSS_WEIGHT = 1.0


class SampleListDataset(Dataset):
    """Small in-memory dataset for selected rollout artifact samples."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad_norm = param.grad.detach().float().norm().item()
        total_sq += grad_norm * grad_norm
    return total_sq ** 0.5


def metric_text(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    tracking_model: nn.Module | None = None,
) -> dict:
    was_training = model.training
    tracking_was_training = tracking_model.training if tracking_model is not None else False
    model.eval()
    if tracking_model is not None:
        tracking_model.eval()
    sums = {
        "ADE": 0.0,
        "FDE": 0.0,
        "tracking_ADE": 0.0,
        "tracking_FDE": 0.0,
        "baseline_constant_ADE": 0.0,
        "baseline_constant_FDE": 0.0,
        "baseline_velocity_ADE": 0.0,
        "baseline_velocity_FDE": 0.0,
    }
    sample_count = 0
    action_correct = 0
    action_total = 0
    teacher_action_correct = 0
    teacher_action_total = 0
    value_action_correct = 0
    value_action_total = 0
    try:
        with torch.no_grad():
            for batch in loader:
                frames = batch["frames"].to(device)
                bbox = batch["player_bbox"].to(device)
                target = batch["future_xy"].to(device)
                visual_features = model.extract_visual_features(frames)
                outputs = model.predict_from_visual_features(
                    visual_features,
                    bbox,
                    return_outputs=True,
                    greedy=True,
                )
                pred = outputs["future_xy"]
                tracking_pred = tracking_model(bbox) if tracking_model is not None else None
                constant_pred = constant_position(bbox, target.shape[1])
                velocity_pred = constant_velocity(bbox, target.shape[1])
                batch_size = target.shape[0]
                sample_count += batch_size
                sums["ADE"] += float(batch_ade(pred, target).sum().item())
                sums["FDE"] += float(batch_fde(pred, target).sum().item())
                if tracking_pred is not None:
                    sums["tracking_ADE"] += float(batch_ade(tracking_pred, target).sum().item())
                    sums["tracking_FDE"] += float(batch_fde(tracking_pred, target).sum().item())
                sums["baseline_constant_ADE"] += float(batch_ade(constant_pred, target).sum().item())
                sums["baseline_constant_FDE"] += float(batch_fde(constant_pred, target).sum().item())
                sums["baseline_velocity_ADE"] += float(batch_ade(velocity_pred, target).sum().item())
                sums["baseline_velocity_FDE"] += float(batch_fde(velocity_pred, target).sum().item())
                if "action_logits" in outputs:
                    labels = model.action_labels(bbox, target)
                    action_pred = outputs["action_logits"].argmax(dim=-1)
                    action_correct += int((action_pred == labels).sum().item())
                    action_total += int(labels.numel())
                    teacher_action_pred = model.teacher_forced_action_logits(
                        visual_features,
                        bbox,
                        target,
                    ).argmax(dim=-1)
                    teacher_action_correct += int((teacher_action_pred == labels).sum().item())
                    teacher_action_total += int(labels.numel())
                    value_action_pred = model.teacher_forced_action_values(
                        visual_features,
                        bbox,
                        target,
                    ).argmax(dim=-1)
                    value_action_correct += int((value_action_pred == labels).sum().item())
                    value_action_total += int(labels.numel())
    finally:
        if was_training:
            model.train()
        if tracking_model is not None and tracking_was_training:
            tracking_model.train()
    metrics = {key: value / max(1, sample_count) for key, value in sums.items()}
    if tracking_model is None:
        metrics["tracking_ADE"] = None
        metrics["tracking_FDE"] = None
    metrics.update({
        "action_accuracy": action_correct / action_total if action_total else None,
        "teacher_action_accuracy": teacher_action_correct / teacher_action_total if teacher_action_total else None,
        "value_action_accuracy": value_action_correct / value_action_total if value_action_total else None,
    })
    return metrics


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


def choose_extra_rollout_sequences(
    all_sequences: list[str],
    train_sequences: list[str],
    eval_sequences: list[str],
    count: int,
    seed: int,
) -> list[str]:
    """Pick deterministic rollout-only sequences outside the training set."""
    if count <= 0:
        return []
    chosen = []
    excluded = set(train_sequences)
    for sequence in eval_sequences:
        if sequence not in excluded and sequence not in chosen:
            chosen.append(sequence)
            if len(chosen) == count:
                return chosen

    candidates = [
        sequence for sequence in all_sequences
        if sequence not in excluded and sequence not in chosen
    ]
    random.Random(seed).shuffle(candidates)
    chosen.extend(candidates[:max(0, count - len(chosen))])
    return sorted(chosen)


def build_dataset(
    data_root: str,
    split: str,
    sequences: list[str],
    input_frames: int,
    visual_frames: int,
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
        visual_frames=visual_frames,
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
    input_frames: int = 25,
    visual_frames: int = 5,
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
    extra_rollout_sequence_count: int = 0,
    policy_kind: str = "chunk",
    endpoint_loss_weight: float = ENDPOINT_LOSS_WEIGHT,
    tracking_lr: float = TRACKING_LR,
    policy_loss_weight: float = 0.05,
    value_loss_weight: float = 1.0,
    value_rank_loss_weight: float = 1.0,
    value_fde_weight: float = 0.5,
    value_target_scale: float = 10.0,
    flow_steps: int = 8,
    flow_noise_scale: float = 0.05,
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    log_every: int = 10,
    preview_path: str = "trajectory_preview.png",
    rollouts_path: str = "trajectory_rollouts.json",
    mcts_simulations: int = 0,
    mcts_top_k: int = 8,
    mcts_c_puct: float = 1.5,
    save_checkpoint: bool = False,
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
        visual_frames=visual_frames,
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
            visual_frames=visual_frames,
            horizon=horizon,
            stride=stride,
            image_size=image_size,
            max_frames=max_frames,
        )
    print(f"Eval dataset: {len(eval_dataset)} samples")

    extra_rollout_sequences = choose_extra_rollout_sequences(
        all_sequences=all_sequences,
        train_sequences=train_sequences,
        eval_sequences=eval_sequences,
        count=extra_rollout_sequence_count,
        seed=seed,
    )
    extra_rollout_samples: list[dict] = []
    for rollout_sequence in extra_rollout_sequences:
        rollout_dataset = build_dataset(
            data_root=data_root,
            split=split,
            sequences=[rollout_sequence],
            input_frames=input_frames,
            visual_frames=visual_frames,
            horizon=horizon,
            stride=stride,
            image_size=image_size,
            max_frames=max_frames,
        )
        rollout_indices, _ = select_sample_indices(
            rollout_dataset,
            count=1,
            selection=sample_selection,
        )
        if rollout_indices:
            extra_rollout_samples.append(rollout_dataset[rollout_indices[0]])
    if extra_rollout_samples:
        print(
            "Extra rollout sequences: "
            f"{extra_rollout_sequences} ({len(extra_rollout_samples)} sample(s))"
        )

    manifest = {
        "seed": seed,
        "split": split,
        "available_sequences": len(all_sequences),
        "train_fraction": train_fraction,
        "eval_fraction": eval_fraction,
        "train_sequences": train_sequences,
        "eval_sequences": eval_sequences,
        "input_frames": input_frames,
        "visual_frames": visual_frames,
        "horizon": horizon,
        "action_stride": action_stride,
        "stride": stride,
        "image_size": image_size,
        "max_frames": max_frames,
        "train_sequence_count": train_sequence_count,
        "eval_sequence_count": eval_sequence_count,
        "overfit_samples": overfit_samples,
        "sample_selection": sample_selection,
        "extra_rollout_sequence_count": extra_rollout_sequence_count,
        "extra_rollout_sequences": extra_rollout_sequences,
        "policy_kind": policy_kind,
        "endpoint_loss_weight": endpoint_loss_weight,
        "tracking_lr": tracking_lr,
        "policy_loss_weight": policy_loss_weight,
        "value_loss_weight": value_loss_weight,
        "value_rank_loss_weight": value_rank_loss_weight,
        "value_fde_weight": value_fde_weight,
        "value_target_scale": value_target_scale,
        "flow_steps": flow_steps,
        "flow_noise_scale": flow_noise_scale,
        "qwen_lr": QWEN_LR,
        "adam_eps": ADAM_EPS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "selected_samples": selected_samples,
        "mcts_simulations": mcts_simulations,
        "mcts_top_k": mcts_top_k,
        "mcts_c_puct": mcts_c_puct,
        "save_checkpoint": save_checkpoint,
    }
    Path("sequence_manifest.json").write_text(json.dumps(manifest, indent=2))

    model = QwenFootballModel(
        model_name=model_name,
        horizon=horizon,
        input_frames=input_frames,
        visual_frames=visual_frames,
        action_stride=action_stride,
        policy_kind=policy_kind,
        flow_steps=flow_steps,
        flow_noise_scale=flow_noise_scale,
    )
    model.to(device)
    tracking_model = TrackingChunkBaseline(horizon=horizon)
    tracking_model.to(device)

    qwen_trainable = [p for p in model.vlm.parameters() if p.requires_grad]
    head_trainable = [
        param for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("vlm.")
    ]
    trainable = head_trainable + qwen_trainable
    if not qwen_trainable:
        raise ValueError("Qwen training is required, but no Qwen parameters are trainable")
    if not trainable:
        raise ValueError("No trainable parameters selected")
    n_head_trainable = sum(p.numel() for p in head_trainable)
    n_qwen_trainable = sum(p.numel() for p in qwen_trainable)
    n_tracking_trainable = sum(p.numel() for p in tracking_model.parameters())
    n_trainable = n_head_trainable + n_qwen_trainable
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")
    print(f"Head trainable params: {n_head_trainable:,}")
    print(f"Tracking baseline trainable params: {n_tracking_trainable:,}")
    print(f"Qwen trainable params: {n_qwen_trainable:,}")
    print(f"Qwen LR: {QWEN_LR}")
    print(f"Tracking LR: {tracking_lr}")
    names = model.qwen_trainable_parameter_names()
    preview = ", ".join(names[:8])
    suffix = " ..." if len(names) > 8 else ""
    print(f"Qwen trainable tensors: {preview}{suffix}")
    manifest["qwen_trainable_params"] = n_qwen_trainable
    manifest["tracking_trainable_params"] = n_tracking_trainable
    manifest["qwen_trainable_tensors"] = names
    Path("sequence_manifest.json").write_text(json.dumps(manifest, indent=2))

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

    optimizer_groups = [{"params": head_trainable, "lr": lr}]
    if qwen_trainable:
        optimizer_groups.append({"params": qwen_trainable, "lr": QWEN_LR})
    optimizer = torch.optim.AdamW(optimizer_groups, eps=ADAM_EPS)
    tracking_optimizer = torch.optim.AdamW(tracking_model.parameters(), lr=tracking_lr, eps=ADAM_EPS)
    trajectory_loss_fn = nn.SmoothL1Loss()

    history = []
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_traj_loss = 0.0
        epoch_endpoint_loss = 0.0
        epoch_tracking_loss = 0.0
        epoch_tracking_endpoint_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_value_rank_loss = 0.0
        epoch_qwen_grad_norm = 0.0
        n_batches = 0
        for batch in loader:
            frames = batch["frames"].to(device)
            bbox = batch["player_bbox"].to(device)
            target = batch["future_xy"].to(device)

            tracking_pred = tracking_model(bbox)
            tracking_trajectory_loss = trajectory_loss_fn(tracking_pred, target)
            tracking_endpoint_loss = trajectory_loss_fn(tracking_pred[:, -1], target[:, -1])
            tracking_loss = tracking_trajectory_loss + endpoint_loss_weight * tracking_endpoint_loss
            if not torch.isfinite(tracking_loss):
                raise FloatingPointError(f"Non-finite tracking loss at epoch {epoch}")
            tracking_optimizer.zero_grad()
            tracking_loss.backward()
            torch.nn.utils.clip_grad_norm_(tracking_model.parameters(), GRAD_CLIP_NORM)
            tracking_optimizer.step()

            visual_features = model.extract_visual_features(frames)
            outputs = model.predict_from_visual_features(
                visual_features,
                bbox,
                return_outputs=True,
            )
            pred = outputs["future_xy"]
            trajectory_loss = trajectory_loss_fn(pred, target)
            endpoint_loss = trajectory_loss_fn(pred[:, -1], target[:, -1])
            policy_loss = torch.zeros((), dtype=trajectory_loss.dtype, device=device)
            value_loss = torch.zeros((), dtype=trajectory_loss.dtype, device=device)
            value_rank_loss = torch.zeros((), dtype=trajectory_loss.dtype, device=device)
            if policy_kind == "chunk":
                loss = trajectory_loss + endpoint_loss_weight * endpoint_loss
            elif policy_kind == "flow":
                flow_outputs = model.flow_matching_outputs(visual_features, bbox, target)
                policy_loss = trajectory_loss_fn(
                    flow_outputs["pred_velocity"],
                    flow_outputs["target_velocity"],
                )
                loss = policy_loss + trajectory_loss + endpoint_loss_weight * endpoint_loss
            else:
                action_labels = model.action_labels(bbox, target)
                teacher_action_logits = model.teacher_forced_action_logits(visual_features, bbox, target)
                teacher_action_values = model.teacher_forced_action_values(visual_features, bbox, target)
                action_value_targets = model.action_value_targets(
                    bbox,
                    target,
                    terminal_weight=value_fde_weight,
                    target_scale=value_target_scale,
                )
                policy_loss = F.cross_entropy(
                    teacher_action_logits.reshape(-1, teacher_action_logits.shape[-1]),
                    action_labels.reshape(-1),
                )
                value_loss = F.mse_loss(teacher_action_values, action_value_targets)
                value_rank_loss = F.cross_entropy(
                    teacher_action_values.reshape(-1, teacher_action_values.shape[-1]),
                    action_labels.reshape(-1),
                )
                loss = (
                    trajectory_loss
                    + policy_loss_weight * policy_loss
                    + value_loss_weight * value_loss
                    + value_rank_loss_weight * value_rank_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")

            optimizer.zero_grad()
            loss.backward()
            epoch_qwen_grad_norm += parameter_grad_norm(qwen_trainable)
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_traj_loss += trajectory_loss.item()
            epoch_endpoint_loss += endpoint_loss.item()
            epoch_tracking_loss += tracking_loss.item()
            epoch_tracking_endpoint_loss += tracking_endpoint_loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_value_rank_loss += value_rank_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_traj_loss = epoch_traj_loss / max(n_batches, 1)
        avg_endpoint_loss = epoch_endpoint_loss / max(n_batches, 1)
        avg_tracking_loss = epoch_tracking_loss / max(n_batches, 1)
        avg_tracking_endpoint_loss = epoch_tracking_endpoint_loss / max(n_batches, 1)
        avg_policy_loss = epoch_policy_loss / max(n_batches, 1)
        avg_value_loss = epoch_value_loss / max(n_batches, 1)
        avg_value_rank_loss = epoch_value_rank_loss / max(n_batches, 1)
        avg_qwen_grad_norm = epoch_qwen_grad_norm / max(n_batches, 1)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            train_metrics = evaluate(model, loader, device, tracking_model=tracking_model)
            eval_metrics = evaluate(model, eval_loader, device, tracking_model=tracking_model)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch:4d} | loss={avg_loss:.6f} | "
                f"traj={avg_traj_loss:.6f} endpoint={avg_endpoint_loss:.6f} "
                f"tracking={avg_tracking_loss:.6f} tracking_endpoint={avg_tracking_endpoint_loss:.6f} "
                f"policy={avg_policy_loss:.6f} "
                f"value={avg_value_loss:.6f} value_rank={avg_value_rank_loss:.6f} "
                f"qwen_grad={avg_qwen_grad_norm:.6f} | "
                f"train_ADE={train_metrics['ADE']:.6f} train_FDE={train_metrics['FDE']:.6f} | "
                f"train_tracking_ADE={train_metrics['tracking_ADE']:.6f} "
                f"train_vel_ADE={train_metrics['baseline_velocity_ADE']:.6f} "
                f"train_act={metric_text(train_metrics['action_accuracy'])} "
                f"train_teacher_act={metric_text(train_metrics['teacher_action_accuracy'])} "
                f"train_value_act={metric_text(train_metrics['value_action_accuracy'])} | "
                f"eval_ADE={eval_metrics['ADE']:.6f} eval_FDE={eval_metrics['FDE']:.6f} | "
                f"eval_tracking_ADE={eval_metrics['tracking_ADE']:.6f} "
                f"eval_vel_ADE={eval_metrics['baseline_velocity_ADE']:.6f} "
                f"eval_act={metric_text(eval_metrics['action_accuracy'])} "
                f"eval_teacher_act={metric_text(eval_metrics['teacher_action_accuracy'])} "
                f"eval_value_act={metric_text(eval_metrics['value_action_accuracy'])} | "
                f"{elapsed:.1f}s"
            )
            history.append({
                "epoch": epoch,
                "loss": avg_loss,
                "trajectory_loss": avg_traj_loss,
                "endpoint_loss": avg_endpoint_loss,
                "tracking_loss": avg_tracking_loss,
                "tracking_endpoint_loss": avg_tracking_endpoint_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "value_rank_loss": avg_value_rank_loss,
                "qwen_grad_norm": avg_qwen_grad_norm,
                "train_ADE": train_metrics["ADE"],
                "train_FDE": train_metrics["FDE"],
                "train_tracking_ADE": train_metrics["tracking_ADE"],
                "train_tracking_FDE": train_metrics["tracking_FDE"],
                "train_baseline_constant_ADE": train_metrics["baseline_constant_ADE"],
                "train_baseline_constant_FDE": train_metrics["baseline_constant_FDE"],
                "train_baseline_velocity_ADE": train_metrics["baseline_velocity_ADE"],
                "train_baseline_velocity_FDE": train_metrics["baseline_velocity_FDE"],
                "train_action_accuracy": train_metrics["action_accuracy"],
                "train_teacher_action_accuracy": train_metrics["teacher_action_accuracy"],
                "train_value_action_accuracy": train_metrics["value_action_accuracy"],
                "eval_ADE": eval_metrics["ADE"],
                "eval_FDE": eval_metrics["FDE"],
                "eval_tracking_ADE": eval_metrics["tracking_ADE"],
                "eval_tracking_FDE": eval_metrics["tracking_FDE"],
                "eval_baseline_constant_ADE": eval_metrics["baseline_constant_ADE"],
                "eval_baseline_constant_FDE": eval_metrics["baseline_constant_FDE"],
                "eval_baseline_velocity_ADE": eval_metrics["baseline_velocity_ADE"],
                "eval_baseline_velocity_FDE": eval_metrics["baseline_velocity_FDE"],
                "eval_action_accuracy": eval_metrics["action_accuracy"],
                "eval_teacher_action_accuracy": eval_metrics["teacher_action_accuracy"],
                "eval_value_action_accuracy": eval_metrics["value_action_accuracy"],
            })

    # Final eval
    final_train = evaluate(model, loader, device, tracking_model=tracking_model)
    final_eval = evaluate(model, eval_loader, device, tracking_model=tracking_model)
    final = {
        "train_ADE": final_train["ADE"],
        "train_FDE": final_train["FDE"],
        "train_tracking_ADE": final_train["tracking_ADE"],
        "train_tracking_FDE": final_train["tracking_FDE"],
        "train_baseline_constant_ADE": final_train["baseline_constant_ADE"],
        "train_baseline_constant_FDE": final_train["baseline_constant_FDE"],
        "train_baseline_velocity_ADE": final_train["baseline_velocity_ADE"],
        "train_baseline_velocity_FDE": final_train["baseline_velocity_FDE"],
        "eval_ADE": final_eval["ADE"],
        "eval_FDE": final_eval["FDE"],
        "eval_tracking_ADE": final_eval["tracking_ADE"],
        "eval_tracking_FDE": final_eval["tracking_FDE"],
        "eval_baseline_constant_ADE": final_eval["baseline_constant_ADE"],
        "eval_baseline_constant_FDE": final_eval["baseline_constant_FDE"],
        "eval_baseline_velocity_ADE": final_eval["baseline_velocity_ADE"],
        "eval_baseline_velocity_FDE": final_eval["baseline_velocity_FDE"],
        "train_action_accuracy": final_train["action_accuracy"],
        "eval_action_accuracy": final_eval["action_accuracy"],
        "train_teacher_action_accuracy": final_train["teacher_action_accuracy"],
        "eval_teacher_action_accuracy": final_eval["teacher_action_accuracy"],
        "train_value_action_accuracy": final_train["value_action_accuracy"],
        "eval_value_action_accuracy": final_eval["value_action_accuracy"],
        "qwen_trainable_params": n_qwen_trainable,
    }
    print(
        "\nFinal: "
        f"train_ADE={final['train_ADE']:.6f} train_FDE={final['train_FDE']:.6f} | "
        f"train_tracking_ADE={final['train_tracking_ADE']:.6f} "
        f"train_vel_ADE={final['train_baseline_velocity_ADE']:.6f} "
        f"train_act={metric_text(final['train_action_accuracy'])} "
        f"train_teacher_act={metric_text(final['train_teacher_action_accuracy'])} "
        f"train_value_act={metric_text(final['train_value_action_accuracy'])} | "
        f"eval_ADE={final['eval_ADE']:.6f} eval_FDE={final['eval_FDE']:.6f} | "
        f"eval_tracking_ADE={final['eval_tracking_ADE']:.6f} "
        f"eval_vel_ADE={final['eval_baseline_velocity_ADE']:.6f} "
        f"eval_act={metric_text(final['eval_action_accuracy'])} "
        f"eval_teacher_act={metric_text(final['eval_teacher_action_accuracy'])} "
        f"eval_value_act={metric_text(final['eval_value_action_accuracy'])}"
    )
    artifact_dataset = eval_dataset
    if extra_rollout_samples:
        artifact_samples = [eval_dataset[idx] for idx in range(len(eval_dataset))]
        artifact_samples.extend(extra_rollout_samples)
        artifact_dataset = SampleListDataset(artifact_samples)

    save_prediction_artifacts(
        model=model,
        dataset=artifact_dataset,
        device=device,
        preview_path=preview_path,
        rollouts_path=rollouts_path,
        split=split,
        input_frames=input_frames,
        horizon=horizon,
        tracking_model=tracking_model,
        mcts_simulations=mcts_simulations,
        mcts_top_k=mcts_top_k,
        mcts_c_puct=mcts_c_puct,
    )

    # Save history
    out_path = Path("training_history.json")
    out_path.write_text(json.dumps(history, indent=2))
    print(f"History saved to {out_path}")

    if save_checkpoint:
        # Save trainable task weights plus the Qwen tensors updated by this run.
        checkpoint = {
            "visual_proj": model.visual_proj.state_dict(),
            "policy_kind": policy_kind,
            "chunk_head": model.chunk_head.state_dict(),
            "flow_head": model.flow_head.state_dict(),
            "tracking_baseline": tracking_model.state_dict(),
            "step_embedding": model.step_embedding.state_dict(),
            "head": model.head.state_dict(),
            "value_head": model.value_head.state_dict(),
            "action_codebook": model.action_codebook.cpu(),
            "action_stride": model.action_stride,
            "flow_steps": model.flow_steps,
            "flow_noise_scale": model.flow_noise_scale,
            "qwen_lr": QWEN_LR,
            "adam_eps": ADAM_EPS,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "qwen_trainable_names": names,
            "qwen_trainable": {
                name: param.detach().cpu()
                for name, param in model.vlm.named_parameters()
                if param.requires_grad
            },
        }
        torch.save(checkpoint, "trajectory_head.pt")
        print("Trainable weights saved to trajectory_head.pt")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-frames", type=int, default=25)
    parser.add_argument("--visual-frames", type=int, default=5)
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
    parser.add_argument("--extra-rollout-sequence-count", type=int, default=0)
    parser.add_argument("--policy-kind", default="chunk", choices=["chunk", "codebook", "flow"])
    parser.add_argument("--endpoint-loss-weight", type=float, default=ENDPOINT_LOSS_WEIGHT)
    parser.add_argument("--tracking-lr", type=float, default=TRACKING_LR)
    parser.add_argument("--policy-loss-weight", type=float, default=0.05)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-rank-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-fde-weight", type=float, default=0.5)
    parser.add_argument("--value-target-scale", type=float, default=10.0)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--flow-noise-scale", type=float, default=0.05)
    parser.add_argument("--mcts-simulations", type=int, default=0)
    parser.add_argument("--mcts-top-k", type=int, default=8)
    parser.add_argument("--mcts-c-puct", type=float, default=1.5)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        sequence=args.sequence,
        split=args.split,
        input_frames=args.input_frames,
        visual_frames=args.visual_frames,
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
        extra_rollout_sequence_count=args.extra_rollout_sequence_count,
        policy_kind=args.policy_kind,
        endpoint_loss_weight=args.endpoint_loss_weight,
        tracking_lr=args.tracking_lr,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        value_rank_loss_weight=args.value_rank_loss_weight,
        value_fde_weight=args.value_fde_weight,
        value_target_scale=args.value_target_scale,
        flow_steps=args.flow_steps,
        flow_noise_scale=args.flow_noise_scale,
        mcts_simulations=args.mcts_simulations,
        mcts_top_k=args.mcts_top_k,
        mcts_c_puct=args.mcts_c_puct,
        save_checkpoint=args.save_checkpoint,
        device=args.device,
    )
