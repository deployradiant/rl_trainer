"""Modal entrypoint: train trajectory prediction on SoccerNet sequence subsets."""

import modal

app = modal.App("rl-trainer-overfit")

# Volume to cache SoccerNet data and HuggingFace models
data_vol = modal.Volume.from_name("soccernet-tracking", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "transformers==4.45.0",
        "Pillow",
        "numpy",
        "qwen-vl-utils",
        "accelerate",
        "soccerNet",
    )
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .add_local_dir(".", "/root")
)


@app.function(
    gpu="H100",
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache},
    image=image,
    timeout=21600,
)
def run_overfit(
    sequence: str = "",
    split: str = "train",
    input_frames: int = 25,
    visual_frames: int = 5,
    horizon: int = 25,
    action_stride: int = 5,
    stride: int = 50,
    image_size: int = 224,
    max_frames: int = 250,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    train_sequence_count: int = 0,
    eval_sequence_count: int = 0,
    seed: int = 7,
    overfit_samples: int = 0,
    sample_selection: str = "first",
    policy_kind: str = "chunk",
    endpoint_loss_weight: float = 1.0,
    tracking_lr: float = 1e-3,
    policy_loss_weight: float = 0.05,
    value_loss_weight: float = 1.0,
    value_rank_loss_weight: float = 1.0,
    value_fde_weight: float = 0.5,
    value_target_scale: float = 10.0,
    flow_steps: int = 8,
    flow_noise_scale: float = 0.05,
    mcts_simulations: int = 0,
    mcts_top_k: int = 8,
    mcts_c_puct: float = 1.5,
    save_checkpoint: bool = False,
):
    import os
    import sys

    # Ensure local modules are importable
    sys.path.insert(0, "/root")
    from training.train_supervised import train

    data_root = "/data/tracking-2023"

    # Download SoccerNet split if not cached
    zip_path = f"{data_root}/{split}.zip"
    if not os.path.exists(zip_path):
        print(f"Downloading SoccerNet tracking {split} split...")
        os.makedirs(data_root, exist_ok=True)
        from SoccerNet.Downloader import SoccerNetDownloader
        dl = SoccerNetDownloader(LocalDirectory="/data/SoccerNet")
        dl.downloadDataTask(task="tracking-2023", split=[split], verbose=True)
        # Move to expected path
        import shutil
        shutil.move(f"/data/SoccerNet/tracking-2023/{split}.zip", zip_path)
        data_vol.commit()

    results = train(
        data_root=data_root,
        sequence=sequence,
        split=split,
        input_frames=input_frames,
        visual_frames=visual_frames,
        horizon=horizon,
        action_stride=action_stride,
        stride=stride,
        image_size=image_size,
        max_frames=max_frames,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        train_sequence_count=train_sequence_count,
        eval_sequence_count=eval_sequence_count,
        seed=seed,
        overfit_samples=overfit_samples,
        sample_selection=sample_selection,
        policy_kind=policy_kind,
        endpoint_loss_weight=endpoint_loss_weight,
        tracking_lr=tracking_lr,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        value_rank_loss_weight=value_rank_loss_weight,
        value_fde_weight=value_fde_weight,
        value_target_scale=value_target_scale,
        flow_steps=flow_steps,
        flow_noise_scale=flow_noise_scale,
        mcts_simulations=mcts_simulations,
        mcts_top_k=mcts_top_k,
        mcts_c_puct=mcts_c_puct,
        save_checkpoint=save_checkpoint,
        device="cuda",
    )

    # Save artifacts to volume
    import shutil
    artifact_files = [
        "sequence_manifest.json",
        "training_history.json",
        "trajectory_preview.png",
        "trajectory_rollouts.json",
    ]
    if save_checkpoint:
        artifact_files.append("trajectory_head.pt")
    for fname in artifact_files:
        if os.path.exists(fname):
            shutil.copy(fname, f"/data/{fname}")
    data_vol.commit()

    return results


@app.local_entrypoint()
def main(
    sequence: str = "",
    split: str = "train",
    input_frames: int = 25,
    visual_frames: int = 5,
    horizon: int = 25,
    action_stride: int = 5,
    stride: int = 50,
    image_size: int = 224,
    max_frames: int = 250,
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    train_sequence_count: int = 0,
    eval_sequence_count: int = 0,
    seed: int = 7,
    overfit_samples: int = 0,
    sample_selection: str = "first",
    policy_kind: str = "chunk",
    endpoint_loss_weight: float = 1.0,
    tracking_lr: float = 1e-3,
    policy_loss_weight: float = 0.05,
    value_loss_weight: float = 1.0,
    value_rank_loss_weight: float = 1.0,
    value_fde_weight: float = 0.5,
    value_target_scale: float = 10.0,
    flow_steps: int = 8,
    flow_noise_scale: float = 0.05,
    mcts_simulations: int = 0,
    mcts_top_k: int = 8,
    mcts_c_puct: float = 1.5,
    save_checkpoint: bool = False,
):
    results = run_overfit.remote(
        sequence=sequence,
        split=split,
        input_frames=input_frames,
        visual_frames=visual_frames,
        horizon=horizon,
        action_stride=action_stride,
        stride=stride,
        image_size=image_size,
        max_frames=max_frames,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        train_sequence_count=train_sequence_count,
        eval_sequence_count=eval_sequence_count,
        seed=seed,
        overfit_samples=overfit_samples,
        sample_selection=sample_selection,
        policy_kind=policy_kind,
        endpoint_loss_weight=endpoint_loss_weight,
        tracking_lr=tracking_lr,
        policy_loss_weight=policy_loss_weight,
        value_loss_weight=value_loss_weight,
        value_rank_loss_weight=value_rank_loss_weight,
        value_fde_weight=value_fde_weight,
        value_target_scale=value_target_scale,
        flow_steps=flow_steps,
        flow_noise_scale=flow_noise_scale,
        mcts_simulations=mcts_simulations,
        mcts_top_k=mcts_top_k,
        mcts_c_puct=mcts_c_puct,
        save_checkpoint=save_checkpoint,
    )
    print(f"\nModal training complete: {results}")
