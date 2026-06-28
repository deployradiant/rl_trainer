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
    gpu="A10G",
    volumes={"/data": data_vol, "/root/.cache/huggingface": hf_cache},
    image=image,
    timeout=21600,
)
def run_overfit(
    sequence: str = "",
    split: str = "train",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    seed: int = 7,
):
    import os
    import sys

    # Ensure local modules are importable
    sys.path.insert(0, "/root")
    from training.train_supervised import train

    data_root = "/data/tracking-2023"
    max_frames = 250

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
        max_frames=max_frames,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        seed=seed,
        device="cuda",
    )

    # Save artifacts to volume
    import shutil
    for fname in [
        "sequence_manifest.json",
        "training_history.json",
        "trajectory_head.pt",
        "trajectory_preview.png",
    ]:
        if os.path.exists(fname):
            shutil.copy(fname, f"/data/{fname}")
    data_vol.commit()

    return results


@app.local_entrypoint()
def main(
    sequence: str = "",
    split: str = "train",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    train_fraction: float = 0.10,
    eval_fraction: float = 0.02,
    seed: int = 7,
):
    results = run_overfit.remote(
        sequence=sequence,
        split=split,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        train_fraction=train_fraction,
        eval_fraction=eval_fraction,
        seed=seed,
    )
    print(f"\nModal training complete: {results}")
