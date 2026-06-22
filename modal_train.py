"""Modal entrypoint: overfit trajectory prediction on a 10s SoccerNet clip."""

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
    timeout=1800,
)
def run_overfit(
    sequence: str = "SNMOT-116",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
):
    import os
    import sys

    # Ensure local modules are importable
    sys.path.insert(0, "/root")
    from training.train_supervised import train

    # Download SoccerNet test split if not cached
    zip_path = "/data/tracking-2023/test.zip"
    if not os.path.exists(zip_path):
        print("Downloading SoccerNet tracking test split...")
        os.makedirs("/data/tracking-2023", exist_ok=True)
        from SoccerNet.Downloader import SoccerNetDownloader
        dl = SoccerNetDownloader(LocalDirectory="/data/SoccerNet")
        dl.downloadDataTask(task="tracking-2023", split=["test"], verbose=True)
        # Move to expected path
        import shutil
        shutil.move("/data/SoccerNet/tracking-2023/test.zip", zip_path)
        data_vol.commit()

    results = train(
        data_root="/data/tracking-2023",
        sequence=sequence,
        split="test",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device="cuda",
    )

    # Save artifacts to volume
    import shutil
    for fname in ["training_history.json", "trajectory_head.pt"]:
        if os.path.exists(fname):
            shutil.copy(fname, f"/data/{fname}")
    data_vol.commit()

    return results


@app.local_entrypoint()
def main(
    sequence: str = "SNMOT-116",
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
):
    results = run_overfit.remote(
        sequence=sequence,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    print(f"\nModal training complete: {results}")
