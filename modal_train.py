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


def _write_smoke_dataset(data_root: str, sequence: str) -> None:
    """Create a tiny SoccerNet-like sequence for Modal smoke tests."""
    from pathlib import Path

    from PIL import Image, ImageDraw

    seq_dir = Path(data_root) / "test" / sequence
    img_dir = seq_dir / "img1"
    gt_dir = seq_dir / "gt"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    width = 224
    height = 224
    num_frames = 35
    box_w = 14
    box_h = 30

    seqinfo = "\n".join([
        "[Sequence]",
        f"name={sequence}",
        "imDir=img1",
        "frameRate=25",
        f"seqLength={num_frames}",
        f"imWidth={width}",
        f"imHeight={height}",
        "imExt=.jpg",
        "",
    ])
    (seq_dir / "seqinfo.ini").write_text(seqinfo)

    gameinfo = "\n".join([
        "[GameInfo]",
        "trackletID_1=player team left;7",
        "",
    ])
    (seq_dir / "gameinfo.ini").write_text(gameinfo)

    gt_rows = []
    for frame_idx in range(1, num_frames + 1):
        x = 30 + frame_idx
        y = 80 + frame_idx // 3
        gt_rows.append(f"{frame_idx},1,{x},{y},{box_w},{box_h},1,-1,-1,-1")

        image = Image.new("RGB", (width, height), color=(24, 120, 72))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, width - 1, height - 1], outline=(235, 235, 235), width=2)
        draw.rectangle([x, y, x + box_w, y + box_h], fill=(220, 40, 40))
        image.save(img_dir / f"{frame_idx:06d}.jpg", quality=90)

    (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n")


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
    smoke: bool = False,
):
    import os
    import sys

    # Ensure local modules are importable
    sys.path.insert(0, "/root")
    from training.train_supervised import train

    if smoke:
        sequence = "SNMOT-SMOKE"
        data_root = "/data/smoke-tracking"
        _write_smoke_dataset(data_root, sequence)
        max_frames = 35
    else:
        data_root = "/data/tracking-2023"
        max_frames = 250

        # Download SoccerNet test split if not cached
        zip_path = f"{data_root}/test.zip"
        if not os.path.exists(zip_path):
            print("Downloading SoccerNet tracking test split...")
            os.makedirs(data_root, exist_ok=True)
            from SoccerNet.Downloader import SoccerNetDownloader
            dl = SoccerNetDownloader(LocalDirectory="/data/SoccerNet")
            dl.downloadDataTask(task="tracking-2023", split=["test"], verbose=True)
            # Move to expected path
            import shutil
            shutil.move("/data/SoccerNet/tracking-2023/test.zip", zip_path)
            data_vol.commit()

    results = train(
        data_root=data_root,
        sequence=sequence,
        split="test",
        max_frames=max_frames,
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
    smoke: bool = False,
):
    results = run_overfit.remote(
        sequence=sequence,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        smoke=smoke,
    )
    print(f"\nModal training complete: {results}")
