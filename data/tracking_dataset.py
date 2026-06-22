"""SoccerNet Tracking v3 dataset for player trajectory prediction.

Data format: MOT-style with per-frame bounding boxes in pixel coordinates.
Each sequence has 750 frames at 25fps (30s clips), 1920x1080 JPG images.
"""

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class TrackletInfo:
    track_id: int
    role: str       # "player team left", "player team right", "ball", "referee", ...
    jersey: str     # jersey number or identifier


def parse_gameinfo(content: str) -> dict:
    """Parse gameinfo.ini into a dict with tracklet metadata."""
    info: dict = {}
    tracklets: dict[int, TrackletInfo] = {}
    for line in content.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if key.startswith("trackletID_"):
            tid = int(key.split("_")[1])
            parts = val.split(";")
            role = parts[0].strip()
            jersey = parts[1].strip() if len(parts) > 1 else ""
            tracklets[tid] = TrackletInfo(tid, role, jersey)
        else:
            info[key] = val
    info["tracklets"] = tracklets
    return info


def parse_seqinfo(content: str) -> dict:
    """Parse seqinfo.ini."""
    info = {}
    for line in content.splitlines():
        line = line.strip()
        if "=" in line:
            key, val = line.split("=", 1)
            info[key.strip()] = val.strip()
    return info


def parse_gt(content: str) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    """Parse gt.txt MOT format: frame, track_id, x, y, w, h, conf, -1, -1, -1.

    Returns: {track_id: {frame: (x, y, w, h)}}
    """
    tracks: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    for line in content.splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        track_id = int(parts[1])
        x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
        tracks.setdefault(track_id, {})[frame] = (x, y, w, h)
    return tracks


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


class SoccerNetTrackingDataset(Dataset):
    """Player trajectory prediction dataset from SoccerNet tracking data.

    Each sample:
        frames:       [T, C, H, W]  -- input frame window (float, 0-1)
        player_bbox:  [T, 4]        -- normalized (cx, cy, w, h) per input frame
        future_xy:    [horizon, 2]  -- normalized future centers

    Coordinates are normalized to [0, 1] by image dimensions.  This is
    pixel-space normalization, not pitch-normalized -- pitch calibration
    comes in a later stage.
    """

    def __init__(
        self,
        data_root: str,
        sequence: str = "SNMOT-116",
        split: str = "test",
        input_frames: int = 5,
        horizon: int = 25,
        stride: int = 50,
        image_size: int = 224,
        max_frames: int | None = 250,
        player_filter: str = "player team left",
    ):
        self.data_root = Path(data_root)
        self.sequence = sequence
        self.split = split
        self.input_frames = input_frames
        self.horizon = horizon
        self.stride = stride
        self.image_size = image_size
        self.max_frames = max_frames

        seq_dir = self.data_root / split / sequence

        # data_root is the tracking task root. It can contain either extracted
        # split directories (test/SNMOT-116/...) or split archives (test.zip).
        if seq_dir.exists():
            self._zip = None
            self._seq_dir = seq_dir
            self.seqinfo = parse_seqinfo((seq_dir / "seqinfo.ini").read_text())
            self.gameinfo = parse_gameinfo((seq_dir / "gameinfo.ini").read_text())
            self.tracks = parse_gt((seq_dir / "gt" / "gt.txt").read_text())
        else:
            zip_path = self.data_root / f"{split}.zip"
            if not zip_path.exists():
                raise FileNotFoundError(f"No data found at {seq_dir} or {zip_path}")
            self._zip = zipfile.ZipFile(zip_path)
            self._seq_dir = None
            prefix = f"{split}/{sequence}"
            self.seqinfo = parse_seqinfo(self._zip.read(f"{prefix}/seqinfo.ini").decode())
            self.gameinfo = parse_gameinfo(self._zip.read(f"{prefix}/gameinfo.ini").decode())
            self.tracks = parse_gt(self._zip.read(f"{prefix}/gt/gt.txt").decode())

        self.img_width = int(self.seqinfo["imWidth"])
        self.img_height = int(self.seqinfo["imHeight"])
        self.seq_length = int(self.seqinfo["seqLength"])
        if max_frames is not None:
            self.seq_length = min(self.seq_length, max_frames)

        # Filter to valid player tracklets
        self.tracklets = {
            tid: info for tid, info in self.gameinfo["tracklets"].items()
            if player_filter in info.role
        }

        # Build sample index: (track_id, start_frame)
        self.samples: list[tuple[int, int]] = []
        window = input_frames + horizon
        for tid in self.tracklets:
            track = self.tracks.get(tid, {})
            if len(track) < window:
                continue
            for start in range(1, self.seq_length - window + 2, stride):
                end = start + window - 1
                if all(f in track for f in range(start, end + 1)):
                    self.samples.append((tid, start))

    def _load_image(self, frame_idx: int) -> Image.Image:
        if self._zip is not None:
            prefix = f"{self.split}/{self.sequence}/img1"
            img_data = self._zip.read(f"{prefix}/{frame_idx:06d}.jpg")
            return Image.open(io.BytesIO(img_data)).convert("RGB")
        else:
            path = self._seq_dir / "img1" / f"{frame_idx:06d}.jpg"
            return Image.open(path).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        track_id, start = self.samples[idx]
        track = self.tracks[track_id]

        # Load input frames
        frames = []
        for f in range(start, start + self.input_frames):
            img = self._load_image(f)
            img = img.resize((self.image_size, self.image_size))
            frames.append(np.array(img, dtype=np.float32) / 255.0)
        frames = torch.from_numpy(np.stack(frames))  # [T, H, W, C]
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        # Player bounding boxes (normalized center + size)
        bboxes = []
        for f in range(start, start + self.input_frames):
            x, y, w, h = track[f]
            cx, cy = bbox_center((x, y, w, h))
            bboxes.append([
                cx / self.img_width, cy / self.img_height,
                w / self.img_width, h / self.img_height,
            ])
        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # [T, 4]

        # Future trajectory (normalized centers)
        future_xy = []
        for f in range(start + self.input_frames, start + self.input_frames + self.horizon):
            x, y, w, h = track[f]
            cx, cy = bbox_center((x, y, w, h))
            future_xy.append([cx / self.img_width, cy / self.img_height])
        future_xy = torch.tensor(future_xy, dtype=torch.float32)  # [horizon, 2]

        return {
            "frames": frames,
            "player_bbox": bboxes,
            "future_xy": future_xy,
        }
