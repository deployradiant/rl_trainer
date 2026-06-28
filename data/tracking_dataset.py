"""SoccerNet Tracking v3 dataset for player trajectory prediction.

Data format: MOT-style with per-frame bounding boxes in pixel coordinates.
Each sequence has 750 frames at 25fps (30s clips), 1920x1080 JPG images.
"""

import io
import bisect
import zipfile
from dataclasses import dataclass
from math import ceil
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


def list_sequences(data_root: str, split: str) -> list[str]:
    """List SoccerNet sequence names from an extracted split or split zip."""
    root = Path(data_root)
    split_dir = root / split
    if split_dir.exists():
        return sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    zip_path = root / f"{split}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No split data found at {split_dir} or {zip_path}")

    sequences = set()
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            parts = name.split("/")
            if len(parts) >= 3 and parts[0] == split and parts[2] == "seqinfo.ini":
                sequences.add(parts[1])
    return sorted(sequences)


def choose_sequence_subsets(
    sequences: list[str],
    train_fraction: float,
    eval_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Pick deterministic, non-overlapping train/eval sequence subsets."""
    if not sequences:
        raise ValueError("Cannot choose subsets from an empty sequence list")
    if train_fraction <= 0 or eval_fraction <= 0:
        raise ValueError("train_fraction and eval_fraction must be positive")
    if train_fraction + eval_fraction > 1:
        raise ValueError("train_fraction + eval_fraction must be <= 1")

    import random

    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, ceil(len(shuffled) * train_fraction))
    n_eval = max(1, ceil(len(shuffled) * eval_fraction))
    if n_train + n_eval > len(shuffled):
        raise ValueError(
            f"Need at least {n_train + n_eval} sequences for train/eval subsets, "
            f"found {len(shuffled)}"
        )
    return sorted(shuffled[:n_train]), sorted(shuffled[n_train:n_train + n_eval])


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

    def motion_score(self, idx: int) -> float:
        """Return future displacement for a sample in normalized image space."""
        track_id, start = self.samples[idx]
        track = self.tracks[track_id]
        current_frame = start + self.input_frames - 1
        final_frame = start + self.input_frames + self.horizon - 1
        current_xy = np.array(bbox_center(track[current_frame]), dtype=np.float32)
        final_xy = np.array(bbox_center(track[final_frame]), dtype=np.float32)
        scale = np.array([self.img_width, self.img_height], dtype=np.float32)
        return float(np.linalg.norm((final_xy - current_xy) / scale))

    def sample_metadata(self, idx: int) -> dict:
        track_id, start = self.samples[idx]
        return {
            "sequence": self.sequence,
            "track_id": track_id,
            "start_frame": start,
            "motion_score": self.motion_score(idx),
        }

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
            "sequence": self.sequence,
            "track_id": track_id,
            "start_frame": start,
        }


class MultiSequenceSoccerNetTrackingDataset(Dataset):
    """Concatenate fixed SoccerNet sequences while preserving sample metadata."""

    def __init__(
        self,
        data_root: str,
        sequences: list[str],
        split: str = "train",
        input_frames: int = 5,
        horizon: int = 25,
        stride: int = 50,
        image_size: int = 224,
        max_frames: int | None = 250,
        player_filter: str = "player team left",
    ):
        self.datasets = [
            SoccerNetTrackingDataset(
                data_root=data_root,
                sequence=sequence,
                split=split,
                input_frames=input_frames,
                horizon=horizon,
                stride=stride,
                image_size=image_size,
                max_frames=max_frames,
                player_filter=player_filter,
            )
            for sequence in sequences
        ]
        self.datasets = [dataset for dataset in self.datasets if len(dataset) > 0]
        self.cumulative_lengths: list[int] = []
        total = 0
        for dataset in self.datasets:
            total += len(dataset)
            self.cumulative_lengths.append(total)

    def __len__(self) -> int:
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        dataset_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        start = 0 if dataset_idx == 0 else self.cumulative_lengths[dataset_idx - 1]
        return dataset_idx, idx - start

    def motion_score(self, idx: int) -> float:
        dataset_idx, local_idx = self._locate(idx)
        return self.datasets[dataset_idx].motion_score(local_idx)

    def sample_metadata(self, idx: int) -> dict:
        dataset_idx, local_idx = self._locate(idx)
        return self.datasets[dataset_idx].sample_metadata(local_idx)

    def __getitem__(self, idx: int) -> dict:
        dataset_idx, local_idx = self._locate(idx)
        return self.datasets[dataset_idx][local_idx]
