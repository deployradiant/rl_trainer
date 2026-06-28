"""Qwen VLM backbone (frozen) + football trajectory head."""

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class TrajectoryHead(nn.Module):
    """MLP head: visual features + player position -> future deltas [horizon, 2]."""

    def __init__(self, input_dim: int, horizon: int = 25, hidden_dim: int = 256):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.horizon, 2)


class QwenFootballModel(nn.Module):
    """Frozen Qwen2-VL vision encoder + trainable trajectory head.

    Input frames are processed through Qwen's vision encoder individually.
    Visual features are pooled per-frame, flattened, projected, and
    concatenated with player bbox info before the trajectory head.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        horizon: int = 25,
        input_frames: int = 5,
        freeze_backbone: bool = True,
        proj_dim: int = 512,
    ):
        super().__init__()
        self.input_frames = input_frames
        self.horizon = horizon

        # Load Qwen VLM
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        # Image processor for Qwen vision encoder
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.image_processor = self.processor.image_processor

        if freeze_backbone:
            for param in self.vlm.parameters():
                param.requires_grad = False

        # Vision encoder hidden dim (Qwen2-VL-2B: 1536)
        vision_dim = self.vlm.config.vision_config.hidden_size

        # Project flattened per-frame visual features
        self.visual_proj = nn.Sequential(
            nn.Linear(vision_dim * input_frames, proj_dim),
            nn.GELU(),
        )

        # Trajectory head: visual features + bbox info -> future xy deltas.
        head_input = proj_dim + input_frames * 4
        self.head = TrajectoryHead(head_input, horizon=horizon)

    def extract_visual_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract pooled visual features per frame.

        Args:
            frames: [B, T, C, H, W] float tensor, values in [0, 1]
        Returns:
            features: [B, T, D] pooled visual features
        """
        B, T, C, H, W = frames.shape

        # Convert to PIL images for the Qwen image processor
        pil_images = []
        for b in range(B):
            for t in range(T):
                img = frames[b, t].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                pil_images.append(Image.fromarray((img * 255).astype("uint8")))

        # Process with Qwen image processor
        inputs = self.image_processor(images=pil_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.vlm.device, self.vlm.dtype)
        grid_thw = inputs["image_grid_thw"].to(self.vlm.device)

        with torch.no_grad():
            hidden_states = self.vlm.visual(pixel_values, grid_thw=grid_thw)

        # hidden_states is post-Qwen patch-merger, so each image has fewer
        # visual tokens than the raw grid_h * grid_w patch count.
        merge_size = self.vlm.config.vision_config.spatial_merge_size
        tokens_per_img = (
            grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2] // (merge_size * merge_size)
        ).tolist()
        features = []
        idx = 0
        for n in tokens_per_img:
            pooled = hidden_states[idx:idx + n].mean(dim=0)  # [D]
            features.append(pooled)
            idx += n

        features = torch.stack(features)  # [B*T, D]
        features = features.view(B, T, -1)  # [B, T, D]
        return features.float()

    def predict_from_visual_features(
        self,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future centers from cached Qwen visual features."""
        B = visual_features.shape[0]
        visual_flat = visual_features.reshape(B, -1)
        visual_proj = self.visual_proj(visual_flat)

        bbox_flat = player_bbox.reshape(B, -1)
        combined = torch.cat([visual_proj, bbox_flat], dim=-1)

        delta_xy = self.head(combined)
        current_xy = player_bbox[:, -1, :2].unsqueeze(1)
        return current_xy + delta_xy

    def forward(self, frames: torch.Tensor, player_bbox: torch.Tensor) -> torch.Tensor:
        """Predict future trajectory.

        Args:
            frames:      [B, T, C, H, W]
            player_bbox: [B, T, 4] normalized (cx, cy, w, h)
        Returns:
            future_xy:   [B, horizon, 2] normalized predicted centers
        """
        visual_features = self.extract_visual_features(frames)  # [B, T, D]
        return self.predict_from_visual_features(visual_features, player_bbox)
