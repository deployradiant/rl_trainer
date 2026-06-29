"""Qwen VLM backbone + football trajectory policy/value heads."""

from math import ceil, pi

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from models.action_chunk import ActionChunkHead
from models.kinematics import KINEMATIC_FEATURE_DIM, constant_velocity_future


def build_relative_action_codebook() -> torch.Tensor:
    """Return [action_count, 2] with heading delta radians and speed multiplier."""
    heading_degrees = [-90, -60, -30, -15, 0, 15, 30, 60, 90]
    speed_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0]
    actions = []
    for heading in heading_degrees:
        for speed in speed_multipliers:
            actions.append((heading * pi / 180.0, speed))
    return torch.tensor(actions, dtype=torch.float32)


class RelativeMotionPolicyHead(nn.Module):
    """MLP policy: visual features + player state -> relative motion action logits."""

    def __init__(self, input_dim: int, action_count: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QwenFootballModel(nn.Module):
    """Qwen2-VL vision encoder + trainable trajectory heads.

    Input frames are processed through Qwen's vision encoder individually.
    The Qwen visual encoder is trainable by default so overfit checks exercise
    Qwen itself instead of only the downstream heads.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        horizon: int = 25,
        input_frames: int = 5,
        visual_frames: int | None = None,
        action_stride: int = 5,
        policy_kind: str = "chunk",
        proj_dim: int = 512,
        step_dim: int = 32,
        speed_floor: float = 0.0015,
    ):
        super().__init__()
        if policy_kind not in {"chunk", "codebook"}:
            raise ValueError("policy_kind must be 'chunk' or 'codebook'")
        self.input_frames = input_frames
        self.visual_frames = visual_frames or input_frames
        if self.visual_frames <= 0 or self.visual_frames > self.input_frames:
            raise ValueError("visual_frames must be between 1 and input_frames")
        self.horizon = horizon
        self.action_stride = action_stride
        self.policy_kind = policy_kind
        self.rollout_steps = ceil(horizon / action_stride)
        self.speed_floor = speed_floor
        self.qwen_has_trainable_parameters = False

        # Load Qwen VLM
        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )

        # Image processor for Qwen vision encoder
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.image_processor = self.processor.image_processor

        self.configure_qwen_training()

        # Vision encoder hidden dim (Qwen2-VL-2B: 1536)
        vision_dim = self.vlm.config.vision_config.hidden_size

        # Project flattened per-frame visual features
        self.visual_proj = nn.Sequential(
            nn.Linear(vision_dim * self.visual_frames, proj_dim),
            nn.GELU(),
        )

        self.register_buffer("action_codebook", build_relative_action_codebook())
        self.step_embedding = nn.Embedding(self.rollout_steps, step_dim)

        # Policy head: visual features + rolling bbox state -> relative action logits.
        head_input = proj_dim + input_frames * 4 + step_dim
        self.head = RelativeMotionPolicyHead(head_input, action_count=len(self.action_codebook))
        self.value_head = RelativeMotionPolicyHead(head_input, action_count=len(self.action_codebook))
        self.chunk_head = ActionChunkHead(
            state_dim=proj_dim + input_frames * KINEMATIC_FEATURE_DIM,
            horizon=horizon,
            step_dim=step_dim,
        )
        self.configure_policy_training()

    def configure_qwen_training(self) -> None:
        """Train Qwen's full visual stack by default."""
        for param in self.vlm.parameters():
            param.requires_grad = False

        matched = self._set_qwen_requires_grad(lambda name: name.startswith("visual."))
        if matched == 0:
            raise ValueError("Could not find trainable Qwen visual parameters")
        self.qwen_has_trainable_parameters = True

    def qwen_trainable_parameter_names(self) -> list[str]:
        return [name for name, param in self.vlm.named_parameters() if param.requires_grad]

    def _set_qwen_requires_grad(self, predicate) -> int:
        matched = 0
        for name, param in self.vlm.named_parameters():
            if predicate(name):
                param.requires_grad = True
                matched += param.numel()
        return matched

    def configure_policy_training(self) -> None:
        """Only train the active task head."""
        if self.policy_kind == "chunk":
            inactive_modules = [self.head, self.value_head, self.step_embedding]
        else:
            inactive_modules = [self.chunk_head]
        for module in inactive_modules:
            for param in module.parameters():
                param.requires_grad = False

    def project_visual_features(self, visual_features: torch.Tensor) -> torch.Tensor:
        B = visual_features.shape[0]
        return self.visual_proj(visual_features.reshape(B, -1))

    def _policy_features(
        self,
        visual_proj: torch.Tensor,
        bbox_state: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        B = visual_proj.shape[0]
        bbox_flat = bbox_state.reshape(B, -1)
        step_ids = torch.full((B,), step, dtype=torch.long, device=visual_proj.device)
        step_features = self.step_embedding(step_ids)
        return torch.cat([visual_proj, bbox_flat, step_features], dim=-1)

    def _policy_logits(
        self,
        visual_proj: torch.Tensor,
        bbox_state: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        return self.head(self._policy_features(visual_proj, bbox_state, step))

    def _action_values(
        self,
        visual_proj: torch.Tensor,
        bbox_state: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        return self.value_head(self._policy_features(visual_proj, bbox_state, step))

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

        with torch.set_grad_enabled(self.qwen_has_trainable_parameters and torch.is_grad_enabled()):
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
        return_outputs: bool = False,
        greedy: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.policy_kind == "chunk":
            return self.predict_chunk_from_visual_features(
                visual_features=visual_features,
                player_bbox=player_bbox,
                return_outputs=return_outputs,
            )
        return self.predict_codebook_from_visual_features(
            visual_features=visual_features,
            player_bbox=player_bbox,
            return_outputs=return_outputs,
            greedy=greedy,
        )

    def predict_chunk_from_visual_features(
        self,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Predict a full future trajectory as a residual over recent velocity."""
        visual_proj = self.project_visual_features(visual_features)
        residual_xy = self.chunk_head(visual_proj, player_bbox)
        base_xy = self._constant_velocity_future(player_bbox)
        future_xy = base_xy + residual_xy
        if return_outputs:
            return {"future_xy": future_xy, "residual_xy": residual_xy, "base_xy": base_xy}
        return future_xy

    def _constant_velocity_future(self, player_bbox: torch.Tensor) -> torch.Tensor:
        return constant_velocity_future(player_bbox, self.horizon)

    def predict_codebook_from_visual_features(
        self,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
        return_outputs: bool = False,
        greedy: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Roll out future centers from cached Qwen visual features and action logits."""
        visual_proj = self.project_visual_features(visual_features)
        B = visual_proj.shape[0]
        bbox_state = player_bbox
        current_xy = bbox_state[:, -1, :2]
        future_points = []
        action_logits = []

        for step in range(self.rollout_steps):
            logits = self._policy_logits(visual_proj, bbox_state, step)
            probs = logits.softmax(dim=-1)
            velocity = self._recent_velocity(bbox_state[:, :, :2])
            candidates = self._candidate_velocities(velocity)
            if greedy:
                action_idx = logits.argmax(dim=-1)
                gather_idx = action_idx.view(B, 1, 1).expand(-1, 1, 2)
                next_velocity = candidates.gather(1, gather_idx).squeeze(1)
            else:
                next_velocity = (probs.unsqueeze(-1) * candidates).sum(dim=1)

            frames_this_step = min(self.action_stride, self.horizon - len(future_points))
            for _ in range(frames_this_step):
                current_xy = current_xy + next_velocity
                future_points.append(current_xy)

            next_bbox = torch.cat([current_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)
            action_logits.append(logits)

        future_xy = torch.stack(future_points, dim=1)
        logits = torch.stack(action_logits, dim=1)
        if return_outputs:
            return {"future_xy": future_xy, "action_logits": logits}
        return future_xy

    def teacher_forced_action_logits(
        self,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
        future_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return action logits at ground-truth rollout states."""
        visual_proj = self.project_visual_features(visual_features)
        bbox_state = player_bbox
        logits_by_step = []

        for step in range(self.rollout_steps):
            logits_by_step.append(self._policy_logits(visual_proj, bbox_state, step))

            end = min((step + 1) * self.action_stride, self.horizon)
            target_xy = future_xy[:, end - 1]
            next_bbox = torch.cat([target_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)

        return torch.stack(logits_by_step, dim=1)

    def teacher_forced_action_values(
        self,
        visual_features: torch.Tensor,
        player_bbox: torch.Tensor,
        future_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Return learned action values at ground-truth rollout states."""
        visual_proj = self.project_visual_features(visual_features)
        bbox_state = player_bbox
        values_by_step = []

        for step in range(self.rollout_steps):
            values_by_step.append(self._action_values(visual_proj, bbox_state, step))

            end = min((step + 1) * self.action_stride, self.horizon)
            target_xy = future_xy[:, end - 1]
            next_bbox = torch.cat([target_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)

        return torch.stack(values_by_step, dim=1)

    def _recent_velocity(self, xy_history: torch.Tensor) -> torch.Tensor:
        return (xy_history[:, -1] - xy_history[:, 0]) / max(1, xy_history.shape[1] - 1)

    def _candidate_velocities(self, velocity: torch.Tensor) -> torch.Tensor:
        angles = self.action_codebook[:, 0].to(velocity.device)
        speed_multipliers = self.action_codebook[:, 1].to(velocity.device)

        raw_speed = velocity.norm(dim=-1, keepdim=True)
        speed = raw_speed.clamp_min(self.speed_floor)
        fallback_direction = torch.zeros_like(velocity)
        fallback_direction[:, 0] = 1.0
        direction = torch.where(raw_speed > self.speed_floor, velocity / raw_speed.clamp_min(1e-8), fallback_direction)
        cos = torch.cos(angles).unsqueeze(0)
        sin = torch.sin(angles).unsqueeze(0)
        dx = direction[:, :1]
        dy = direction[:, 1:]
        rotated = torch.stack([
            dx * cos - dy * sin,
            dx * sin + dy * cos,
        ], dim=-1).squeeze(1)
        return rotated * speed.unsqueeze(1) * speed_multipliers.view(1, -1, 1)

    def action_labels(self, player_bbox: torch.Tensor, future_xy: torch.Tensor) -> torch.Tensor:
        """Quantize ground-truth future movement into relative action labels."""
        bbox_state = player_bbox
        current_xy = bbox_state[:, -1, :2]
        labels = []
        previous_end = 0

        for step in range(self.rollout_steps):
            end = min((step + 1) * self.action_stride, self.horizon)
            target_xy = future_xy[:, end - 1]
            frames_this_step = end - previous_end
            target_velocity = (target_xy - current_xy) / max(1, frames_this_step)
            velocity = self._recent_velocity(bbox_state[:, :, :2])
            candidates = self._candidate_velocities(velocity)
            distances = (candidates - target_velocity.unsqueeze(1)).norm(dim=-1)
            labels.append(distances.argmin(dim=-1))

            next_bbox = torch.cat([target_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)
            current_xy = target_xy
            previous_end = end

        return torch.stack(labels, dim=1)

    def action_value_targets(
        self,
        player_bbox: torch.Tensor,
        future_xy: torch.Tensor,
        terminal_weight: float = 0.5,
        target_scale: float = 10.0,
    ) -> torch.Tensor:
        """Build teacher-forced action-value targets for every codebook action.

        The target is an advantage over the best one-step endpoint action at the
        same teacher-forced state. Best action is 0, worse actions are negative.
        MCTS can then maximize learned value without reading the future labels.
        """
        bbox_state = player_bbox
        current_xy = bbox_state[:, -1, :2]
        targets = []
        previous_end = 0

        for step in range(self.rollout_steps):
            end = min((step + 1) * self.action_stride, self.horizon)
            target_xy = future_xy[:, end - 1]
            frames_this_step = end - previous_end
            velocity = self._recent_velocity(bbox_state[:, :, :2])
            candidates = self._candidate_velocities(velocity)
            candidate_end = current_xy.unsqueeze(1) + candidates * max(1, frames_this_step)
            endpoint_error = (candidate_end - target_xy.unsqueeze(1)).norm(dim=-1)
            best_error = endpoint_error.min(dim=-1, keepdim=True).values
            step_weight = 1.0 + (terminal_weight if end == self.horizon else 0.0)
            targets.append((best_error - endpoint_error) * step_weight * target_scale)

            next_bbox = torch.cat([target_xy, bbox_state[:, -1, 2:]], dim=-1).unsqueeze(1)
            bbox_state = torch.cat([bbox_state[:, 1:], next_bbox], dim=1)
            current_xy = target_xy
            previous_end = end

        return torch.stack(targets, dim=1)

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

    def forward_outputs(
        self,
        frames: torch.Tensor,
        player_bbox: torch.Tensor,
        greedy: bool = False,
    ) -> dict[str, torch.Tensor]:
        visual_features = self.extract_visual_features(frames)
        return self.predict_from_visual_features(
            visual_features,
            player_bbox,
            return_outputs=True,
            greedy=greedy,
        )
