from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


@dataclass
class PyramidKVConfig:
    """Configuration for pyramid-shaped per-layer KV memory compression."""

    enabled: bool = True
    first_full_layers: int = 4
    last_full_layers: int = 4
    max_block_size: int = 4
    anchor_tokens: int = 64
    recent_tokens: int = 512
    compressor_hidden_dim: int = 64
    layer_block_sizes: Optional[str] = None


class KVBlockCompressor(nn.Module):
    """Learned weighted pooling over local KV blocks.

    The module preserves head_dim and kv_heads. It only reduces the sequence
    dimension by replacing each complete block with one learned summary KV.
    """

    def __init__(self, head_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(head_dim * 2, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        key_raw: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # key_raw/value: [batch, kv_heads, blocks, block_size, head_dim]
        scores = self.score(torch.cat([key_raw, value], dim=-1)).float()
        weights = torch.softmax(scores, dim=-2).to(key_raw.dtype)
        key_summary = (weights * key_raw).sum(dim=-2)
        value_summary = (weights * value).sum(dim=-2)
        return key_summary, value_summary


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_one(
    states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (states * cos) + (rotate_half(states) * sin)


def _parse_explicit_block_sizes(raw: Optional[str], num_layers: int) -> Optional[list[int]]:
    if raw is None or raw.strip() == "":
        return None
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if len(values) != num_layers:
        raise ValueError(
            f"layer_block_sizes has {len(values)} entries, expected {num_layers}"
        )
    if any(x < 1 for x in values):
        raise ValueError("layer_block_sizes entries must be >= 1")
    return values


def build_pyramid_block_sizes(num_layers: int, cfg: PyramidKVConfig) -> list[int]:
    explicit = _parse_explicit_block_sizes(cfg.layer_block_sizes, num_layers)
    if explicit is not None:
        return explicit

    if num_layers <= 1:
        return [1]

    center = (num_layers - 1) / 2.0
    sizes: list[int] = []
    for idx in range(num_layers):
        if idx < cfg.first_full_layers or idx >= num_layers - cfg.last_full_layers:
            sizes.append(1)
            continue
        depth = 1.0 - abs(idx - center) / max(center, 1.0)
        block_size = 1 + int(round(depth * (cfg.max_block_size - 1)))
        sizes.append(max(1, block_size))
    return sizes


def _compress_key_value_memory(
    module: nn.Module,
    key_raw: torch.Tensor,
    value: torch.Tensor,
    block_size: int,
    anchor_tokens: int,
    recent_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, kv_heads, seq_len, head_dim = key_raw.shape
    device = key_raw.device

    if block_size <= 1 or seq_len <= anchor_tokens + recent_tokens + block_size:
        positions = torch.arange(seq_len, device=device, dtype=torch.long)
        return key_raw, value, positions

    anchor_len = min(anchor_tokens, seq_len)
    recent_start = max(anchor_len, seq_len - recent_tokens)
    middle_len = max(recent_start - anchor_len, 0)
    full_block_tokens = middle_len // block_size * block_size

    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []

    if anchor_len > 0:
        key_parts.append(key_raw[:, :, :anchor_len, :])
        value_parts.append(value[:, :, :anchor_len, :])
        pos_parts.append(torch.arange(anchor_len, device=device, dtype=torch.long))

    if full_block_tokens > 0:
        block_start = anchor_len
        block_end = anchor_len + full_block_tokens
        num_blocks = full_block_tokens // block_size
        key_blocks = key_raw[:, :, block_start:block_end, :].reshape(
            bsz, kv_heads, num_blocks, block_size, head_dim
        )
        value_blocks = value[:, :, block_start:block_end, :].reshape(
            bsz, kv_heads, num_blocks, block_size, head_dim
        )
        key_summary, value_summary = module.pyramid_kv_compressor(key_blocks, value_blocks)
        key_parts.append(key_summary)
        value_parts.append(value_summary)
        pos_parts.append(
            torch.arange(
                block_start + block_size - 1,
                block_end,
                block_size,
                device=device,
                dtype=torch.long,
            )
        )

    tail_start = anchor_len + full_block_tokens
    if tail_start < recent_start:
        key_parts.append(key_raw[:, :, tail_start:recent_start, :])
        value_parts.append(value[:, :, tail_start:recent_start, :])
        pos_parts.append(torch.arange(tail_start, recent_start, device=device, dtype=torch.long))

    if recent_start < seq_len:
        key_parts.append(key_raw[:, :, recent_start:, :])
        value_parts.append(value[:, :, recent_start:, :])
        pos_parts.append(torch.arange(recent_start, seq_len, device=device, dtype=torch.long))

    key_memory = torch.cat(key_parts, dim=-2)
    value_memory = torch.cat(value_parts, dim=-2)
    memory_positions = torch.cat(pos_parts, dim=0)
    return key_memory, value_memory, memory_positions


def _gather_attention_mask(
    attention_mask: Optional[torch.Tensor],
    memory_positions: torch.Tensor,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    return attention_mask.index_select(-1, memory_positions.to(attention_mask.device))


def _get_attention_interface(modeling_qwen3, module: nn.Module) -> Callable:
    attention_functions = modeling_qwen3.ALL_ATTENTION_FUNCTIONS
    implementation = getattr(module.config, "_attn_implementation", "eager")
    fallback = modeling_qwen3.eager_attention_forward
    if hasattr(attention_functions, "get_interface"):
        return attention_functions.get_interface(implementation, fallback)
    return attention_functions.get(implementation, fallback)


def pyramid_qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        return self._pyramid_kv_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    modeling_qwen3 = self._pyramid_kv_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_raw = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states = apply_rope_one(query_states, cos, sin)

    key_memory_raw, value_memory, memory_positions = _compress_key_value_memory(
        self,
        key_raw,
        value_states,
        block_size=self.pyramid_kv_block_size,
        anchor_tokens=self.pyramid_kv_config.anchor_tokens,
        recent_tokens=self.pyramid_kv_config.recent_tokens,
    )
    memory_cos = cos.index_select(1, memory_positions.to(cos.device))
    memory_sin = sin.index_select(1, memory_positions.to(sin.device))
    key_memory = apply_rope_one(key_memory_raw, memory_cos, memory_sin)
    compressed_mask = _gather_attention_mask(attention_mask, memory_positions)

    attention_interface = _get_attention_interface(modeling_qwen3, self)
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_memory,
        value_memory,
        compressed_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, "sliding_window", None),
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    self._pyramid_kv_last_memory_len = int(key_memory.shape[-2])
    self._pyramid_kv_last_seq_len = int(hidden_states.shape[-2])
    return attn_output, attn_weights


def patch_qwen3_pyramid_kv(model: nn.Module, cfg: PyramidKVConfig) -> list[int]:
    """Patch Qwen3Attention modules with learned pyramid KV compression."""

    if not cfg.enabled:
        return []
    if cfg.max_block_size < 1:
        raise ValueError("max_block_size must be >= 1")
    if cfg.first_full_layers < 0 or cfg.last_full_layers < 0:
        raise ValueError("first_full_layers and last_full_layers must be >= 0")
    if cfg.anchor_tokens < 0 or cfg.recent_tokens < 0:
        raise ValueError("anchor_tokens and recent_tokens must be >= 0")

    try:
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError as exc:
        raise ImportError(
            "Qwen3 pyramid KV requires a recent transformers version with "
            "transformers.models.qwen3. Install the same transformers version "
            "used by the local Qwen3-0.6B checkpoint."
        ) from exc

    attention_modules = [
        module for module in model.modules() if module.__class__.__name__ == "Qwen3Attention"
    ]
    if not attention_modules:
        raise ValueError("No Qwen3Attention modules found. Check model class/version.")

    block_sizes = build_pyramid_block_sizes(len(attention_modules), cfg)
    for module, block_size in zip(attention_modules, block_sizes):
        module.pyramid_kv_config = cfg
        module.pyramid_kv_block_size = block_size
        if block_size <= 1:
            continue
        if not hasattr(module, "pyramid_kv_compressor"):
            module.pyramid_kv_compressor = KVBlockCompressor(
                head_dim=module.head_dim,
                hidden_dim=cfg.compressor_hidden_dim,
            ).to(device=next(module.parameters()).device, dtype=next(module.parameters()).dtype)
        if not hasattr(module, "_pyramid_kv_original_forward"):
            module._pyramid_kv_original_forward = module.forward
            module._pyramid_kv_modeling_qwen3 = modeling_qwen3
            module.forward = types.MethodType(pyramid_qwen3_attention_forward, module)

    if hasattr(model, "config"):
        model.config.use_cache = False
        model.config.pyramid_kv_block_sizes = block_sizes
        model.config.pyramid_kv_anchor_tokens = cfg.anchor_tokens
        model.config.pyramid_kv_recent_tokens = cfg.recent_tokens
        model.config.pyramid_kv_max_block_size = cfg.max_block_size
    return block_sizes


def set_trainable_scope(model: nn.Module, scope: str) -> tuple[int, int]:
    """Set trainable parameters and return (trainable, total)."""

    if scope not in {"all", "compressor", "compressor_and_attention"}:
        raise ValueError(f"Unsupported trainable scope: {scope}")

    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "pyramid_kv_compressor" in name:
                param.requires_grad = True
            elif scope == "compressor_and_attention" and any(
                marker in name
                for marker in (
                    ".self_attn.q_proj.",
                    ".self_attn.k_proj.",
                    ".self_attn.v_proj.",
                    ".self_attn.o_proj.",
                    ".self_attn.q_norm.",
                    ".self_attn.k_norm.",
                )
            ):
                param.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def format_layer_block_sizes(block_sizes: list[int]) -> str:
    return ",".join(str(x) for x in block_sizes)


def estimate_memory_lengths(seq_len: int, block_sizes: list[int], cfg: PyramidKVConfig) -> list[int]:
    lengths: list[int] = []
    for block_size in block_sizes:
        if block_size <= 1 or seq_len <= cfg.anchor_tokens + cfg.recent_tokens + block_size:
            lengths.append(seq_len)
            continue
        anchor_len = min(cfg.anchor_tokens, seq_len)
        recent_start = max(anchor_len, seq_len - cfg.recent_tokens)
        middle_len = max(recent_start - anchor_len, 0)
        full_block_tokens = middle_len // block_size * block_size
        compressed = full_block_tokens // block_size + (middle_len - full_block_tokens)
        lengths.append(anchor_len + compressed + (seq_len - recent_start))
    return lengths
