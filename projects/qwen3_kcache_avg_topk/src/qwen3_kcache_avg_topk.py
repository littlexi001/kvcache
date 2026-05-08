from __future__ import annotations

import math
import types
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KCacheAvgTopKConfig:
    """Decode-time KV cache block selection from averaged K-cache blocks."""

    enabled: bool = True
    block_size: int = 10
    topk_ratio: float = 0.10
    first_sparse_layer: int = 3
    last_sparse_layer: int = 27
    min_blocks_to_keep: int = 1
    prefill_uses_full_attention: bool = True


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)


def _get_attention_interface(modeling_qwen3, module: nn.Module) -> Callable:
    attention_functions = modeling_qwen3.ALL_ATTENTION_FUNCTIONS
    implementation = getattr(module.config, "_attn_implementation", "eager")
    fallback = modeling_qwen3.eager_attention_forward
    if hasattr(attention_functions, "get_interface"):
        return attention_functions.get_interface(implementation, fallback)
    return attention_functions.get(implementation, fallback)


def _build_block_token_mask(
    selected_blocks: torch.Tensor,
    kv_len: int,
    block_size: int,
) -> torch.Tensor:
    # selected_blocks: [batch, heads, q_len, keep_blocks]
    token_blocks = torch.arange(kv_len, device=selected_blocks.device) // block_size
    return (token_blocks.view(1, 1, 1, 1, kv_len) == selected_blocks.unsqueeze(-1)).any(dim=-2)


def _avg_block_topk_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float,
    scaling: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg: KCacheAvgTopKConfig = module.kcache_avg_topk_config
    bsz, _, q_len, head_dim = query.shape
    kv_len = key.shape[-2]

    full_key = repeat_kv(key, module.num_key_value_groups)
    full_value = repeat_kv(value, module.num_key_value_groups)
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(head_dim)

    if kv_len == 0:
        raise ValueError("KV cache is empty; cannot run averaged K-cache selection.")

    pad_len = (-kv_len) % cfg.block_size
    if pad_len:
        key_for_avg = F.pad(key, (0, 0, 0, pad_len))
        block_valid = torch.ones(kv_len, dtype=key.dtype, device=key.device)
        block_valid = F.pad(block_valid, (0, pad_len)).view(-1, cfg.block_size)
    else:
        key_for_avg = key
        block_valid = torch.ones(kv_len, dtype=key.dtype, device=key.device).view(
            -1, cfg.block_size
        )

    num_blocks = key_for_avg.shape[-2] // cfg.block_size
    key_blocks = key_for_avg.view(bsz, key.shape[1], num_blocks, cfg.block_size, head_dim)
    counts = block_valid.sum(dim=-1).clamp_min(1).view(1, 1, num_blocks, 1)
    avg_key = key_blocks.sum(dim=-2) / counts
    avg_key = repeat_kv(avg_key, module.num_key_value_groups)

    block_scores = torch.matmul(query, avg_key.transpose(-2, -1)) * scaling
    keep_blocks = max(cfg.min_blocks_to_keep, math.ceil(num_blocks * cfg.topk_ratio))
    keep_blocks = min(keep_blocks, num_blocks)
    selected_blocks = torch.topk(block_scores.float(), k=keep_blocks, dim=-1).indices
    token_mask = _build_block_token_mask(selected_blocks, kv_len, cfg.block_size)

    scores = torch.matmul(query, full_key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., :q_len, :kv_len]
    scores = scores.masked_fill(~token_mask, torch.finfo(scores.dtype).min)

    attn = F.softmax(scores.float(), dim=-1).to(query.dtype)
    attn = F.dropout(attn, p=dropout, training=module.training)
    output = torch.matmul(attn, full_value)
    output = output.transpose(1, 2).contiguous()

    module._kcache_avg_topk_last_kv_len = int(kv_len)
    module._kcache_avg_topk_last_num_blocks = int(num_blocks)
    module._kcache_avg_topk_last_keep_blocks = int(keep_blocks)
    return output, attn


def kcache_avg_topk_qwen3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values=None,
    cache_position: Optional[torch.Tensor] = None,
    **kwargs,
):
    cfg: KCacheAvgTopKConfig = self.kcache_avg_topk_config

    if cfg.prefill_uses_full_attention and hidden_states.shape[-2] != 1:
        return self._kcache_avg_topk_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    modeling_qwen3 = self._kcache_avg_topk_modeling_qwen3
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = modeling_qwen3.apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if key_states.shape[-2] < cfg.block_size:
        attention_interface = _get_attention_interface(modeling_qwen3, self)
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
    else:
        attn_output, attn_weights = _avg_block_topk_attention(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def patch_qwen3_kcache_avg_topk(model: nn.Module, cfg: KCacheAvgTopKConfig) -> list[int]:
    """Patch Qwen3Attention layers 3-27 by default for decode-time KV selection."""

    if not cfg.enabled:
        return []
    if cfg.block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not 0.0 < cfg.topk_ratio <= 1.0:
        raise ValueError("topk_ratio must be in (0, 1]")
    if cfg.first_sparse_layer < 0 or cfg.last_sparse_layer < cfg.first_sparse_layer:
        raise ValueError("Invalid sparse layer range")

    try:
        from transformers.models.qwen3 import modeling_qwen3
    except ImportError as exc:
        raise ImportError(
            "Qwen3 K-cache average top-k requires a transformers version with "
            "transformers.models.qwen3."
        ) from exc

    attention_modules = [
        module for module in model.modules() if module.__class__.__name__ == "Qwen3Attention"
    ]
    if not attention_modules:
        raise ValueError("No Qwen3Attention modules found. Check model class/version.")

    patched_layers: list[int] = []
    for layer_idx, module in enumerate(attention_modules):
        if layer_idx < cfg.first_sparse_layer or layer_idx > cfg.last_sparse_layer:
            continue
        module.kcache_avg_topk_config = cfg
        if not hasattr(module, "_kcache_avg_topk_original_forward"):
            module._kcache_avg_topk_original_forward = module.forward
            module._kcache_avg_topk_modeling_qwen3 = modeling_qwen3
            module.forward = types.MethodType(kcache_avg_topk_qwen3_attention_forward, module)
        patched_layers.append(layer_idx)

    if hasattr(model, "config"):
        model.config.kcache_avg_topk_block_size = cfg.block_size
        model.config.kcache_avg_topk_topk_ratio = cfg.topk_ratio
        model.config.kcache_avg_topk_sparse_layers = patched_layers
        model.config.use_cache = True
    return patched_layers
