"""Sparse chunk attention patch for Qwen3-style Hugging Face models.

This module deliberately patches only Qwen3 eager attention modules. The oracle
mode computes full attention scores first, so it is an upper-bound experiment,
not an inference acceleration path. Router mode replaces oracle chunk selection
with a small trainable chunk scorer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChunkSparseConfig:
    mode: str = "baseline"
    num_chunks: int = 20
    keep_middle: int = 3
    router_dim: int = 128
    router_temperature: float = 1.0
    router_aux_weight: float = 0.05


class ChunkRouter(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, router_dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(head_dim, router_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, router_dim, bias=False)
        self.score = nn.Linear(router_dim, 1, bias=False)
        self.num_heads = num_heads

    def forward(self, query: torch.Tensor, chunk_repr: torch.Tensor) -> torch.Tensor:
        # query: [bsz, heads, q_len, head_dim]
        # chunk_repr: [bsz, heads, q_len, chunks, head_dim]
        q = self.q_proj(query).unsqueeze(-2)
        k = self.k_proj(chunk_repr)
        return self.score(torch.tanh(q + k)).squeeze(-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, kv_heads * n_rep, seq_len, head_dim)


def _causal_valid_mask(
    q_len: int,
    kv_len: int,
    device: torch.device,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    q_positions = torch.arange(kv_len - q_len, kv_len, device=device)
    k_positions = torch.arange(kv_len, device=device)
    valid = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
    valid = valid.unsqueeze(0).unsqueeze(0)
    if attention_mask is not None:
        valid = valid & torch.isfinite(attention_mask[..., :q_len, :kv_len])
    return valid


def _chunk_ids_for_queries(
    valid: torch.Tensor,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Returns chunk ids [bsz, heads, q_len, kv_len] and valid chunk flags
    # [bsz, heads, q_len, num_chunks]. Chunks are relative to each query row's
    # unmasked causal prefix, avoiding future leakage during training.
    bsz, heads, q_len, kv_len = valid.shape
    lengths = valid.sum(dim=-1).clamp_min(1)
    rel_pos = valid.cumsum(dim=-1) - 1
    chunk_ids = (rel_pos * num_chunks / lengths.unsqueeze(-1)).floor().long()
    chunk_ids = chunk_ids.clamp_(0, num_chunks - 1)
    chunk_ids = torch.where(valid, chunk_ids, torch.full_like(chunk_ids, -1))
    chunk_valid = torch.zeros(
        bsz, heads, q_len, num_chunks, dtype=torch.bool, device=valid.device
    )
    chunk_valid.scatter_(-1, chunk_ids.clamp_min(0), valid)
    return chunk_ids, chunk_valid


def _oracle_keep_mask(
    attn_scores: torch.Tensor,
    valid: torch.Tensor,
    num_chunks: int,
    keep_middle: int,
) -> torch.Tensor:
    chunk_ids, chunk_valid = _chunk_ids_for_queries(valid, num_chunks)
    scores = attn_scores.float().masked_fill(~valid, 0.0)
    chunk_mass = torch.zeros_like(chunk_valid, dtype=torch.float32)
    chunk_mass.scatter_add_(-1, chunk_ids.clamp_min(0), scores)
    chunk_mass = chunk_mass.masked_fill(~chunk_valid, float("-inf"))

    keep = torch.zeros_like(chunk_valid)
    keep[..., 0] = chunk_valid[..., 0]
    keep[..., -1] = chunk_valid[..., -1]

    middle = chunk_mass[..., 1:-1]
    k = min(keep_middle, max(num_chunks - 2, 0))
    if k > 0:
        top = torch.topk(middle, k=k, dim=-1).indices + 1
        keep.scatter_(-1, top, True)
    return keep.gather(-1, chunk_ids.clamp_min(0)).logical_and(valid)


def _oracle_chunk_labels(
    attn_scores: torch.Tensor,
    valid: torch.Tensor,
    num_chunks: int,
    keep_middle: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_ids, chunk_valid = _chunk_ids_for_queries(valid, num_chunks)
    scores = attn_scores.float().masked_fill(~valid, 0.0)
    chunk_mass = torch.zeros_like(chunk_valid, dtype=torch.float32)
    chunk_mass.scatter_add_(-1, chunk_ids.clamp_min(0), scores)
    chunk_mass = chunk_mass.masked_fill(~chunk_valid, float("-inf"))
    labels = torch.zeros_like(chunk_valid, dtype=torch.float32)
    k = min(keep_middle, max(num_chunks - 2, 0))
    if k > 0:
        top = torch.topk(chunk_mass[..., 1:-1], k=k, dim=-1).indices + 1
        labels.scatter_(-1, top, 1.0)
    return labels, chunk_valid


def _router_chunk_repr(
    key: torch.Tensor,
    valid: torch.Tensor,
    num_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_ids, chunk_valid = _chunk_ids_for_queries(valid, num_chunks)
    bsz, heads, q_len, kv_len = chunk_ids.shape
    head_dim = key.shape[-1]
    expanded_key = key.unsqueeze(2).expand(bsz, heads, q_len, kv_len, head_dim)
    chunk_sum = torch.zeros(
        bsz, heads, q_len, num_chunks, head_dim, dtype=key.dtype, device=key.device
    )
    chunk_sum.scatter_add_(
        -2,
        chunk_ids.clamp_min(0).unsqueeze(-1).expand_as(expanded_key),
        expanded_key * valid.unsqueeze(-1),
    )
    token_count = torch.zeros(
        bsz, heads, q_len, num_chunks, 1, dtype=key.dtype, device=key.device
    )
    token_count.scatter_add_(
        -2,
        chunk_ids.clamp_min(0).unsqueeze(-1),
        valid.unsqueeze(-1).to(key.dtype),
    )
    return chunk_sum / token_count.clamp_min(1), chunk_valid


def _router_keep_mask(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    sparse_cfg: ChunkSparseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_repr, chunk_valid = _router_chunk_repr(key, valid, sparse_cfg.num_chunks)
    scores = module.chunk_router(query, chunk_repr).float()
    select_scores = scores / sparse_cfg.router_temperature
    select_scores = select_scores.masked_fill(~chunk_valid, float("-inf"))

    keep = torch.zeros_like(chunk_valid)
    keep[..., 0] = chunk_valid[..., 0]
    keep[..., -1] = chunk_valid[..., -1]
    k = min(sparse_cfg.keep_middle, max(sparse_cfg.num_chunks - 2, 0))
    if k > 0:
        top = torch.topk(select_scores[..., 1:-1], k=k, dim=-1).indices + 1
        keep.scatter_(-1, top, True)

    chunk_ids, _ = _chunk_ids_for_queries(valid, sparse_cfg.num_chunks)
    keep_mask = keep.gather(-1, chunk_ids.clamp_min(0)).logical_and(valid)
    return keep_mask, scores, chunk_valid


def sparse_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scaling = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., : query.shape[-2], : key.shape[-2]]

    sparse_cfg: ChunkSparseConfig = module.chunk_sparse_config
    valid = _causal_valid_mask(query.shape[-2], key.shape[-2], query.device, attention_mask)
    valid = valid.expand_as(scores)

    if sparse_cfg.mode == "oracle":
        keep_mask = _oracle_keep_mask(scores.detach(), valid, sparse_cfg.num_chunks, sparse_cfg.keep_middle)
    elif sparse_cfg.mode == "router":
        keep_mask, router_scores, chunk_valid = _router_keep_mask(module, query, key, valid, sparse_cfg)
        labels, label_valid = _oracle_chunk_labels(
            scores.detach(), valid, sparse_cfg.num_chunks, sparse_cfg.keep_middle
        )
        middle_valid = chunk_valid[..., 1:-1] & label_valid[..., 1:-1]
        if middle_valid.any():
            aux = F.binary_cross_entropy_with_logits(
                router_scores[..., 1:-1][middle_valid],
                labels[..., 1:-1][middle_valid],
            )
        else:
            aux = router_scores.sum() * 0.0
        module._chunk_router_loss = aux
    else:
        keep_mask = valid

    scores = scores.masked_fill(~keep_mask, torch.finfo(scores.dtype).min)
    attn = F.softmax(scores.float(), dim=-1).to(query.dtype)
    attn = F.dropout(attn, p=dropout, training=module.training)
    output = torch.matmul(attn, value)
    output = output.transpose(1, 2).contiguous()
    return output, attn


def patch_qwen3_chunk_attention(model: nn.Module, cfg: ChunkSparseConfig) -> None:
    if cfg.mode == "baseline":
        return

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as exc:
        raise ImportError(
            "Qwen3 chunk attention requires a recent transformers version with "
            "Qwen3 support and ALL_ATTENTION_FUNCTIONS. Install/activate the "
            "same transformers version used by the local Qwen3-0.6B model."
        ) from exc

    ALL_ATTENTION_FUNCTIONS["chunk_sparse"] = sparse_attention_forward
    model.config._attn_implementation = "chunk_sparse"

    patched = 0
    for module in model.modules():
        class_name = module.__class__.__name__
        if class_name != "Qwen3Attention":
            continue
        module.chunk_sparse_config = cfg
        if cfg.mode == "router" and not hasattr(module, "chunk_router"):
            num_heads = getattr(module.config, "num_attention_heads", None)
            head_dim = getattr(module, "head_dim", None)
            if num_heads is None or head_dim is None:
                raise ValueError("Could not infer Qwen3Attention num_heads/head_dim")
            module.chunk_router = ChunkRouter(num_heads, head_dim, cfg.router_dim).to(
                next(module.parameters()).device
            )
        patched += 1

    if patched == 0:
        raise ValueError("No Qwen3Attention modules found. Check model class and transformers version.")

    if cfg.mode == "router" and not hasattr(model, "_chunk_sparse_original_forward"):
        model._chunk_sparse_original_forward = model.forward

        def forward_with_router_loss(*args, **kwargs):
            outputs = model._chunk_sparse_original_forward(*args, **kwargs)
            aux_losses = [
                module._chunk_router_loss
                for module in model.modules()
                if hasattr(module, "_chunk_router_loss")
            ]
            if aux_losses and getattr(outputs, "loss", None) is not None:
                loss = outputs.loss + cfg.router_aux_weight * torch.stack(aux_losses).mean()
                outputs.loss = loss
                if isinstance(outputs, dict):
                    outputs["loss"] = loss
            return outputs

        model.forward = forward_with_router_loss
