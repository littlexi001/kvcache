# Qwen3 Pyramid KV Compression

This project modifies Qwen3 attention for continued pretraining with a
pyramid-shaped KV memory:

- Early layers keep full KV.
- Middle layers replace older KV blocks with learned summaries.
- Final layers keep full KV.
- Hidden states keep the original sequence length at every layer.

The compression path is:

```text
hidden_states -> q/k/v projections
raw K + V -> learned block compressor
compressed raw K -> RoPE at block-end positions
Q full sequence attends to compressed K/V memory
```

The compressor only changes the KV sequence dimension:

```text
[batch, kv_heads, seq, head_dim] -> [batch, kv_heads, shorter_seq, head_dim]
```

This first implementation targets continued pretraining/evaluation with
`use_cache=false`. If a compressed layer receives `past_key_values`, it falls
back to the original full-attention forward path.

## Recommended training order

1. Sanity check with random tokens:

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh sanity
```

2. Train only the new compressors from the existing Qwen3-0.6B weights:

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
```

For large DCLM-style directories, streaming is enabled by default. This avoids
building a huge Arrow cache before training. Streaming datasets do not have a
fixed length, so real-data stages default to `MAX_STEPS=10000`.

3. Unfreeze attention plus compressors:

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh attention
```

4. Optional low-LR full-model continued pretraining:

```bash
bash projects/pyramid_kv_compression/scripts/run_8gpu.sh full
```

## Default paths

```bash
MODEL_PATH=/mnt/workspace/lym_code/models/Qwen3-0.6B
DATA_PATH=/mnt/workspace/dclm
```

Override them when launching:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B DATA_PATH=/path/to/dclm \
  bash projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
```

## Useful knobs

```bash
MAX_BLOCK_SIZE=2   # closest to 2 tokens -> 1 summary in middle layers
MAX_BLOCK_SIZE=4   # stronger middle compression, default
ANCHOR_TOKENS=64
RECENT_TOKENS=512
FIRST_FULL_LAYERS=4
LAST_FULL_LAYERS=4
SEQ_LENGTH=4096
STREAMING=true
DATASET_FORMAT=auto
DATA_FILES_GLOB="**/*.parquet"
```

For exact manual layer control, pass a comma-separated block-size list:

```bash
LAYER_BLOCK_SIZES=1,1,1,1,2,2,3,3,4,4,4,4,3,3,2,2,1,1,1,1 \
  bash projects/pyramid_kv_compression/scripts/run_8gpu.sh compressor
```

The list length must match the number of Qwen3 attention layers.
