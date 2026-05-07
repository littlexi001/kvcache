# Qwen3 KV Chunk Routing Experiments

This repo contains a minimal experiment harness for modifying Qwen3-0.6B attention
into three comparable modes:

1. `baseline`: original full attention.
2. `oracle`: compute full attention scores, split valid past tokens into 20 chunks,
   always keep chunk 1 and the recent chunk, then keep the top-3 middle chunks by
   attention mass.
3. `router`: learn a low-cost chunk router that selects top-3 middle chunks from
   chunk summaries before exact attention.

The intended machine paths are:

```bash
MODEL=/mnt/workspace/lym_code/models/Qwen3-0.6B
DATA=/mnt/workspace/dclm
```

By default, the scripts initialize the Qwen3-0.6B architecture from random
weights. The tokenizer and model config are still read from `MODEL`, but the
model parameters are not loaded from the checkpoint.

## Run on 8 GPUs

Baseline:

```bash
bash projects/qwen3_chunk_routing/scripts/run_8gpu.sh baseline
```

Oracle sparse upper bound:

```bash
bash projects/qwen3_chunk_routing/scripts/run_8gpu.sh oracle
```

Router:

```bash
bash projects/qwen3_chunk_routing/scripts/run_8gpu.sh router
```

The scripts use `torchrun --nproc_per_node=8` and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

## Run with nohup

Start a background training job:

```bash
bash projects/qwen3_chunk_routing/scripts/nohup_run_8gpu.sh oracle
```

Watch the log:

```bash
tail -f projects/qwen3_chunk_routing/logs/oracle_*.log
```

Use TensorBoard:

```bash
tensorboard --logdir projects/qwen3_chunk_routing/outputs/qwen3_oracle/tensorboard --host 0.0.0.0 --port 6006
```

## Random Token Data

The default data mode uses `/mnt/workspace/dclm`. To train on synthetic random
token data instead:

```bash
DATA_MODE=random_tokens bash projects/qwen3_chunk_routing/scripts/nohup_run_8gpu.sh router
```

## Recommended Order

Run the experiments in this order:

1. `baseline`: verifies data/model/training.
2. `oracle`: measures the upper bound of the 20-to-5 chunk structure.
3. `router`: tests the deployable selector.

If `oracle` loses too much compared with `baseline`, the chunk structure itself is
too aggressive. If `oracle` is close but `router` is weak, improve router training
or chunk summaries.
