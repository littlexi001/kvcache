# Qwen3 KV Cache Experiments

This workspace now contains separate Qwen3-0.6B experiment projects:

- `projects/qwen3_chunk_routing`: the previous chunk mask/router experiment.
- `projects/pyramid_kv_compression`: the new pyramid KV compression pretraining experiment.
- `projects/qwen3_kcache_avg_topk`: an inference-time K-cache average block
  selector that keeps layers 0-2 unchanged and applies top-10% block selection
  to layers 3-27.

Use the project-local README files and scripts inside each folder.
