# 留存快照：Dr. GRPO 与 RAFT（2026-10-31 — 11-01）

报告：[docs/rl-status-2026-11-01.md](../../docs/rl-status-2026-11-01.md)。本目录只放快照（脚本副本、评测报告、训练指标、打包清单、校验和）；权重、轨迹和训练包留在服务器包目录，用 `SHA256SUMS.server` 核对（路径相对包根目录）。`SHA256SUMS` 是本目录文件的校验和。

## 目录

| 路径 | 内容 |
|---|---|
| `scripts/` | 本阶段新增或改动的脚本：`rl_grpo.py`（加 `--algorithm dr_grpo`）、`probe.sh`（服务端参数可配）、`chain_drgrpo.sh`、`chain_raft.sh`、`chain_raft2.sh`、`pack_raft.py`、`make_resume_ckpt.py`，以及评测用的 `eval_adapter.sh` |
| `eval/<检查点>/report.json` | 榜单协议评测摘要（七集抽样 1925 题，贪心），地址已码成 `$RWKV_API` |
| `rl/<探针>/` | Dr. GRPO 冒烟、三档扫描和长跑的 `metrics.jsonl` 与 `args.json`；`drgrpo-select.json` 是自动选档结果 |
| `raft/r1/` | 第一轮：s100 采样报告、A/B 两臂的打包清单、B 的续训起点 `checkpoint.json`（`SOURCE` 记录来自哪个 RL 检查点）、训练日志 |
| `raft/r2/` | 第二轮：raft1-a 采样报告、打包清单、训练日志 |

## 关键产物（服务器包目录，校验和见 `SHA256SUMS.server`）

| 产物 | 路径 | sha256 前 16 位 |
|---|---|---|
| raft1-a adapter（全线最好，七集 0.407） | `outputs/raft/sft-a/adapter-final.pth` | 48a3054fa23609a2 |
| raft1-b adapter（0.384） | `outputs/raft/sft-b/adapter-final.pth` | e80ab95b04088b43 |
| raft2 adapter（0.405） | `outputs/raft2/sft/adapter-final.pth` | 414e0bbe751ff9e9 |
| 第一轮训练包 A / B | `outputs/raft/pack-{a,b}/train.jsonl` | 375ec1ad937135f2 / 8b3203f1f08e4ae8 |
| 第二轮训练包 | `outputs/raft2/pack/train.jsonl` | ffb2aaa5e75789fd |
| 教师轨迹（SFT v2 训练包） | `/mnt/nvme0n1/no22/segments-sft2.jsonl` | f4085424236e5b3f |
| 起点 s100 | `rl-out/server/d53cd0acfc854114943d3b2a4e027a68_1-step-100/adapter.pth` | 52cc094b943df3f2 |
| 训练、推理服务端（v1.8.8 发布包） | `rwkv_lighting_cuda/` | 发布包 sha256 93d37881ece53cd7 |

## 参数

- 采样：`eval_base.py --format g1k --think full --max_turns 4 --final_answer_turn --max_obs_tokens 500 --max_turn_tokens 2048 --topk 3 --n 8 --temperature 1.0`。
- SFT（A 和第二轮）：`rwkv_miss_tune --targets all --ctx 6144 --chunk 6144 --batch-size 8 --warmup-steps 10 --rank 16 --alpha 16 --lr 3e-5 --lr-final 3e-6 --epochs 2`，seed 1234。
- SFT（B）：同上，但 `--lr 1e-5 --lr-final 1e-6 --epochs 1 --resume outputs/raft/b-init`。
- 题池：
  - 第一轮：`outputs/filter/drgrpo-merged`，即 sft2-s100-e5t3 的 1189 题加上 sft2-e5t3 的 734 题；
  - 第二轮：再加从 `train_rl_sft2_r2.parquet` 新抽的 NQ、HotpotQA 各 1250 题，random_state 1101。
