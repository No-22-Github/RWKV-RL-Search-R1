# RL 全线留存快照（截至 2026-10-29）

总览和结论见 [docs/rl-lineage.md](../../docs/rl-lineage.md)；10-28 夜间调参的细节见 [docs/rl-status-2026-10-29.md](../../docs/rl-status-2026-10-29.md)。这里只放复现所需的训练指标、脚本快照、校验和与命令。

## 目录

| 路径 | 内容 |
|---|---|
| `runs/<run>/args.json` | 训练客户端的完整参数，以及启动时服务端的 health 信息 |
| `runs/<run>/metrics.jsonl` | 每步指标（reward、acc、format_ok、检索次数、生成长度、grad_norm、学习率、耗时） |
| `runs/<run>/expected.jsonl` | 只有路线二 r2 和夜间探针有：每步期望 acc，即当步 16 题的题池标签平均（标签是 s100 在 T=1 下 8 次的正确率）；由 `scripts/export_expected.py` 在服务器上导出 |
| `labels/questions.jsonl` | r2 和探针用的题池标签 |
| `scripts/` | 跑路线二和夜间探针时用的脚本快照，以及画图脚本 |
| `leaderboard/` | SearchAgent 榜单各家的原始评测 json（2026-10-29 从 Space 的 `eval-results/` 拉取） |
| `figures/` | 折线图和榜单对比图，由 `scripts/plot_rl.py` 生成 |
| `SHA256SUMS` | 服务器上大文件（题池、评测集、adapter、训练服务端）和本快照文件的校验和 |

各检查点的评测摘要不在这里，统一放在 `outputs/upstream-eval/<检查点>/report.json`（已进库）。

## run 一览

横轴的累计步数 = 服务端步数 + 偏移。v2 是从 v1-s100 的 adapter 冷启动的，服务端步数重新从 0 计，所以路线一 v2 以后都要加 100。

| run | 路线 | 起点 | 服务端步数 | 累计步数 | 训练机器 |
|---|---|---|---|---|---|
| `sr1-v1` | 一 | SFT v1 | 0–99 | 0–99 | $SFT_HOST（Pro 6000，`~/no22/rl-sr1-v1`） |
| `sr1-v2` | 一 | v1-s100 adapter | 0–251 | 100–351 | Pro6000 `<用户>@$TRAIN_HOST` |
| `sr1-v3b` | 一 | v2 step-200 完整检查点 | 200–310 | 300–410 | 同上 |
| `sr1-v4` | 一 | v3b step-300 完整检查点 | 300–415 | 400–515 | 同上 |
| `sft2-grpo` | 二 | SFT v2（`sft2-r16-lr3e-5-e2`） | 0–101（100 以后被 r2 覆盖） | 0–99 | 同上 |
| `sft2-grpo-r2` | 二 | 第一段 step-100 完整检查点 | 100–199 | 100–199 | 同上 |
| `probe-lr1e-5-collapsed` | 探针 A | 路线二 s100 | — | — | 同上 |
| `probe-lr5e-6-wu8` | 探针 B | 同上 | — | — | 同上 |
| `probe-lr5e-6-seqnorm-neg05` | 探针 C | 同上 | — | — | 同上 |

- Pro6000 上的 run 都在 `/mnt/nvme0n1/no22/rl-sr1-v2/outputs/rl/<run>/`。
- r2 第一次启动在第 20 步被别人的进程挤掉显存（`outputs/rl/sft2-grpo-r2-crashed20`，没用），从 step-100 重新跑的才是这里的 `sft2-grpo-r2`。
- 冒烟和废弃的 run（`smoke`、`sr1-v3`、`sr1-v4smoke`）没收。

## 复现折线图（本机即可，不需要服务器）

```bash
.venv/bin/python scripts/plot_rl.py
```

默认读 `provenance/rl-lineage-20261029/runs/` 和 `outputs/upstream-eval/`，输出到本目录的 `figures/`。

| 图 | 内容 |
|---|---|
| `fig1_eval_lineage.png` | 两条路线的榜单协议七集平均，横轴是累计 RL 步数 |
| `fig2_train_lineage.png` | 各段 RL 的训练 acc、生成长度、检索次数、格式正确率（10 步滑动平均），左右两列各一条路线 |
| `fig3_probe_dynamics.png` | 夜间三组探针和 r2 前 30 步对照，含 acc 减期望 acc；× 是中途停训的位置 |
| `fig4_leaderboard.png` | 和 SearchAgent 榜单各家的七集平均对比 |

## 复现训练和评测

服务器包根目录是 `/mnt/nvme0n1/no22/rl-sr1-v2`。

- 只用 `.venv/bin/python`，不要 `uv run` / `uv sync`（会把 torch 换成 CPU 版）。
- 各阶段的启动命令：
  - 路线一 v1–v4：见 [docs/rl-status-2026-10-26.md](../../docs/rl-status-2026-10-26.md)、[docs/rl-status-2026-10-27-pm.md](../../docs/rl-status-2026-10-27-pm.md) 和 `scripts/chain_v4.sh`。
  - 路线二：见 [provenance/sft-v2-20261027/README.md](../sft-v2-20261027/README.md) 第 6、7 节（`scripts/chain_sft2.sh`、`scripts/chain_sft2_r2.sh`）。
- 夜间探针用的训练服务端：`$TUNE_BIN`（commit d93775e，学习率逐请求传入）。

```bash
# 夜间探针（训练 30 步 → 停服务端 → 按榜单协议评测终点）
NAME=lr5e-6-seqnorm-neg05 LR=5e-6 CLIENT_EXTRA="--warmup_steps 8 --abort_fmt 0.9 --seq_norm --neg_scale 0.5" bash scripts/probe.sh 2>&1 | tee -a outputs/probe-lr5e-6-seqnorm-neg05.log
# 导出每步期望 acc（画图用）
.venv/bin/python scripts/export_expected.py outputs/rl/<run> ...
# 任意 adapter 按榜单协议评测（8848 训练服务端必须先停）
bash scripts/eval_adapter.sh <adapter_id> <adapter.pth>
# 错题归因
.venv/bin/python scripts/error_breakdown.py outputs/upstream-eval/sft2r2-s150-greedy-t3/samples.jsonl
```

rollout 用 T=1.0 采样，重跑的逐步数字不会完全一致，但走势应当一致。
