# SFT v2 留存（2026-10-27 起）

用途：写报告、向 TencentBAC SearchAgent Leaderboard 提交时复现和说明数据来源。本目录只放**快照**（脚本副本、排除表、校验和、命令）；大文件留在原处，用 `SHA256SUMS` 核对。规划和决策过程见 `docs/sft-v2-plan.md`。

项目不是 git 仓库，`scripts/` 会被继续修改，所以每个阶段用到的脚本都复制一份到这里的 `scripts/`。

## 链路

基础模型 rwkv7-g1k-2.9b-20260930-ctx25600（sha256 前缀 d8f8ecd4）→ SFT v2（MiSS）→ GRPO 约 200 步。

## 1. 候选题与排除

- 训练题来源：Search-R1 `train.parquet`（NQ 79168 + HotpotQA 90447，榜单只允许这两个训练集）。
- 排除（按 (id, source)，id 在两数据集间撞号）：
  - 旧 RL 筛题时见过的题：`exclusion/filter-{s100,s250,s300-e5}.jsonl`（服务器 `outputs/filter/*/questions.jsonl` 副本，7126 题）；
  - v1 蒸馏和各次试点用过的题（`outputs/distill/{v1,pilot*,v2_pilot*}/trajectories.jsonl`）。合计 9194。
  - 第一轮 RL（rl-sr1-v1）用的 800 道随机题没有留下记录，未能排除。
- 候选：`data/sft/candidates-v2-4000.parquet`，排除后 NQ / HotpotQA 各随机 2000（random_state=303）。

## 2. 用 s300 筛难度（不花 API）

Pro6000 上，adapter `rl-sr1-v3b-s300`（`checkpoints/rl/rl-sr1-v3b-s300.pth`），榜单协议：

```bash
.venv/bin/python scripts/eval_base.py --data data/search_r1/candidates-v2-4000.parquet --format g1k --think full \
  --max_turns 4 --final_answer_turn --max_obs_tokens 500 --greedy --max_turn_tokens 2048 --topk 3 \
  --api http://127.0.0.1:8100 --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64 \
  --adapter_id rl-sr1-v3b-s300 --out outputs/sft-select/s300-cand4000
```

结果 `outputs/sft-select/s300-cand4000/`：NQ EM 0.396，HotpotQA 0.525。服务器与本机 `eval_base.py` 校验和一致（f2e620aa…）。

## 3. 教师题单（2000 题）

`data/sft/teacher-v2-2000.parquet`，`bucket` 列标来源（random_state=505）：

| bucket | NQ | HotpotQA | 规则 |
|---|---|---|---|
| v1 | 345 | 516 | v1 SFT 训练包（train 821 + val 40）的题，全部重跑 |
| wrong | 312 | 500 | s300 答错（HotpotQA 随机 500；NQ 随机 450 后再删 138 凑 2000） |
| hard_ok | 43 | 184 | s300 答对且检索 ≥3 次，全部 |
| easy | 50 | 50 | s300 1–2 次检索答对，随机 |

中间产物 `data/sft/teacher-batch1.parquet`（1277 题，合并前的版本，random_state=404）。

## 4. 教师蒸馏

- 教师：`Qwen/Qwen3.7-Flash`，Command Code OpenAI 兼容接口 `$API_BASE`，开思考，temperature 0.3、top_p 1、max_tokens 8192。
- 环境与榜单评测逐字节一致：e5（wiki-18，Flat，Pro6000 `:8002`）top-3、每次检索结果按 RWKV 分词器截 500 token、最多 4 次检索 + 第 5 轮带收尾提示只答。
- 教师额外看到一段学生看不到的 Teacher notes（`TEACHER_SUFFIX`：至少检索一次、答案尽量短、思考简短）。

```bash
set -a; source .env; set +a   # API_BASE、API_KEY、RETRIEVER_URL、TRAIN_HOST 等，见 .env.example
export NO_PROXY=$TRAIN_HOST
uv run python scripts/distill_teacher.py --data data/sft/teacher-v2-2000.parquet --n_per_source 0 --teacher_suffix \
  --max_steps 5 --topk 3 --max_obs_tokens 500 --retriever $RETRIEVER_URL --parallel 16 --out outputs/distill/v2
```

输出 `outputs/distill/v2/`（trajectories.jsonl、stats.json），日志 `outputs/distill/v2.log`。2026-10-27 20:10–21:00 跑完，sha256 已入 SHA256SUMS。

结果：教师 EM 0.594（2000 题），7 条接口错误（400 / 524 / 代理断开）未重跑。

| bucket | NQ EM | HotpotQA EM |
|---|---|---|
| v1 | 0.899 | 0.875 |
| easy | 0.76 | 0.88 |
| hard_ok | 0.581 | 0.685 |
| wrong | 0.176 | 0.274 |

## 5. 后处理与打包

```bash
cd scripts && uv run python pack_sft_v2.py --trajectories ../outputs/distill/v2/trajectories.jsonl \
  --out ../data/sft/searchr1-g1k-v2-pack-20261027
```

`scripts/pack_sft_v2.py` 依次：
1. 只留 status=answer 且 EM 对、≥1 次检索、无重复检索、每步思考非空；
2. `strip_leak.fix_step`：整句删掉复述 Teacher notes 的句子（215 步），删不干净的整条丢（1 条）；
3. 保留教师原始思考，每步 ≤800 RWKV token（丢 82 条）；
4. 按 (qid, source) 去重；验证集每个来源 30 条（seed 0）。

**答案对齐没有用**：试过把"包含标准答案"的长答案收缩成最短标准答案（`--align`），76 条里大量是坏对齐（'John Whaite'→'John'、'Western Asia'→'Western'、列表只挑一项而思考说要列清单），默认关闭。

训练包 `data/sft/searchr1-g1k-v2-pack-20261027/`：**train 1030 / validation 60**（v1 是 821 / 40），按 bucket（含验证集）：v1 729、wrong 154、hard_ok 132、easy 75，普通题（v1+easy）约 74%。train 检索次数 1/2/3/4 = 543/315/73/99。token p50 1399、p99 4082、max 5288（约 10 行超 4096 → 训练用 `--ctx 6144`）。渲染格式与 v1 逐字节一致（system 段、`Assistant: <think` 结尾、训练段 `>思考</think>\n动作\n\nUser:`）。

## 试点与对照（决策依据）

| 目录 | 内容 | 结论 |
|---|---|---|
| `outputs/distill/v2_pilot` | 同环境 100 题 | 教师 EM 0.531（v1 BM25 时 0.467），有 4 次检索的正确轨迹 |
| `outputs/distill/v2_pilot_quiet` | 同 100 题，`--quiet_suffix` 要求思考里别提 Teacher notes | 无效：复述的步 6.4% → 7.3% |
| `scripts/leak_stats.py` | 泄漏率 / 思考长度统计 | 原始思考每步 p50 约 130 token，≤800 token 丢约 10% |
| `scripts/rewrite_think.py` | 让教师改写思考 | 用户决定不用（保留原始思考），留档 |

## 提交榜单时要说明

- RL 前做过 SFT 蒸馏：教师 Qwen3.7-Flash，题目只来自 NQ + HotpotQA 训练集，检索环境同榜单协议。
- 用 RL 中间权重 s300 给 SFT 选题（只用于挑题目难度，不用它的输出做训练数据）。
- agent 格式是 G1K 的 `<tool_call>`，不是 `<search>`；每轮最多生成 2048 token（上游 500）。

## 6. SFT v2 训练 + RL 过夜链（2026-10-27 21:24 启动）

全部在 Pro6000（$TRAIN_HOST）上，screen `sft2`，日志 `rl-sr1-v2/outputs/sft2-chain.log`。脚本：`scripts/chain_sft2.sh`（入口）、`scripts/train-miss-sft2.sh`、`scripts/chain_v5.sh`（加了 FILTER_OUT / RUN_OUT / DATA / RL_LOG 环境变量，默认值不变）。

1. **SFT**：基础模型（sha256 d8f8ecd4…）+ `segments-sft2.jsonl`（= 训练包 train.jsonl，sha256 f4085424…），MiSS r16 / alpha 16、全部目标、ctx 6144、batch 8、warmup 10、2 epoch（258 步）、lr 3e-5 与 5e-5（lr-final = lr/10），`--wkv_tape`。两档并行会和 e5（37GB）+ 推理服务一起爆显存，改串行；单档峰值 78.6GB（含 e5 和推理服务），约 3700 tok/s、每档约 16 分钟。产物 `/mnt/nvme0n1/no22/rwkv_lighting_cuda/train-miss-sft2/<run>/adapter-final.pth`，拷到 `rl-sr1-v2/adapters/<run>.pth`。
2. **选档**：两档按榜单协议评七集抽样（`outputs/upstream-eval/<run>-greedy-t3`），七集平均高的进 RL，结果写 `outputs/sft2-select.json`；低于 0.30 停链。
3. **RL 起点**：停推理服务 8100，`rwkv_miss_tune_server` 1.8.7 从选中的 adapter 冷启动：`--ctx 12288 --chunk 128 --max-batch 256 --cache-capacity 640 --keep-versions 2 --lr 2e-6 --output rl-out/server`（日志 `server-sft2.log`）。
4. **筛题**：题源 `data/search_r1/train_rl_sft2.parquet`（train.parquet 去掉候选 4000 题、教师 2000 题和所有蒸馏/试点题，共排除 6170，剩 163445；sha256 29fd8b90…），`filter_questions.py` T1.0 × 8、seed 888，只跑 n=1600 一段，mixed ≥ 600 即建池（下限 500），输出 `outputs/filter/sft2-e5t3`。v4 筛到 1500 mixed 用了 6.7 小时，过夜来不及，所以只筛一段。
5. **GRPO**：`rl_grpo.py` 16 题 × 8、T1.0、lr 2e-6、奖励 EM 1 / 答错 0（f1 0.3 部分分）/ 无合法答案 -0.2、`--topk 3 --max_obs_tokens 500 --max_turns 5 --max_turn_tokens 1024`（= 榜单协议 4 次检索 + 1 轮只答），200 步，每 10 步导出 adapter、每 50 步完整检查点。输出 `outputs/rl/sft2-grpo/`，日志 `outputs/rl-sft2-grpo.log`。

开训前冒烟：v1 SFT adapter 起服务端（ctx 12288，推理服务开着），1 步 16 × 8：显存峰值 56GB，单步 319 秒（rollout 153 + train 166），acc 0.375、mixed 9/16。

服务器上 `rl_grpo.py` / `filter_questions.py` / `eval_base.py` 与本机校验和一致（e3b7ada6… / 598be2b3… / f2e620aa…）。

## 7. 第一段 RL 结果与第二段（扩池续训，2026-10-28）

**第一段**（`outputs/rl/sft2-grpo`，服务端 run 前缀 `d53cd0acfc854114943d3b2a4e027a68`）：选中 `sft2-r16-lr3e-5-e2`；筛 1600 题得池 734 题，跑到客户端 step 99 后手动停（step 91 走完第 2 轮，mixed 10 步均值 14.2 → 8.6）。题池 acc 10 步均值 0.549 → 0.63（40 步后走平），fmt 0.913 → 0.98（前 10 步完成）。

榜单协议七集（`outputs/upstream-eval/<run>-greedy-t3`，推理服务 `--decode-batch-size 1024`）：

| | 2wiki | bamboogle | hotpotqa | musique | nq | popqa | triviaqa | 七集平均 | format_ok | 检索 |
|---|---|---|---|---|---|---|---|---|---|---|
| SFT v2 lr3e-5 | 0.377 | 0.432 | 0.380 | 0.153 | 0.340 | 0.393 | 0.577 | 0.3789 | 0.811 | 2.23 |
| RL s50 | 0.330 | 0.424 | 0.353 | 0.183 | 0.383 | 0.417 | 0.570 | 0.3801 | 0.931 | 2.12 |
| RL s100 | 0.390 | 0.432 | 0.387 | 0.183 | 0.377 | 0.430 | 0.583 | 0.3974 | 0.990 | 2.48 |

判读：格式红利（0.81 → 0.99）只换来约 2 点；s50 → s100 检索次数上升、多跳集回升，是行为期签名。s100 与旧路线最好（s300 / v2-s200 0.399）持平。

**第二段**：`scripts/chain_sft2_r2.sh`（入口，日志 `outputs/sft2-r2-chain.log`）。停 8100 → 修 step-100 `checkpoint.json` 的 null 键 → 题源 `train_rl_sft2_r2.parquet` = `train_rl_sft2.parquet` 去掉 `outputs/filter/sft2-e5t3/questions.jsonl` 里全部已筛题（旧池不再进新池）→ 服务端 `--resume rl-out/server/d53cd0ac…_1-step-100`（其余参数同第 6 节，日志 `server-sft2-r2.log`）→ `chain_v5.sh`：s100 策略 T1.0 × 8、seed 888，`N_LIST="1600 2400 3200 4000"`，mixed ≥ 1000 建池（下限 800），输出 `outputs/filter/sft2-s100-e5t3`；GRPO 参数同第 6 节，100 步（服务端 step 100 → 200），输出 `outputs/rl/sft2-grpo-r2`。`chain_v5.sh` 新增 `N_LIST` 环境变量，默认值不变。
