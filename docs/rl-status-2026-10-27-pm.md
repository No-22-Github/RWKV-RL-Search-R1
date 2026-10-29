# Search-R1 RL 状态报告（下午）

日期：2026-10-27 ｜ 覆盖：榜单协议对齐 → v4 结案 → 链路评测 → v5 就绪

## TL;DR

- **找到可提交的榜单**：TencentBAC 的 [SearchAgent Leaderboard](https://huggingface.co/spaces/TencentBAC/SearchAgent_Leaderboard)（Search-R1 七集 + FictionalHot，8 集平均 EM，e5 top-3、最多 4 次检索、只许用 NQ+HotpotQA 训练集）。按抽样估计，s300 的 8 集平均约 35.3，暂列第 2（ReSeek-7B 37.74 第一），**3B 档第一稳拿**。
- **按榜单协议的七集平均：s300 0.399，v4-s100 0.404，7B Base PPO（Search-R1 论文）0.431**，差 2.7 个点，差距主要在 NQ 和 MuSiQue。
- **v4 结案：零增益**。从 s300 练了 115 步，三种口径（训练环境 top-5 / 3 次检索 / 榜单协议）held-out 都和 s300 打平，只是检索变多了。训练里 step 50–89 查过头，90 以后自己收回来，但这些都没有泛化到新题上。
- **链路评测**：基础模型 0.051 → SFT 0.318 → v1 0.341 → v2 0.399 → v3b 0.399 → v4 0.404。**收益几乎全部来自 SFT（+26.7，主要是学会格式和工具调用）和 v2（+5.8）**；自 v2-s200 起 RL 在 held-out 上一直为零。
- **评测提速 5–8 倍**：推理服务挪到 Pro6000（8100，`--decode-batch-size 1024`），客户端也在服务器上跑，七集抽样 1925 题 7.5 分钟。
- **v5 就绪**：训练环境对齐榜单协议（top-3、检索结果截 500 token、4 次检索 + 1 轮只答），从 s300 重筛重训，脚本 `chain_v5.sh` 已同步到服务器，还没开跑。

## 评测口径（本轮新增「榜单协议」）

| 口径 | 题库 | 检索 | 解码 | 轮次 |
|---|---|---|---|---|
| 内部口径（历史） | `test_sample.parquet` 200 题 × 8 | e5 Flat top-5，不截断 | T0.3 / top_p 0.5 | 3 次检索 + 提示收尾（`--max_turns 4`） |
| **榜单协议** | `test_upstream7_s300.parquet` 七集各 300（Bamboogle 125），共 1925 | e5 Flat top-3，每次截 500 token | 贪心（n=1, top_k=1） | 4 次检索 + 1 轮只答（`--max_turns 4 --final_answer_turn`） |
| FictionalHot 抽样 | `data/fictionalhot/fictionalhot_s100.parquet` 六子集各 100 | 本机 BM25 + 虚构文档（8004，低配） | 同榜单协议 | 同榜单协议 |

七集平均 = 每集 EM 的简单平均（同论文）。单集 300 题误差约 ±2.8 点，七集平均约 ±1.2 点；Bamboogle 只有 125 题，约 ±4.3。

榜单协议的评测命令（服务器上，结果在 `outputs/upstream-eval/<name>/`）：

```bash
.venv/bin/python scripts/eval_base.py --data data/search_r1/test_upstream7_s300.parquet --format g1k --think full --max_turns 4 --final_answer_turn --max_obs_tokens 500 --greedy --max_turn_tokens 2048 --topk 3 --api http://127.0.0.1:8100 --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64 --adapter_id <id> --out outputs/upstream-eval/<name>
```

## 一、链路评测（榜单协议，七集抽样）

| 阶段 | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | **七集平均** | format_ok | 检索次数 |
|---|---|---|---|---|---|---|---|---|---|---|
| 基础模型 | 0.027 | 0.147 | 0.033 | 0.067 | 0.017 | 0.027 | 0.040 | **0.051** | 0.196 | 2.73 |
| SFT（`sr1v1-r16-lr5e-5-e2`） | 0.327 | 0.513 | 0.347 | 0.320 | 0.247 | 0.103 | 0.368 | **0.318** | 0.759 | 2.28 |
| v1-s100 | 0.333 | 0.543 | 0.377 | 0.303 | 0.340 | 0.113 | 0.376 | **0.341** | 0.980 | 1.86 |
| v2-s200 | 0.383 | 0.620 | 0.437 | 0.387 | 0.393 | 0.157 | 0.416 | **0.399** | 0.997 | 2.86 |
| v3b-s300 | 0.397 | 0.627 | 0.443 | 0.397 | 0.390 | 0.163 | 0.376 | **0.399** | 0.998 | 2.29 |
| v4-s100 | 0.377 | 0.607 | 0.473 | 0.410 | 0.370 | 0.150 | 0.440 | **0.404** | 0.998 | 3.01 |
| 7B Base PPO（论文） | 0.480 | 0.638 | 0.457 | 0.433 | 0.382 | 0.196 | 0.432 | **0.431** | — | — |

读法：
- 基础模型 → SFT 的 +26.7 主要是格式和工具调用（format_ok 0.196 → 0.759），不代表知识变多了；SFT 还有约 24% 未收尾，v1 把它修到 0.98。
- RL 里真正带来分数的只有 v2（+5.8）。v3b、v4 在榜单协议下都是零增益。
- v2-s200 和 s300 七集平均相同，但 v2-s200 检索更多（2.86 vs 2.29），s300 的 NQ/TriviaQA 略高、Bamboogle 略低。

## 二、v4 结案

v4 = s300 起、e5 Flat top-5 重筛题池（1664 题）、16 题 × 8、lr 2e-6、f1 0.3、`--max_turns 4`（3 次检索 + 提示收尾）。训练到 step 115，用户手动停训。

| 口径 | s300 | v4-s50 | v4-s100 | v4-s110 |
|---|---|---|---|---|
| 训练环境（top-5，T0.3×8，Pro6000） | 0.474 | — | 0.479 | — |
| 训练式收尾（七集，top-3，3 次检索） | 0.390 | — | 0.395 | — |
| 榜单协议（七集） | 0.399 | 0.385 | 0.404 | 0.392 |
| 榜单协议检索次数 | 2.29 | 3.20 | 3.01 | 2.86 |

- **s50 查过头**：72% 的题查满 4 次，这部分正确率只有 0.289，其中 34% 出现完全相同的重复检索；91 条在最后一轮因为超过 2048 token 被截断。
- **训练动态**：step 50–89 比筛题期望多查约 1 次（diff 跌到最低）；step 90 以后自己收回，到 step 110 检索次数回到约 2.0。step 104 以后是第二遍题（背题）。
- **训练日志里「扣除难度后的 acc」上升，是题池内部的收益，held-out 没有跟着涨**，同 v2 时的教训。
- 原因分析：训练环境和榜单协议不一致（top-5 对 top-3、3 次后提示收尾对 4 次），模型学成了「一直查到被叫停」；但就算在训练环境里也没涨，所以环境不对齐不是唯一原因。

### 训练侧的两个修复

- **ctx 溢出导致 /v1/train 400**（step 18）：overflow 轨迹末尾接上的检索结果让历史超过 8192。`rl_grpo.train_samples` 现在会去掉末尾全部 `loss_mask=0` 的分段（不影响梯度），剩下仍超长的样本跳过并打日志。
- **服务器上的 `filter_questions.py` 是旧版**（按裸 id 建池），v4 题池里混进了 18 道搭车题（1.1%，影响可忽略）。已同步成按 (id, source) 建池的版本。

## 三、FictionalHot

- 题库 `TencentBAC/FictionalHot`：5115 题（PopQA 1415 / 2Wiki 1247 / TriviaQA 1122 / HotpotQA 735 / NQ 358 / MuSiQue 238），转成 `data/fictionalhot/fictionalhot_eval.parquet`。
- 虚构文档在 `TencentBAC/ReSeek-corpus` 的 `hot-wiki-18.jsonl.gz` 里，追加在 tar 包之后：15587 篇，按段落切成 52645 段，存为 `data/fictionalhot/corpus/fictional_docs.jsonl`。腾讯的 e5 索引多出约 5.6 万条向量，他们的切块方式没有公开。
- 检索：`retriever/bm25_server.py --extra_jsonl ...`（本机 8004）。虚构段落按「插进 wiki-18 后」的口径打 BM25 分，与 Lucene 逐位一致，启动时自检。e5 版 `retriever/hot_merge_server.py` 已写好，未部署（上服务器前要改成 CUDA 编码，并用 `.venv/bin/python` 启动）。
- 结果（抽样 600，按全量题数加权）：s300 2.7%、v4-s50 2.0%（误差约 ±0.7）；榜单上各家 3.0–6.1。题库噪声大：标准答案只有约 31% 出现在虚构语料里。
- 优先级低：只占 8 集平均的 1/8，正式提交前再用 e5 版补测。

## 四、评测基建（Pro6000）

- 旧推理机（`$RWKV_API` 所在机器）是 W7900，推理服务默认只同时解码 64 行，高并发下很慢。
- Pro6000 上的 screen `infer`：`rl-sr1-v2/rwkv_lighting_cuda/rwkv_lighting_cuda`，模型 `/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth`，端口 **8100**（8000 被 BM25 占了），`--decode-batch-size 1024`，单批上限 1456。已注册：rl-sr1-{v1-s100, v2-s200, v3b-s300, v4-s50, v4-s100, v4-s110}、sr1v1-r16-lr5e-5-e2。
- `eval_base.py` 改动：生成请求原来走系统代理（断网重连后卡死），已修；改成按块独立走完全部轮次的流水线；检索加重试；新增 `--greedy` / `--final_answer_turn` / `--max_obs_tokens`。服务器 venv 没装 `rwkv` 包，分词器和词表已拷到 `scripts/` 下。
- **服务器上一律用 `.venv/bin/python`，不要 `uv run` / `uv sync`**（会把 torch 换成 CPU 版）。
- 本机权重已归档到 `checkpoints/rl/rl-sr1-*.pth`（附 SHA256SUMS）。

## 五、下一步

1. **v5 开训**（`chain_v5.sh`，已在服务器包根目录）：
   - 先停掉 8100 推理服务，或调小 `--decode-batch-size`，给训练腾显存；
   - 训练服务端用 `--ctx 12288 --resume <v3b step-300 完整检查点>` 重启（resume 前看一眼 `checkpoint.json` 的 null 键）；
   - `bash chain_v5.sh 2>&1 | tee -a outputs/v5-chain.log`：top-3、截 500、`--max_turns 5`，在新环境下重筛到 1500 道 mixed 题后自动开训 200 步，输出 `outputs/rl/sr1-v5`；
   - 每 25–50 步按榜单协议评一次（重起 8100 或训练间隙），基线 s300 0.399；
   - 起点也可以考虑 v2-s200（同分，检索更多），目前定 s300。
2. **SFT 值不值得重做**：链路评测显示 SFT 贡献最大，但主要是格式。如果 v5 跑 50–100 步仍是零增益，下一档手段是按榜单协议（e5 top-3、4 次检索）重新蒸馏、加大数据量（当年只有 843 条，用的还是 BM25）。
3. **提交前**：全量七集（5.2 万题，约 3.5 小时）+ e5 版 FictionalHot；补跑每轮 500 token 的版本；准备方法说明（SFT 蒸馏 + GRPO + 格式差异说明）。

## 产物位置

- 本机：`outputs/upstream-eval/`（榜单协议各权重，含 `lineage-*`）、`outputs/rl-eval/*-pro6000`（内部口径重测）、`checkpoints/rl/`（全部 adapter）。
- 服务器 `/mnt/nvme0n1/no22/rl-sr1-v2/`：`outputs/upstream-eval/`、`outputs/rl/sr1-v4/`（v4 训练指标）、`outputs/filter/s300-e5/`（v4 题池）、`scripts/eval_chain_1027.sh`、`scripts/eval_lineage_1027.sh`、`chain_v5.sh`。
