# RWKV-RL-Search-R1

在 Search-R1（NQ / HotpotQA）环境上给 RWKV 做 RL。

## 环境

```bash
uv sync                    # 主依赖：torch、rwkv、datasets 等
uv sync --all-groups       # 另外装上检索服务（faiss + e5）
```

- macOS：PyPI 默认版本的 torch（MPS），检索用 `faiss-cpu`
- Linux：CUDA 13.0 版 torch（`download.pytorch.org/whl/cu130`），检索用 `faiss-gpu-cu12`

## 环境变量

服务地址、服务器主机和教师接口的密钥都不入库，统一放在 `.env`：

```bash
cp .env.example .env      # 填上真实地址和密钥
set -a; source .env; set +a
```

脚本的默认地址都读这些变量（`RWKV_API`、`RETRIEVER_URL`、`TUNE_SERVER` 等）。文档和归档的 `report.json` 里出现的 `$RWKV_API`、`$TRAIN_HOST` 之类，都是指 `.env` 里对应的值。

## 本地 BM25 检索

```bash
uv run python retriever/build_corpus_store.py   # 一次性：wiki-18 语料 → 分块压缩存储（约 7 分钟，内存 <100MB）
uv run python retriever/bm25_server.py          # 127.0.0.1:8000/retrieve，接口同 Search-R1
```

需要 Java 21（默认找 Homebrew 的 `openjdk@21`，也可以自己设 `JAVA_HOME`）。服务常驻内存约 700MB。

## 推理 API

2.9B 基模部署在 `$RWKV_API`，后端是 [rwkv_lightning_cuda](https://github.com/Alic-Li/rwkv_lightning_cuda)，用 `/v1/batch/completions` 做续写。

## 目录

- `data/`：Search-R1 数据（`PeterJinGo/nq_hotpotqa_train`）
- `models/`：RWKV 权重
- `outputs/`：评测和训练输出
- `docs/`：实验报告。先看全线总览 [docs/rl-lineage.md](docs/rl-lineage.md)（两条 SFT + RL 路线、全部检查点、折线图、下一步）；最近一期是 [RL 状态 2026-10-29](docs/rl-status-2026-10-29.md)
- `provenance/`：各阶段的留存快照（训练指标、脚本副本、命令、校验和）。全线训练指标在 `provenance/rl-lineage-20261029/`，重画图用 `.venv/bin/python scripts/plot_rl.py`
