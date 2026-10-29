"""难度筛题：用当前服务端策略（建议用 RL 起点 adapter 启动）对训练题做 T1.0 采样，
找出组内有对有错的题（GRPO 有学习信号），写成 parquet 给 rl_grpo.py --data 用。

用法（服务端和 BM25 检索都要先起好）：
  uv run python scripts/filter_questions.py --server http://127.0.0.1:8848 \
    --data data/search_r1/train.parquet --out outputs/filter/s100 --n 2400

- 从 train.parquet 里按 --seed 的固定 permutation 取前 --n 题（加大 --n 重跑可增量续筛，
  已筛过的题跳过）。
- 每题 --group 次（默认 8，与训练一致），每批 --batch_questions 题，只采样不训练，
  服务端策略保持启动时的版本不动，筛完可以直接接着开 rl_grpo.py。
- 结果 questions.jsonl 每题一行（对几条、格式成功率、检索次数等），最后写
  filtered.parquet（默认保留 0 < correct < group 的题）并打印分布。
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_grpo import Server, new_rows, rollout, score  # noqa: E402
from eval_base import G1KFormat  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True)
    p.add_argument("--retriever", default="http://127.0.0.1:8000")
    p.add_argument("--data", default="data/search_r1/train.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=2400, help="从 permutation 取前多少题")
    p.add_argument("--group", type=int, default=8, help="每题采样数，与训练组大小一致")
    p.add_argument("--batch_questions", type=int, default=16, help="每批筛多少题（自动压到 max_batch//group 内）")
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--max_turn_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--max_obs_tokens", type=int, default=0, help="检索结果截断，须与训练一致（榜单协议 500）")
    p.add_argument("--min_correct", type=int, default=1, help="filtered.parquet 保留的正确条数下限")
    p.add_argument("--max_correct", type=int, default=None, help="正确条数上限，默认 group-1（全对剔除）")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--timeout", type=float, default=3600)
    a = p.parse_args()
    if a.max_correct is None:
        a.max_correct = a.group - 1

    os.makedirs(a.out, exist_ok=True)
    qpath = os.path.join(a.out, "questions.jsonl")
    srv = Server(a.server, a.timeout)
    health = srv.get("/health")
    a.ctx, a.max_batch = health["ctx"], health["max_batch"]
    a.r_wrong, a.r_invalid, a.f1_weight = 0.0, 0.0, 0.0
    a.batch_questions = min(a.batch_questions, a.max_batch // a.group)
    if a.batch_questions < 1:
        sys.exit(f"max_batch {a.max_batch} 连一组（{a.group}）都放不下")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump({"args": {k: v for k, v in vars(a).items()}, "health": health}, f, ensure_ascii=False, indent=2,
                  default=str)

    df = pd.read_parquet(a.data)
    order = np.random.default_rng(a.seed).permutation(len(df))[: a.n]
    df = df.iloc[order].reset_index(drop=True)

    # train.parquet 的 id 在 nq/hotpotqa 之间会撞号，所有去重/记录必须按 (id, data_source)
    done = {}
    if os.path.exists(qpath):
        with open(qpath) as f:
            for line in f:
                r = json.loads(line)
                done[(r["qid"], r["source"])] = r
    todo = [i for i, q in enumerate(df.itertuples()) if (q.id, q.data_source) not in done]
    print(f"policy {health['policy_version']} gemm {health.get('gemm_mode')}: {len(done)} 已筛, "
          f"{len(todo)} 待筛, 每批 {a.batch_questions} 题", flush=True)

    fmt = G1KFormat("full")
    t0 = time.time()
    for ci in range(0, len(todo), a.batch_questions):
        chunk = todo[ci : ci + a.batch_questions]
        rows = new_rows(df.iloc[chunk], fmt, srv, a.group)
        rollout(rows, fmt, srv, health["policy_version"], a, a.seed + ci)
        score(rows, a)
        by_q = {}
        for r in rows:
            key = (r["qid"], r["source"])
            d = by_q.setdefault(key, {"qid": r["qid"], "source": r["source"], "correct": 0, "fmt": 0,
                                      "search": 0, "gen": 0, "end": {}})
            d["correct"] += r["correct"]
            d["fmt"] += r["format_ok"]
            d["search"] += r["n_search"]
            d["gen"] += r["gen_tokens"]
            d["end"][r["end"]] = d["end"].get(r["end"], 0) + 1
        with open(qpath, "a") as f:
            for key in [(r["qid"], r["source"]) for r in rows[:: a.group]]:
                d = by_q[key]
                d["avg_search"], d["avg_gen"] = round(d["search"] / a.group, 2), round(d["gen"] / a.group)
                del d["search"], d["gen"]
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                done[key] = d
        kept = sum(a.min_correct <= d["correct"] <= a.max_correct for d in done.values())
        fdone = len(done)
        print(f"{fdone}/{len(df)}: mixed {kept} ({kept / fdone:.0%})  "
              f"{(time.time() - t0) / max(ci // a.batch_questions + 1, 1):.1f}s/批", flush=True)

    correct = np.array([d["correct"] for d in done.values()])
    print("正确条数分布:", {int(k): int((correct == k).sum()) for k in range(a.group + 1)
                        if (correct == k).sum()})
    keep = {k for k, d in done.items() if a.min_correct <= d["correct"] <= a.max_correct}
    sub = df[[(i, s) in keep for i, s in zip(df.id, df.data_source)]].reset_index(drop=True)
    assert sub.id.is_unique or sub.duplicated(["id", "data_source"]).sum() == 0
    fpath = os.path.join(a.out, "filtered.parquet")
    sub.to_parquet(fpath)
    print(f"保留 {len(sub)}/{len(done)} 题 -> {fpath}", flush=True)


if __name__ == "__main__":
    main()
