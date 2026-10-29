"""探针的训练信号：每步实际 acc 减去按题池筛题标签（s100 在 T=1.0 下 8 次的正确率）算的期望 acc。

题目难度逐批变化，直接看训练 acc 噪声太大；扣掉期望之后的 diff 才看得出策略有没有在变强。
起点不是筛题策略时有固定偏移，只看走势。可以同时传几个 run 目录对照（同 seed 同题序时逐步配对）。

用法：python scripts/probe_diff.py outputs/rl/probe-lr1e-5 [outputs/rl/sft2-grpo-r2 ...] [--labels questions.jsonl]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_run(run, labels):
    args = json.load(open(os.path.join(run, "args.json")))["args"]
    df = pd.read_parquet(args["data"])
    order = np.random.default_rng(args["seed"]).permutation(len(df))
    rows = []
    for line in open(os.path.join(run, "metrics.jsonl")):
        m = json.loads(line)
        idx = order[(m["step"] * args["questions"]) % len(df):][: args["questions"]]
        exp = [labels.get((str(q.id), str(q.data_source))) for q in df.iloc[idx].itertuples()]
        exp = [e for e in exp if e is not None]
        rows.append({"step": m["step"], "acc": m["acc"], "exp": float(np.mean(exp)) if exp else float("nan"),
                     "search": m["avg_search"], "gen": m["avg_gen_tokens"], "fmt": m["format_ok"],
                     "gnorm": (m.get("train") or {}).get("grad_norm")})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", default="outputs/filter/sft2-s100-e5t3/questions.jsonl")
    p.add_argument("--bin", type=int, default=10)
    p.add_argument("--max_step", type=int, default=0, help=">0 时只看前 N 步（和短探针对齐）")
    a = p.parse_args()
    labels = {}
    for line in open(a.labels):
        if line.strip():
            r = json.loads(line)
            labels[(str(r["qid"]), str(r["source"]))] = r["correct"] / 8
    for run in a.runs:
        d = load_run(run, labels)
        if a.max_step:
            d = d[d.step < a.max_step]
        d["diff"] = d.acc - d.exp
        d["bin"] = d.step // a.bin * a.bin
        g = d.groupby("bin").agg(acc=("acc", "mean"), exp=("exp", "mean"), diff=("diff", "mean"),
                                 search=("search", "mean"), gen=("gen", "mean"), fmt=("fmt", "mean"),
                                 gnorm=("gnorm", "median"))
        print(f"== {run}（{len(d)} 步，总 diff {d['diff'].mean():+.3f}）")
        print(g.round(3).to_string())


if __name__ == "__main__":
    main()
