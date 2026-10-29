"""把每步的期望 acc（按题池标签算，见 probe_diff.py）导出成 expected.jsonl，画图时就不用再读题池 parquet。

用法（服务器包根目录）：python scripts/export_expected.py outputs/rl/<run> [...] [--labels questions.jsonl]
每个 run 目录下写 expected.jsonl，每行 {"step", "exp"}。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_diff import load_run  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--labels", default="outputs/filter/sft2-s100-e5t3/questions.jsonl")
    a = p.parse_args()
    labels = {}
    for line in open(a.labels):
        if line.strip():
            r = json.loads(line)
            labels[(str(r["qid"]), str(r["source"]))] = r["correct"] / 8
    for run in a.runs:
        d = load_run(run, labels)
        with open(os.path.join(run, "expected.jsonl"), "w") as f:
            for s, e in zip(d.step, d.exp):
                f.write(json.dumps({"step": int(s), "exp": round(float(e), 4)}) + "\n")
        print(f"{run}: {len(d)} 步")


if __name__ == "__main__":
    main()
