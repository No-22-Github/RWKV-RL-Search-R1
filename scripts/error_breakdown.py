"""错题归因：把评测 samples.jsonl 里答错的样本分成几类，看剩下的分丢在哪。

  未作答     没给出 <answer>（跑满轮次 / 超长截断 / 格式坏）
  措辞差一点 预测和标准答案互相包含，或 token F1 ≥ 0.5（答对了实体，EM 没算上）
  检索没查到 所有检索结果里都不含任何标准答案
  查到没读对 检索结果里有标准答案，但预测不对

用法：python scripts/error_breakdown.py outputs/upstream-eval/<run>/samples.jsonl
"""

import collections
import json
import re
import sys

sys.path.insert(0, "scripts")
from eval_base import normalize_answer  # noqa: E402

TOOL_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)


def f1(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = sum((collections.Counter(p) & collections.Counter(g)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def classify(r):
    if r["correct"]:
        return "对"
    golds = [normalize_answer(g) for g in r["golden"] if normalize_answer(g)]
    pred = r.get("pred")
    if r["end"] != "answer" or pred is None:
        return "未作答"
    p = normalize_answer(pred)
    if p and any(g in p or p in g for g in golds) or max(f1(pred, g) for g in r["golden"]) >= 0.5:
        return "措辞差一点"
    retrieved = normalize_answer(" ".join(TOOL_RE.findall(r["response"])))
    if any(g in retrieved for g in golds):
        return "查到没读对"
    return "检索没查到"


def main():
    rows = [json.loads(line) for line in open(sys.argv[1])]
    cats = ["对", "措辞差一点", "查到没读对", "检索没查到", "未作答"]
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        c = classify(r)
        by[r["source"]][c] += 1
        by["全部"][c] += 1
    print(f"{'集':<16}" + "".join(f"{c:>8}" for c in cats) + f"{'题数':>7}")
    for src in sorted(k for k in by if k != "全部") + ["全部"]:
        n = sum(by[src].values())
        print(f"{src:<16}" + "".join(f"{by[src][c] / n:8.3f}" for c in cats) + f"{n:7d}")


if __name__ == "__main__":
    main()
