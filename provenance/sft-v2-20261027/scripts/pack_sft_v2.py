"""SFT v2 打包：教师轨迹 → 删泄漏句 → 答案对齐 → 过滤 → MiSS 训练包（train/validation.jsonl + meta + manifest）。

和 pack_sft.py（v1）的区别：
- 思考保留教师原文，只用 strip_leak 整句删掉复述 Teacher notes 的句子；删不干净的整条丢。
- 答案对齐（--align，默认关）：教师答案不 EM 但包含某个标准答案时，把 <answer> 换成被包含的最短标准答案。
  v2 实测 76 条里大量是坏对齐（'John Whaite'→'John'、'Western Asia'→'Western'、列表只挑一项而思考说要列清单），默认不用。
- 长度按 RWKV token 计：每步思考 ≤ --max_think_tokens。
- 丢掉有完全相同重复检索的轨迹。
- 按 (qid, source) 去重和切分（train.parquet 的 id 在 nq/hotpotqa 间撞号）。
"""

import argparse
import copy
import hashlib
import json
import os
import random
from collections import Counter

import rwkv
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER

from distill_teacher import render
from eval_base import G1KFormat, clean_txt, normalize_answer
from strip_leak import fix_step


def align_answer(t: dict) -> bool:
    """教师答案包含标准答案时收缩到最短的那个；成功返回 True。"""
    pred = normalize_answer(t["pred"] or "")
    cands = [g for g in t["golden"] if normalize_answer(g) and normalize_answer(g) in pred]
    if not cands:
        return False
    g = min(cands, key=lambda x: len(normalize_answer(x)))
    act = t["steps"][-1]["action"]
    act["answer"], act["text"] = g, f"<answer>{g}</answer>"
    t["pred_raw"], t["pred"], t["correct"] = t["pred"], g, True
    return True


def process(t: dict, args, tok) -> tuple[dict | None, str]:
    """返回 (处理后的轨迹, 原因)；轨迹为 None 表示丢弃。"""
    if not t.get("steps"):
        return None, "error"
    if t["status"] != "answer":
        return None, t["status"]
    t = copy.deepcopy(t)
    aligned = False
    if not t["correct"]:
        if not (args.align and t.get("hit") and align_answer(t)):
            return None, "wrong"
        aligned = True
    if t["n_search"] < args.min_search:
        return None, "no_search"
    queries = [s["action"]["query"] for s in t["steps"] if s["action"]["type"] == "search"]
    if len(set(q.lower() for q in queries)) < len(queries):
        return None, "dup_query"
    if any(not s["reasoning"].strip() for s in t["steps"]):
        return None, "empty_think"
    if "fail" in [fix_step(s) for s in t["steps"]]:
        return None, "leak_fail"
    if any(len(tok.encode(clean_txt(s["reasoning"]))) > args.max_think_tokens for s in t["steps"]):
        return None, "long_think"
    t["aligned"] = aligned
    t["leak_stripped"] = sum("reasoning_raw" in s for s in t["steps"])
    return t, "aligned" if aligned else "kept"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trajectories", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val_per_source", type=int, default=30)
    p.add_argument("--min_search", type=int, default=1)
    p.add_argument("--max_think_tokens", type=int, default=800)
    p.add_argument("--align", action="store_true", help="开启答案对齐（默认关，见文件头）")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    fmt = G1KFormat("full")
    tok = TRIE_TOKENIZER(os.path.join(os.path.dirname(rwkv.__file__), "rwkv_vocab_v20230424.txt"))

    kept, seen, reasons = [], set(), Counter()
    for path in args.trajectories:
        for line in open(path):
            t = json.loads(line)
            key = (t.get("qid"), t.get("source"))
            if key in seen:
                reasons["dup_question"] += 1
                continue
            t2, why = process(t, args, tok)
            reasons[why] += 1
            if t2 is not None:
                seen.add(key)
                kept.append(t2)
    random.Random(args.seed).shuffle(kept)

    val = set()
    for src in sorted({t["source"] for t in kept}):
        val |= {(t["qid"], t["source"]) for t in [t for t in kept if t["source"] == src][: args.val_per_source]}

    os.makedirs(args.out, exist_ok=True)
    manifest = {
        "inputs": [{"path": p, "sha256": sha(p)} for p in args.trajectories],
        "args": vars(args), "filter": dict(reasons.most_common()),
        "leak_stripped_steps": sum(t["leak_stripped"] for t in kept), "splits": {},
    }
    for split in ("train", "validation"):
        rows = [t for t in kept if ((t["qid"], t["source"]) in val) == (split == "validation")]
        path = os.path.join(args.out, f"{split}.jsonl")
        lens = []
        # rwkv_miss_tune 的解析器只接受 {"segments": [...]} 一个字段，meta 按行号另存
        with open(path, "w") as f, open(os.path.join(args.out, f"{split}.meta.jsonl"), "w") as fm:
            for t in rows:
                segs = render(t, fmt)
                lens.append(len(tok.encode("".join(s["text"] for s in segs))))
                f.write(json.dumps({"segments": segs}, ensure_ascii=False) + "\n")
                fm.write(json.dumps({"qid": t["qid"], "source": t["source"], "n_search": t["n_search"],
                                     "aligned": t["aligned"], "leak_stripped": t["leak_stripped"]}) + "\n")
        lens.sort()
        manifest["splits"][split] = {
            "path": path, "sha256": sha(path), "rows": len(rows),
            "by_source": dict(Counter(t["source"] for t in rows)),
            "by_n_search": dict(sorted(Counter(t["n_search"] for t in rows).items())),
            "aligned": sum(t["aligned"] for t in rows),
            "tokens": {"p50": lens[len(lens) // 2], "p99": lens[int(len(lens) * 0.99)], "max": lens[-1]} if lens else None,
        }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: manifest[k] for k in ("filter", "leak_stripped_steps", "splits")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
