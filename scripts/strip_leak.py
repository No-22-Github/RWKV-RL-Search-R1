"""删掉 teacher 思考里复述附加指令的句子，其余原文不动。

teacher 会在思考里复述只有它看得见的 Teacher notes（"always run at least one search" 等），在 prompt 里叫它别提也压不住
（v2_pilot_quiet 对照：复述率不降）。泄漏基本是整句出现，这里按句切分、整句删除，不调 API。
删完仍有泄漏或删掉一半以上的步判失败，整条轨迹打 leak_fail（打包时丢弃）。原文保留在 reasoning_raw。

在 v1 + 两次 v2 试点的答对轨迹上：255 个泄漏步删成功 252 个。
"""

import argparse
import json
import re

# 只抓 Teacher notes 独有的内容；学生 prompt 里本来就有的（"short entity or phrase"、"untrusted data"）引用不算泄漏
LEAK_RE = re.compile(
    r"(teacher note|at least one search|even if you think you know|as short as possible|"
    r"only the entity, name, number or date|keep (your|my) thinking brief|no markdown)", re.I)

# 句末标点（可带右引号/括号）+ 空白 + 下一句开头
SENT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z*(\[\"'0-9])")
BULLET_RE = re.compile(r"^\s*(\d+\.|[*-])\s*")


def strip_text(text: str) -> str:
    out = []
    for line in text.split("\n"):
        sents = SENT_RE.split(line)
        keep = [s for s in sents if not LEAK_RE.search(s)]
        if len(keep) == len(sents):
            out.append(line)
            continue
        new = " ".join(keep).rstrip()
        if re.search(r"[A-Za-z0-9]", BULLET_RE.sub("", new)):
            out.append(new)  # 整行只剩列表符号就整行删掉
    return "\n".join(out)


def fix_step(st: dict) -> str:
    if not LEAK_RE.search(st["reasoning"]):
        return "clean"
    new = strip_text(st["reasoning"])
    if LEAK_RE.search(new) or len(new) < 0.5 * len(st["reasoning"]):
        return "fail"
    st["reasoning_raw"], st["reasoning"] = st["reasoning"], new
    return "fixed"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="distill_teacher 的 trajectories.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    trajs = [json.loads(l) for l in open(args.inp)]
    res = []
    for t in trajs:
        if t.get("status") != "answer" or not t.get("correct"):
            continue
        r = [fix_step(st) for st in t["steps"]]
        t["leak_fail"] = "fail" in r
        res += r
    with open(args.out, "w") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print({k: res.count(k) for k in ("clean", "fixed", "fail")},
          "trajs dropped:", sum(t.get("leak_fail", False) for t in trajs))


if __name__ == "__main__":
    main()
