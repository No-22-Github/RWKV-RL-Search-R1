"""把 teacher 的原始推理改写成短而朴素的思考，再写回 trajectories.jsonl（原始推理保留在 reasoning_raw）。

teacher 的 reasoning 常是 Markdown 列表，还会在看到检索结果之前“预演”结果；2.9B 学这种思考既浪费 token 又学会编造。
改写只看题目、之前的检索结果和本步动作，要求 2–4 句纯文本、不出现未见过的事实、结论与动作一致。

凭据从环境变量读：API_BASE、API_KEY。
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

PROMPT = """You are cleaning up the private reasoning of a search agent so a small model can imitate it.

Question: {question}

What the agent has seen so far (search results of earlier steps):
{history}

The agent's original reasoning for this step:
<<<
{reasoning}
>>>

The action the agent then took:
{action}

Rewrite the reasoning as 2-4 short plain sentences in first person, no Markdown, no lists, no quotes of the instructions.
Rules:
- Use only facts from the question, the search results shown above, or common knowledge the original reasoning relied on. Never describe search results that have not been shown yet.
- If the action is a search, say what is still unknown and why this query helps. If the action is an answer, point to the evidence and state the answer.
- The conclusion must match the action exactly.
Output only the rewritten reasoning."""


def chat(base, key, model, content):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.2,
            "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(5):
        try:
            r = requests.post(f"{base}/chat/completions", json=body, timeout=300, headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def history_text(steps):
    out = []
    for i, st in enumerate(steps, 1):
        if st["action"]["type"] == "search":
            docs = " | ".join(f"{d['title']}: {d['text'][:300]}" for d in st.get("docs", []))
            out.append(f"[search {i}] {st['action']['query']} -> {docs}")
    return "\n".join(out) or "(nothing yet)"


def rewrite(traj, args, base, key):
    for k, st in enumerate(traj["steps"]):
        if "reasoning_raw" in st:
            continue
        content = PROMPT.format(question=traj["question"], history=history_text(traj["steps"][:k]),
                                reasoning=st["reasoning"][:6000], action=st["action"]["text"])
        new = chat(base, key, args.model, content)
        st["reasoning_raw"], st["reasoning"] = st["reasoning"], new
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True, help="distill_teacher 的 trajectories.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3.7-Flash")
    p.add_argument("--parallel", type=int, default=32)
    p.add_argument("--only_correct", action="store_true", help="只改写答对的轨迹（其余原样写出）")
    args = p.parse_args()
    base, key = os.environ["API_BASE"].rstrip("/"), os.environ["API_KEY"]

    trajs = [json.loads(l) for l in open(args.inp)]
    todo = [t for t in trajs if t.get("status") == "answer" and (t.get("correct") or not args.only_correct)]
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(rewrite, t, args, base, key) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done, {time.time() - t0:.0f}s", flush=True)
    with open(args.out, "w") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
