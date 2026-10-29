"""rwkv_miss_tune_server 的本地替身，只用来联调 rl_grpo.py，不做任何训练。

- /v1/tokenize、/v1/detokenize：本地按 RWKV 词表贪心最长匹配（与服务端 TrieTokenizer 同算法）。
- /v1/completions：把 token 历史解码成文本，转发给推理服务（/v1/batch/completions，带 adapter），
  生成文本再编码回 token；logprobs 是假的。
- /v1/logprobs：返回假值。
- /v1/train：按服务端的规则校验请求（turns/loss_mask/advantages 对齐、ctx、版本），不更新任何东西。
- /v1/publish：只换版本号。

  uv run python scripts/fake_rl_server.py --port 8848 --adapter_id sr1v1-r16-lr5e-5-e2
"""

import argparse
import ast
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


class Tokenizer:
    def __init__(self, path: str):
        self.idx2bytes = {}
        for line in open(path, encoding="utf-8"):
            i = int(line[: line.index(" ")])
            x = ast.literal_eval(line[line.index(" ") : line.rindex(" ")])
            self.idx2bytes[i] = x.encode("utf-8") if isinstance(x, str) else x
        self.bytes2idx = {v: k for k, v in self.idx2bytes.items()}
        self.maxlen = max(len(v) for v in self.idx2bytes.values())

    def encode(self, text: str) -> list[int]:
        b, out, i = text.encode("utf-8"), [], 0
        while i < len(b):
            for n in range(min(self.maxlen, len(b) - i), 0, -1):
                t = self.bytes2idx.get(b[i : i + n])
                if t is not None:
                    out.append(t)
                    i += n
                    break
            else:
                raise ValueError(f"untokenizable byte at {i}")
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.idx2bytes.get(i, b"") for i in ids).decode("utf-8", errors="replace")


class State:
    def __init__(self, a):
        self.a = a
        self.tok = Tokenizer(a.vocab)
        self.step = 0
        self.published = 0
        self.version = "v0"
        self.reference = "v0"
        self.lock = threading.Lock()

    def health(self, _):
        return {"step": self.step, "published_step": self.published, "policy_version": self.version,
                "reference_version": self.reference, "reference_step": 0, "ctx": self.a.ctx,
                "max_batch": self.a.max_batch, "vocab_size": 65536, "fake": True}

    def check_version(self, r, train=False):
        v = r.get("policy_version", self.version)
        if v not in (self.version, self.reference):
            raise HTTPError(409, "policy version is unknown or retired")
        if train and (v != self.version or self.published != self.step):
            raise HTTPError(409, "stale rollout")

    def completions(self, r):
        self.check_version(r)
        prompts = r["prompts"]
        if not 0 < len(prompts) <= self.a.max_batch:
            raise HTTPError(400, "invalid prompts batch")
        for p in prompts:
            if len(p) + r["max_tokens"] > self.a.ctx:
                raise HTTPError(400, "prompt + max_tokens exceeds ctx")
        body = {"contents": [self.tok.decode(p) for p in prompts], "max_tokens": r["max_tokens"],
                "stop": r.get("stop", []), "temperature": r.get("temperature", 1.0), "top_p": 1.0,
                "adapter_id": self.a.adapter_id}
        # 和 eval_base 一样走 SSE：非流式长请求会被网关在 60s 左右断成 502
        res = requests.post(f"{self.a.api}/v1/batch/completions", json={**body, "stream": True, "chunk_size": 32},
                            stream=True, timeout=(10, 600))
        res.raise_for_status()
        texts, fins = {}, {}
        for line in res.iter_lines():
            if not line.startswith(b"data: "):
                continue
            if line[6:] == b"[DONE]":
                break
            ev = json.loads(line[6:])
            if "error" in ev:
                raise RuntimeError(ev["error"])
            for c in ev["choices"]:
                texts[c["index"]] = texts.get(c["index"], "") + c["delta"].get("content", "")
                if c.get("finish_reason"):
                    fins[c["index"]] = c["finish_reason"]
        out = []
        for i in range(len(prompts)):
            text, finish = texts.get(i, ""), fins.get(i, "length")
            ids = self.tok.encode(text)
            matched = next((s for s in r.get("stop", []) if text.endswith(s)), None)
            out.append({"tokens": ids, "logprobs": [-1.0] * len(ids), "text": text,
                        "finish_reason": "stop" if matched else finish,
                        **({"matched_stop": matched} if matched else {}),
                        "cached_prompt_tokens": 0, "prefill_tokens": 0})
        return {"policy_version": r.get("policy_version", self.version), "step": self.published,
                "completions": out, "generated_tokens": sum(len(c["tokens"]) for c in out)}

    def logprobs(self, r):
        self.check_version(r)
        return {"logprobs": [[-1.0] * (len(s) - 1) for s in r["sequences"]]}

    def train(self, r):
        with self.lock:
            self.check_version(r, train=True)
            kept = positions = 0
            for s in r["samples"]:
                n = 0
                turns = s["turns"] if "turns" in s else [s]
                if "turns" in s and isinstance(s.get("advantages"), list):
                    raise HTTPError(400, "multi-turn sample-level advantages must be scalar")
                for t in turns:
                    ids, mask = t["tokens"], t["loss_mask"]
                    adv = t.get("advantages", s.get("advantages"))
                    if len(mask) != len(ids):
                        raise HTTPError(400, "loss_mask must align with tokens")
                    if isinstance(adv, list) and len(adv) != len(ids):
                        raise HTTPError(400, "advantages must be token-aligned")
                    for j, m in enumerate(mask):
                        if m not in (0, 1, True, False):
                            raise HTTPError(400, "bad loss_mask entry")
                        if m and n + j > 0:
                            kept += 1
                    n += len(ids)
                    if n > self.a.ctx:
                        raise HTTPError(400, f"training history of {n} tokens exceeds ctx")
                positions += n - 1
            if not kept:
                raise HTTPError(400, "batch has no selected prediction targets")
            self.step += 1
            return {"loss": 0.0, "grad_norm": 0.0, "step": self.step, "trained_tokens": kept,
                    "total_tokens": positions, "updated": True, "publish_required": True, "seconds": 0.0}

    def publish(self, r):
        with self.lock:
            if r["expected_step"] != self.step:
                raise HTTPError(409, "learner step changed")
            if self.published != self.step:
                self.published = self.step
                self.version = f"v{self.step}"
            out = {"policy_version": self.version, "step": self.step}
            if r.get("save") or r.get("checkpoint"):
                out["directory"] = f"fake/step-{self.step}"
            if r.get("checkpoint"):
                out["checkpoint"] = f"fake/step-{self.step}/training"
            return out


class HTTPError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8848)
    p.add_argument("--api", default=os.environ.get("RWKV_API", "http://127.0.0.1:8100"),
                   help="推理服务地址，默认读环境变量 RWKV_API")
    p.add_argument("--adapter_id", default="sr1v1-r16-lr5e-5-e2")
    p.add_argument("--vocab", default=os.environ.get("RWKV_VOCAB", "rwkv_vocab_v20230424.txt"),
                   help="词表路径，默认读环境变量 RWKV_VOCAB")
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--max_batch", type=int, default=64)
    st = State(p.parse_args())
    routes = {"/health": st.health, "/v1/completions": st.completions, "/v1/logprobs": st.logprobs,
              "/v1/train": st.train, "/v1/publish": st.publish,
              "/v1/tokenize": lambda r: {"tokens": st.tok.encode(r["text"])},
              "/v1/detokenize": lambda r: {"text": st.tok.decode(r["tokens"])}}

    class H(BaseHTTPRequestHandler):
        def handle_one(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else {}
                if self.path not in routes:
                    raise HTTPError(404, "no route")
                code, out = 200, routes[self.path](body)
            except HTTPError as e:
                code, out = e.code, {"error": str(e)}
            except Exception as e:
                code, out = 500, {"error": f"{type(e).__name__}: {e}"}
            data = json.dumps(out).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = handle_one

        def log_message(self, *args):
            pass

    print(f"fake rl server on :{st.a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", st.a.port), H).serve_forever()


if __name__ == "__main__":
    main()
