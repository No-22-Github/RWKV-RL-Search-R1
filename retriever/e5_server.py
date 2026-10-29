"""e5 精确检索服务（wiki-18 Flat 索引），双后端。

  --backend torch : 从 faiss 索引文件取出向量，fp16 常驻 GPU，分批矩阵乘检索。
                    Blackwell 上 faiss-gpu 没有 kernel（CUDA error 209），这是 GPU 路线。
  --backend faiss : faiss-cpu 全库扫描（慢，兜底用）。

接口与 bm25_server.py 同一套（POST /retrieve），默认端口 8002。
docid = wiki-18.jsonl 行号，正文从 data/retrieval/wiki-18-store 按行号取。
模型加载强制 local_files_only（网络受限时 from_pretrained 的在线检查会挂起）。

用法（服务器，GPU）：
  uv run --group retriever python retriever/e5_server.py \
      --index_path /mnt/nvme0n1/no22/e5-index/e5_Flat.index \
      --store_dir data/retrieval/wiki-18-store \
      --host 0.0.0.0 --port 8002 --backend torch
"""

import argparse
import json
import os
import time
import zlib
from functools import lru_cache

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


class CorpusStore:
    """读 build_corpus_store.py 生成的分块压缩语料，按行号取正文。"""

    def __init__(self, store_dir: str):
        with open(os.path.join(store_dir, "meta.json")) as f:
            self.block = json.load(f)["block"]
        self.offsets = np.load(os.path.join(store_dir, "offsets.npy"))
        self.f = open(os.path.join(store_dir, "corpus.bin"), "rb")
        self._read_block = lru_cache(maxsize=1024)(self._read_block)

    def _read_block(self, b: int) -> list[str]:
        start, end = int(self.offsets[b]), int(self.offsets[b + 1])
        return json.loads(zlib.decompress(os.pread(self.f.fileno(), end - start, start)))

    def __getitem__(self, i: int) -> dict:
        return {"id": str(i), "contents": self._read_block(i // self.block)[i % self.block]}


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


def _load_torch_vectors(index_path: str) -> torch.Tensor:
    """把 faiss 索引里的 float32 向量分块搬成 GPU fp16 张量（峰值 CPU 内存不翻倍）。"""
    t0 = time.monotonic()
    idx = faiss.read_index(index_path)
    n, d = idx.ntotal, idx.d
    print(f"[e5] index loaded: {type(idx).__name__} d={d} ntotal={n} "
          f"metric={'IP' if idx.metric_type == 0 else 'L2'} ({time.monotonic() - t0:.1f}s)", flush=True)
    vb = torch.empty((n, d), dtype=torch.float16, device="cuda")
    xb = faiss.rev_swig_ptr(idx.get_xb(), n * d).reshape(n, d)  # 零拷贝视图
    step = 2_000_000
    t1 = time.monotonic()
    for i in range(0, n, step):
        chunk = xb[i : i + step]
        vb[i : i + chunk.shape[0]].copy_(torch.from_numpy(chunk))
    del xb, idx  # 释放 faiss CPU 副本（64.6GB）
    torch.cuda.synchronize()
    print(f"[e5] torch backend ready: {n}x{d} fp16 on GPU "
          f"({time.monotonic() - t1:.1f}s, vram={torch.cuda.memory_allocated()/1e9:.1f}GB)", flush=True)
    return vb


class E5Index:
    def __init__(self, index_path: str, store_dir: str, model_name: str,
                 backend: str, use_gpu: bool, mmap: bool, encode_batch: int):
        self.backend = backend
        self.index = None
        self.vb = None
        if backend == "torch":
            assert torch.cuda.is_available(), "torch 后端需要 CUDA 版 torch（torch.cuda.is_available() 为 False）"
            self.vb = _load_torch_vectors(index_path)
        else:
            t0 = time.monotonic()
            if mmap:
                print("[e5] reading index with IO_FLAG_MMAP ...", flush=True)
                index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
            else:
                print("[e5] reading index into RAM ...", flush=True)
                index = faiss.read_index(index_path)
            print(f"[e5] index loaded: {type(index).__name__} d={index.d} "
                  f"ntotal={index.ntotal} metric={'IP' if index.metric_type == 0 else 'L2'} "
                  f"({time.monotonic() - t0:.1f}s)", flush=True)
            self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
            if use_gpu and not self.use_gpu:
                print("[e5] WARNING: --gpu requested but no GPU available, CPU fallback", flush=True)
            if self.use_gpu:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                t1 = time.monotonic()
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
                print(f"[e5] cloned to GPU 0, fp16 ({time.monotonic() - t1:.1f}s)", flush=True)
            else:
                faiss.omp_set_num_threads(max(1, os.cpu_count() or 1))
                print(f"[e5] faiss CPU, omp threads = {faiss.omp_get_max_threads()}", flush=True)
            self.index = index
        self.store = CorpusStore(store_dir)
        want_gpu = backend == "torch" or use_gpu
        self.device = "cuda" if (want_gpu and torch.cuda.is_available()) else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
        if self.device == "cuda":
            model = model.half()
        self.model = model.to(self.device).eval()
        self.encode_batch = encode_batch
        print(f"[e5] encoder device={self.device}", flush=True)

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> np.ndarray:
        feed = ["query: " + t for t in texts]
        out = []
        for i in range(0, len(feed), self.encode_batch):
            enc = self.tok(feed[i : i + self.encode_batch], padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            emb = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(emb, dim=-1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), dtype="float32")

    def _search_torch(self, qemb: np.ndarray, topk: int):
        nq = qemb.shape[0]
        all_s, all_i = [], []
        qchunk = 64  # (64 x 21M) fp16 中间量 ~2.7GB，显存无压力
        for i in range(0, nq, qchunk):
            qt = torch.from_numpy(np.ascontiguousarray(qemb[i : i + qchunk])).to(
                self.vb.device, dtype=self.vb.dtype)
            s = qt @ self.vb.T
            top = torch.topk(s, topk, dim=1)
            all_s.append(top.values.float().cpu().numpy())
            all_i.append(top.indices.cpu().numpy())
            del s
        return np.concatenate(all_s, 0), np.concatenate(all_i, 0)

    def retrieve(self, queries: list[str], topk: int) -> list[list[dict]]:
        t0 = time.monotonic()
        qemb = self._encode(queries)
        if self.backend == "torch":
            scores, idxs = self._search_torch(qemb, topk)
        else:
            scores, idxs = self.index.search(qemb, topk)
        result = []
        for qs, is_ in zip(scores, idxs):
            docs = []
            for s, i in zip(qs, is_):
                if i < 0:
                    continue
                doc = self.store[int(i)]
                docs.append({"document": {"id": doc["id"], "contents": doc["contents"]},
                             "score": float(s)})
            result.append(docs)
        print(f"[retrieve] n={len(queries)} topk={topk} {time.monotonic() - t0:.1f}s", flush=True)
        return result


def build_app(args) -> FastAPI:
    searcher = E5Index(args.index_path, args.store_dir, args.model,
                       args.backend, args.gpu, args.mmap, encode_batch=256)
    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(req: QueryRequest):
        topk = req.topk or args.topk
        result = searcher.retrieve(req.queries, topk)
        if not req.return_scores:
            result = [[{k: v for k, v in d.items() if k != "score"} for d in docs]
                       for docs in result]
        return {"result": result}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--store_dir", default="data/retrieval/wiki-18-store")
    parser.add_argument("--model", default="intfloat/e5-base-v2")
    parser.add_argument("--backend", choices=["torch", "faiss"], default="torch")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-mmap", dest="mmap", action="store_false")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
