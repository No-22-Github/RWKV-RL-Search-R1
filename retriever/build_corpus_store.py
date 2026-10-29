"""把 wiki-18 语料转成分块压缩存储，支持按行号随机读取，内存占用很小。

wiki-18.jsonl.gz 实际是 tar.gz（里面只有一个 jsonl）。这里流式读取，每 BLOCK 条文档
用 zlib 压成一块写进 corpus.bin，块偏移写进 offsets.npy。BM25 索引的 docid 就是行号。
"""

import argparse
import json
import os
import tarfile
import zlib

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", default="data/retrieval/wiki-18.jsonl.gz")
    parser.add_argument("--out_dir", default="data/retrieval/wiki-18-store")
    parser.add_argument("--block", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    offsets = [0]
    buf, n, mismatched = [], 0, 0
    with tarfile.open(args.corpus_path, mode="r|gz") as tar, open(os.path.join(args.out_dir, "corpus.bin"), "wb") as out:
        member = next(m for m in tar if m.isfile())
        f = tar.extractfile(member)
        for line in tqdm(f, desc="docs", unit_scale=True):
            doc = json.loads(line)
            if doc["id"] != str(n):
                mismatched += 1
            buf.append(doc["contents"])
            n += 1
            if len(buf) == args.block:
                out.write(zlib.compress(json.dumps(buf).encode()))
                offsets.append(out.tell())
                buf = []
        if buf:
            out.write(zlib.compress(json.dumps(buf).encode()))
            offsets.append(out.tell())

    np.save(os.path.join(args.out_dir, "offsets.npy"), np.array(offsets, dtype=np.int64))
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"num_docs": n, "block": args.block, "id_mismatch": mismatched}, f)
    print(f"docs={n} blocks={len(offsets) - 1} id_mismatch={mismatched}")


if __name__ == "__main__":
    main()
