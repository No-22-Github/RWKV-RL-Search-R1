#!/usr/bin/env bash
# 任意 adapter 按榜单协议测七集抽样（贪心），并和 sft2-rl-s100 / sft2r2-s150 对比
# 推理服务 8100 没起就以 1024 并发起一个（screen infer）；训练服务端跑着时别用
# 用法（服务器包根目录）：bash scripts/eval_adapter.sh <adapter_id> <adapter.pth 路径>
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
API=http://127.0.0.1:8100
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
ID=${1:?adapter_id}
CK=${2:?adapter.pth 路径}
DECODE_BS=${DECODE_BS:-1024}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

curl -s -m 3 http://127.0.0.1:8848/health >/dev/null && { say "FATAL: 8848 训练服务端还在，先停"; exit 1; }
if ! curl -s -m 3 "$API/v1/adapters" >/dev/null; then
  say "起推理服务 8100（decode-batch-size $DECODE_BS）"
  screen -dmS infer bash -c "cd $PWD && rwkv_lighting_cuda/rwkv_lighting_cuda --model-path $MODEL \
    --vocab-path ./rwkv_lighting_cuda/rwkv_vocab_v20230424.txt --host 0.0.0.0 --port 8100 \
    --decode-batch-size $DECODE_BS --chunk-size 128 --state-db-path infer_sessions.db 2>&1 | tee infer.log"
  for _ in $(seq 1 60); do sleep 5; curl -s -m 3 "$API/v1/adapters" >/dev/null && break; done
  curl -s -m 3 "$API/v1/adapters" >/dev/null || { say "FATAL: 8100 没起来，看 infer.log"; exit 1; }
fi
curl -sS -X POST "$API/v1/adapters" -H 'Content-Type: application/json' \
  --data "{\"adapter_id\":\"$ID\",\"path\":\"$(realpath "$CK")\"}"; echo

OUT=outputs/upstream-eval/$ID-greedy-t3
say "开测 $ID -> $OUT"
$PY scripts/eval_base.py --data data/search_r1/test_upstream7_s300.parquet --format g1k --think full \
  --max_turns 4 --final_answer_turn --max_obs_tokens 500 --greedy --max_turn_tokens 2048 --topk 3 \
  --api $API --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64 \
  --adapter_id "$ID" --out "$OUT" > "$OUT.log" 2>&1 || { say "FAIL，看 $OUT.log"; exit 1; }

$PY - "$OUT/report.json" <<'PYEOF'
import json, sys
runs = {"s100": "outputs/upstream-eval/sft2-rl-s100-greedy-t3/report.json",
        "s150": "outputs/upstream-eval/sft2r2-s150-greedy-t3/report.json", "新": sys.argv[1]}
R = {k: json.load(open(p))["report"] for k, p in runs.items()}
sets = [k for k in R["新"] if k != "all"]
print("set".ljust(16) + "".join(k.rjust(8) for k in R))
for s in sets:
    print(s.ljust(16) + "".join(f"{R[k][s]['acc@1(EM)']:8.3f}" for k in R))
print("avg7".ljust(16) + "".join(f"{sum(R[k][s]['acc@1(EM)'] for s in sets) / len(sets):8.3f}" for k in R))
a = R["新"]["all"]
print(f"format_ok {a['format_ok']}  avg_search {a['avg_search']}  end {a['end']}")
PYEOF
say "评测完成 $ID"
