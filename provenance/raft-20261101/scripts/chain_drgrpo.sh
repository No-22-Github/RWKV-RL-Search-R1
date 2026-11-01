#!/usr/bin/env bash
# 10-31 夜：Dr.GRPO 学习率扫描 → 选档 → 每步 32 题长跑。都从 SFT v2 RL s100 冷启动。
#   0) 确保 e5 在线；冒烟 2 步（4 题），核对客户端算的可训练 token 数和服务端 trained_tokens 一致
#   1) 三档学习率各 30 步（题池、种子与 10-28 探针相同，可用 probe_diff 逐步对照），终点按榜单协议评测
#   2) 选档：没被 abort、末段格式 ≥0.95、生成长度没漂移（末 10 步 / 前 5 步在 0.6–1.6 之间），取七集最高；
#      都不过关就用 2e-5 + --neg_scale 0.5 + --kl_coef 0.05
#   3) 长跑：每步 32 题、64 步（合并题池约 1 个 epoch），每 16 步导出 adapter，评 s32/s48/s64
#
# 用法（服务器包根目录，screen 里）：
#   bash scripts/chain_drgrpo.sh 2>&1 | tee -a outputs/drgrpo-chain.log
# 环境变量：MICRO_BSZ（默认 4，服务端 --micro-bsz）  SKIP_SWEEP=1 跳过 1)，直接用 PHASE2_LR / PHASE2_EXTRA 长跑
# TUNE_BIN 在 .env 里，换新服务端只改 .env
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f .env ] && { set -a; . ./.env; set +a; }
MICRO_BSZ=${MICRO_BSZ:-4}
LRS=${LRS:-"1e-5 2e-5 4e-5"}
DR="--algorithm dr_grpo --warmup_steps 4 --abort_fmt 0.9"
SRV="--micro-bsz $MICRO_BSZ"
RETRIEVER=http://127.0.0.1:8002
POOL_A=outputs/filter/sft2-s100-e5t3/filtered.parquet   # r2 和探针的题池（1189）
POOL_B=outputs/filter/sft2-e5t3/filtered.parquet        # 第一段 RL 的题池（734）
MERGED=outputs/filter/drgrpo-merged/filtered.parquet
SEL=outputs/drgrpo-select.json
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }

# ---------- 0) e5 + 冒烟 ----------
if ! curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
     -d '{"queries":["ping"],"topk":1}' >/dev/null; then
  say "起 e5（screen e5）"
  screen -dmS e5 bash -c "cd $PWD && $PY retriever/e5_server.py --index_path /mnt/nvme0n1/no22/e5-index/e5_Flat.index \
    --store_dir data/retrieval/wiki-18-store --host 0.0.0.0 --port 8002 --backend torch 2>&1 | tee -a e5-flat.log"
  for _ in $(seq 1 60); do
    sleep 10
    curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
      -d '{"queries":["ping"],"topk":1}' >/dev/null && break
  done
fi
curl -s -m 8 -X POST "$RETRIEVER/retrieve" -H 'Content-Type: application/json' \
  -d '{"queries":["ping"],"topk":1}' >/dev/null || die "e5 没起来，看 e5-flat.log"
say "e5 在线；服务端 $TUNE_BIN（$($TUNE_BIN --version 2>&1 | head -1)），micro-bsz $MICRO_BSZ"

SMOKE=dr-smoke-$(date +%H%M)
NAME=$SMOKE LR=2e-5 STEPS=2 EVAL=0 SRV_EXTRA="$SRV" CLIENT_EXTRA="$DR --questions 4" \
  bash scripts/probe.sh 2>&1 | tee -a "outputs/probe-$SMOKE.log"
$PY - "outputs/rl/probe-$SMOKE/metrics.jsonl" <<'PYEOF' || die "冒烟没过"
import json, sys
ms = [json.loads(l) for l in open(sys.argv[1])]
assert len(ms) == 2, f"只跑了 {len(ms)} 步"
for m in ms:
    tr, dr = m.get("train") or {}, m.get("dr_grpo") or {}
    if not tr:
        print(f"step {m['step']}: 没有有效组，跳过核对"); continue
    n, t = dr["masked_tokens"], tr["trained_tokens"]
    print(f"step {m['step']}: masked {n} trained {t} scale {dr['scale']} loss {tr['loss']} gnorm {tr['grad_norm']} "
          f"秒 {m['seconds']}")
    assert abs(n - t) <= 0.02 * t, "客户端 token 数和服务端 trained_tokens 对不上，Dr.GRPO 分母会偏"
PYEOF
say "冒烟通过"

# ---------- 1) 学习率扫描 ----------
if [ "${SKIP_SWEEP:-0}" != 1 ]; then
  for LR in $LRS; do
    N=dr-lr$LR
    say "扫描 $N"
    NAME=$N LR=$LR STEPS=30 DATA=$POOL_A SRV_EXTRA="$SRV" CLIENT_EXTRA="$DR" \
      bash scripts/probe.sh 2>&1 | tee -a "outputs/probe-$N.log"
  done

  # ---------- 2) 选档 ----------
  $PY - "$SEL" $LRS <<'PYEOF'
import json, os, sys
import numpy as np
out, lrs = sys.argv[1], sys.argv[2:]
res = []
for lr in lrs:
    run, rep = f"outputs/rl/probe-dr-lr{lr}", f"outputs/upstream-eval/probe-dr-lr{lr}-s30-greedy-t3/report.json"
    ms = [json.loads(l) for l in open(f"{run}/metrics.jsonl")] if os.path.exists(f"{run}/metrics.jsonl") else []
    r = {"lr": lr, "steps": len(ms)}
    if len(ms) >= 30 and os.path.exists(rep):
        R = json.load(open(rep))["report"]
        sets = [k for k in R if k != "all"]
        gen0 = np.mean([m["avg_gen_tokens"] for m in ms[:5]])
        r.update(avg7=round(float(np.mean([R[s]["acc@1(EM)"] for s in sets])), 4),
                 fmt_tail=round(float(np.mean([m["format_ok"] for m in ms[-5:]])), 3),
                 gen_ratio=round(float(np.mean([m["avg_gen_tokens"] for m in ms[-10:]]) / gen0), 3))
        r["ok"] = r["fmt_tail"] >= 0.95 and 0.6 <= r["gen_ratio"] <= 1.6
    else:
        r["ok"] = False
    res.append(r)
    print(r)
good = [r for r in res if r["ok"]]
best = max(good, key=lambda r: r["avg7"]) if good else None
sel = {"lr": best["lr"], "extra": ""} if best else {"lr": "2e-5", "extra": "--neg_scale 0.5 --kl_coef 0.05"}
json.dump({"sweep": res, "phase2": sel}, open(out, "w"), ensure_ascii=False, indent=2)
print("phase2:", sel)
PYEOF
  [ -e "$SEL" ] || die "选档失败"
  PHASE2_LR=$($PY -c "import json;print(json.load(open('$SEL'))['phase2']['lr'])")
  PHASE2_EXTRA=$($PY -c "import json;print(json.load(open('$SEL'))['phase2']['extra'])")
fi
PHASE2_LR=${PHASE2_LR:?SKIP_SWEEP=1 时要给 PHASE2_LR}
PHASE2_EXTRA=${PHASE2_EXTRA:-}

# ---------- 3) 每步 32 题长跑 ----------
mkdir -p "$(dirname "$MERGED")"
$PY - "$POOL_A" "$POOL_B" "$MERGED" <<'PYEOF'
import sys
import pandas as pd
a, b = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
m = pd.concat([a, b]).drop_duplicates(subset=["id", "data_source"]).reset_index(drop=True)
m.to_parquet(sys.argv[3])
print(f"合并题池 {len(a)} + {len(b)} -> {len(m)}")
PYEOF
N=dr-q32-lr$PHASE2_LR
say "长跑 $N extra='$PHASE2_EXTRA'"
NAME=$N LR=$PHASE2_LR STEPS=64 DATA=$MERGED CACHE=640 CACHE_MB=10240 SRV_EXTRA="$SRV" \
  CLIENT_EXTRA="$DR --questions 32 --save_every 16 $PHASE2_EXTRA" \
  bash scripts/probe.sh 2>&1 | tee -a "outputs/probe-$N.log"
for S in 32 48; do
  CK=$(ls -dt rl-out/probe/$N/*_1-step-$S 2>/dev/null | head -1)
  [ -n "$CK" ] && [ -e "$CK/adapter.pth" ] && bash scripts/eval_adapter.sh "probe-$N-s$S" "$CK/adapter.pth"
done
say "CHAIN DONE"
