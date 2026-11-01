#!/usr/bin/env bash
# 11-01 RAFT 第二轮：用第一轮最好的 A（raft1-a，七集 0.407）当采样策略，题池扩大，仍从基础模型训
#   0) 题池：drgrpo-merged（1923）+ 新抽 NQ/HotpotQA 各 NEW_PER 题（train_rl_sft2_r2 去掉两轮筛题用过的 4800 题和合并池）
#   1) A 每题采 8 次（榜单协议环境，T=1）
#   2) 打包：教师 1030 条 + A 答对的轨迹（全对的简单题只留 EASY_KEEP）
#   3) 从基础模型 SFT，参数同第一轮 A（lr 3e-5→3e-6，2 epoch）
#   4) 榜单协议评测
# 前提：e5 8002、推理 8100 在线，8848 没有服务。
# 用法（服务器包根目录，screen 里）：bash scripts/chain_raft2.sh 2>&1 | tee -a outputs/raft2/chain.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TUNE=rwkv_lighting_cuda/rwkv_miss_tune
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
VOCAB=rwkv_lighting_cuda/rwkv_vocab_v20230424.txt
TEACHER=/mnt/nvme0n1/no22/segments-sft2.jsonl
POLICY=outputs/raft/sft-a/adapter-final.pth
NEW_PER=${NEW_PER:-1250}
EASY_KEEP=${EASY_KEEP:-0.5}
R=outputs/raft2
POOL=$R/pool.parquet
SAMPLES=$R/a-pool-n8
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }
mkdir -p $R

# ---------- 0) 题池 ----------
if [ ! -e $POOL ]; then
  $PY - $POOL $NEW_PER <<'PYEOF' || die "建题池失败"
import json, sys
import pandas as pd
out, per = sys.argv[1], int(sys.argv[2])
src = pd.read_parquet("data/search_r1/train_rl_sft2_r2.parquet")
merged = pd.read_parquet("outputs/filter/drgrpo-merged/filtered.parquet")
seen = {(str(i), s) for i, s in zip(merged.id, merged.data_source)}
for d in ["sft2-e5t3", "sft2-s100-e5t3"]:
    for line in open(f"outputs/filter/{d}/questions.jsonl"):
        q = json.loads(line)
        seen.add((str(q["qid"]), q["source"]))
left = src[[(str(i), s) not in seen for i, s in zip(src.id, src.data_source)]]
new = pd.concat([left[left.data_source == s].sample(per, random_state=1101) for s in ("nq", "hotpotqa")])
pool = pd.concat([merged, new]).reset_index(drop=True)
assert not pool.duplicated(subset=["id", "data_source"]).any()
pool.to_parquet(out)
print(f"题池：合并池 {len(merged)} + 新题 {len(new)}（每源 {per}，random_state 1101）= {len(pool)}")
PYEOF
fi

# ---------- 1) 采样 ----------
if [ ! -e $SAMPLES/report.json ]; then
  curl -sS -X POST http://127.0.0.1:8100/v1/adapters -H 'Content-Type: application/json' \
    --data "{\"adapter_id\":\"raft1-a\",\"path\":\"$(realpath $POLICY)\"}"; echo
  say "开始采样 -> $SAMPLES"
  $PY scripts/eval_base.py --data $POOL --format g1k --think full \
    --max_turns 4 --final_answer_turn --max_obs_tokens 500 --max_turn_tokens 2048 --topk 3 --n 8 --temperature 1.0 \
    --api http://127.0.0.1:8100 --retriever http://127.0.0.1:8002 --batch_size 16 --workers 64 \
    --adapter_id raft1-a --out $SAMPLES > $SAMPLES.log 2>&1 || die "采样失败，看 $SAMPLES.log"
fi
$PY -c "import json;R=json.load(open('$SAMPLES/report.json'))['report']
for k,v in R.items(): print(k,{x:v[x] for x in v if x.startswith(('acc@1','pass@8(EM)','avg_search'))})"
say "采样完成"

# ---------- 2) 打包 ----------
$PY scripts/pack_raft.py $SAMPLES/samples.jsonl --teacher $TEACHER --easy_keep $EASY_KEEP --out $R/pack || die "打包失败"

# ---------- 3) SFT ----------
say "开训"
[ -e $R/sft/adapter-final.pth ] || CUDA_VISIBLE_DEVICES=0 $TUNE --model $MODEL --vocab $VOCAB --targets all \
  --ctx 6144 --chunk 6144 --batch-size 8 --warmup-steps 10 --rank 16 --alpha 16 \
  --data $R/pack/train.jsonl --lr 0.00003 --lr-final 0.000003 --epochs 2 --save-every 100000 \
  --output $R/sft > $R/sft.log 2>&1
say "训练 exit $?"; tail -c 300 $R/sft.log; echo
[ -e $R/sft/adapter-final.pth ] || die "没有 adapter-final.pth"

# ---------- 4) 评测 ----------
bash scripts/eval_adapter.sh raft2 $R/sft/adapter-final.pth
say "RAFT2 DONE"
