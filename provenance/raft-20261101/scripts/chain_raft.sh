#!/usr/bin/env bash
# 11-01 拒绝采样自蒸馏（RAFT）：s100 在合并题池（1923 题）上每题采 8 次（榜单协议环境）→ 打包 → 两臂 SFT → 评测
#   A：从基础模型训，教师轨迹（SFT v2 的 1030 条）+ RAFT 轨迹，lr 3e-5→3e-6，2 epoch（同 SFT v2）
#   B：从 s100 接着训，只用 RAFT 轨迹，lr 1e-5→1e-6，1 epoch。rwkv_miss_tune 不能从 adapter 初始化，
#      用 make_resume_ckpt.py 把 RL 检查点改造成离线检查点再 --resume（先跑一步拿配置模板）
# A、B 同卡并行；都跑完后按榜单协议评测终点。前提：e5 8002、推理 8100 在线，采样已在 screen raftsample 里跑。
#
# 用法（服务器包根目录，screen 里）：bash scripts/chain_raft.sh 2>&1 | tee -a outputs/raft/chain.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
TUNE=rwkv_lighting_cuda/rwkv_miss_tune
MODEL=/mnt/nvme0n1/no22/rwkv7-g1k-2.9b-20260930-ctx25600.pth
VOCAB=rwkv_lighting_cuda/rwkv_vocab_v20230424.txt
TEACHER=/mnt/nvme0n1/no22/segments-sft2.jsonl
RL_CK=rl-out/server/d53cd0acfc854114943d3b2a4e027a68_1-step-100/training
SAMPLES=outputs/raft/s100-merged-n8
R=outputs/raft
COMMON=(--model "$MODEL" --vocab "$VOCAB" --targets all --ctx 6144 --chunk 6144 --batch-size 8 --warmup-steps 10
  --rank 16 --alpha 16)
A_ARGS=(--data $R/pack-a/train.jsonl --lr 0.00003 --lr-final 0.000003 --epochs 2)
B_ARGS=(--data $R/pack-b/train.jsonl --lr 0.00001 --lr-final 0.000001 --epochs 1)
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { say "FATAL: $*"; exit 1; }

# ---------- 0) 等采样 ----------
while screen -ls | grep -q '\.raftsample\b'; do sleep 60; done
[ -e $SAMPLES/report.json ] || die "采样没出 report.json，看 $SAMPLES.log"
$PY -c "import json;r=json.load(open('$SAMPLES/report.json'))['report']['all'];print({k:r[k] for k in r if k.startswith(('acc','pass','avg_search','end'))})"
say "采样完成"

# ---------- 1) 打包 ----------
$PY scripts/pack_raft.py $SAMPLES/samples.jsonl --teacher $TEACHER --out $R/pack-a || die "打包 A 失败"
$PY scripts/pack_raft.py $SAMPLES/samples.jsonl --out $R/pack-b || die "打包 B 失败"

# ---------- 2) B 的起点：模板一步 → 改造 ----------
B_INIT=$R/b-init
if [ ! -e $B_INIT/checkpoint.json ]; then
  rm -rf $R/b-template
  CUDA_VISIBLE_DEVICES=0 $TUNE "${COMMON[@]}" "${B_ARGS[@]}" --save-every 1 --output $R/b-template > $R/b-template.log 2>&1 &
  pid=$!
  for _ in $(seq 1 120); do [ -e $R/b-template/checkpoint-1/checkpoint.json ] && break; sleep 5; done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  [ -e $R/b-template/checkpoint-1/checkpoint.json ] || die "模板检查点没出来，看 $R/b-template.log"
  $PY scripts/make_resume_ckpt.py --template $R/b-template/checkpoint-1 --source $RL_CK --out $B_INIT || die "改造失败"
fi

# ---------- 3) 先训 B；A 训练时评测 B（e5 + 8100 已占约 63GB，两个 SFT 同时跑怕爆显存）----------
say "开训 B"
[ -e $R/sft-b/adapter-final.pth ] || CUDA_VISIBLE_DEVICES=0 $TUNE "${COMMON[@]}" "${B_ARGS[@]}" --save-every 1000 \
  --resume $B_INIT --output $R/sft-b > $R/sft-b.log 2>&1
say "B exit $?"; tail -c 300 $R/sft-b.log; echo
if [ -e $R/sft-b/adapter-final.pth ]; then
  bash scripts/eval_adapter.sh raft1-b $R/sft-b/adapter-final.pth > $R/eval-b.log 2>&1 &
fi
pe=$!
say "开训 A"
[ -e $R/sft-a/adapter-final.pth ] || CUDA_VISIBLE_DEVICES=0 $TUNE "${COMMON[@]}" "${A_ARGS[@]}" --save-every 1000 \
  --output $R/sft-a > $R/sft-a.log 2>&1
say "A exit $?"; tail -c 300 $R/sft-a.log; echo
wait $pe 2>/dev/null; cat $R/eval-b.log 2>/dev/null

# ---------- 4) 评测 A ----------
[ -e $R/sft-a/adapter-final.pth ] && bash scripts/eval_adapter.sh raft1-a $R/sft-a/adapter-final.pth
say "RAFT DONE"
