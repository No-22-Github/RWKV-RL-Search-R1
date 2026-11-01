"""把在线 RL 检查点改造成 rwkv_miss_tune 能 --resume 的离线检查点，用来从 RL 策略接着做 SFT。

rwkv_miss_tune 没有「从已有 adapter 初始化」的参数，--resume 又要求 checkpoint.json 的 config 和本次
命令行逐字段一致（含数据 sha256、planned_steps）。做法：
1. 先用本次的完整参数加 --save-every 1 跑出 checkpoint-1，当模板（config 由训练器自己算，必然一致）；
2. 张量名和形状两边相同（MiSS 的 D：master/gradient/adam_m/adam_v，外加 initial_state），
   master 换成 RL 检查点的，gradient 和 Adam 状态清零，initial_state 用模板的；
3. step=0、epoch=1、row=0，tensor_digest 重算（就是 training.pth 的 sha256）。

用法：python scripts/make_resume_ckpt.py --template <run>/checkpoint-1 \
        --source rl-out/server/<id>-step-100/training --out <新目录>
"""

import argparse
import hashlib
import json
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="同一组参数跑出来的 checkpoint-1")
    p.add_argument("--source", required=True, help="RL 检查点的 training 目录")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    tpl = torch.load(os.path.join(a.template, "training.pth"), map_location="cpu")
    src = torch.load(os.path.join(a.source, "training.pth"), map_location="cpu")
    assert set(tpl) == set(src), "两边张量名不一致"
    out = {}
    for k, t in tpl.items():
        assert src[k].shape == t.shape and src[k].dtype == t.dtype == torch.float32, k
        if k.endswith(".master"):
            out[k] = src[k].contiguous()
        elif k == "initial_state":
            out[k] = t.contiguous()
        else:
            out[k] = torch.zeros_like(t)
    os.makedirs(a.out, exist_ok=False)
    path = os.path.join(a.out, "training.pth")
    torch.save(out, path)

    m = json.load(open(os.path.join(a.template, "checkpoint.json")))
    m.update(step=0, epoch=1, row=0, tensor_digest=hashlib.sha256(open(path, "rb").read()).hexdigest())
    json.dump(m, open(os.path.join(a.out, "checkpoint.json"), "w"), indent=1)
    with open(os.path.join(a.out, "SOURCE"), "w") as f:
        f.write(os.path.realpath(a.source) + "\n")
    n = sum(k.endswith(".master") for k in out)
    print(f"{a.out}: {n} 个 master 来自 {a.source}，Adam 清零，step 0")


if __name__ == "__main__":
    main()
