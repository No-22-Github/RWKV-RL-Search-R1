"""Search-R1 RL 全线折线图，只读仓库里的文件（provenance 快照 + outputs/upstream-eval 的 report.json），本机就能复现。

  fig1_eval_lineage.png    两条路线的榜单协议七集平均，横轴是累计 RL 步数
  fig2_train_lineage.png   各段 RL 的训练指标（acc、生成长度、检索次数、格式正确率），左右两列各一条路线
  fig3_probe_dynamics.png  10-28 夜间调参探针前 30 步，和 r2 同题序对照（含 acc - 期望 acc）
  fig4_leaderboard.png     和 SearchAgent 榜单各家的七集平均对比（榜单原始 json 在 leaderboard/）

路线一：SFT v1 → v1（0–100）→ v2（从 v1-s100 的 adapter 起、服务端步数重新计，累计 100–352）
        → v3b（从 v2 的 200 续到 311，累计 300–411）→ v4（从 300 续到 416，累计 400–516）
路线二：SFT v2 → 第一段 RL（0–100）→ r2（从 100 续到 200）；探针都从 s100 起

用法：python scripts/plot_rl.py [--root provenance/rl-lineage-20261029] [--evals outputs/upstream-eval]
依赖 matplotlib、numpy（.venv 里都有）。
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# 配色：dataviz 参考调色板的分类色（浅色模式，按固定顺序取，已过色盲校验），文字用中性墨色
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CJK_FONTS = ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]

# 各段 RL：(路线, 目录, 图例名, 颜色, 累计步数偏移)。横轴 = 服务端步数 + 偏移；
# v2 换了服务端、步数从 0 重计，所以路线一 v2 以后都加上 v1 的 100 步
LINEAGE_RUNS = [
    (0, "sr1-v1", "v1 · BM25 top-3", BLUE, 0),
    (0, "sr1-v2", "v2 · BM25 top-3，从 v1-s100 起", ORANGE, 100),
    (0, "sr1-v3b", "v3b · 多跳题池，从 v2-s200 续", AQUA, 100),
    (0, "sr1-v4", "v4 · e5 top-5，从 v3b-s300 续", YELLOW, 100),
    (1, "sft2-grpo", "第一段 · e5 top-3 截 500", BLUE, 0),
    (1, "sft2-grpo-r2", "r2 · 扩池续训，从 s100 续", ORANGE, 0),
]
ROUTE_NAMES = ["路线一：SFT v1 → v1 → v2 → v3b → v4", "路线二：SFT v2 → RL → r2"]
PROBE_RUNS = [
    ("sft2-grpo-r2", "r2 参照 · lr 2e-6", BLUE, "o"),
    ("probe-lr1e-5-collapsed", "A · lr 1e-5 冷启动", ORANGE, "s"),
    ("probe-lr5e-6-wu8", "B · lr 5e-6 预热 8 步", AQUA, "^"),
    ("probe-lr5e-6-seqnorm-neg05", "C · B + 轨迹等权 + 负优势×0.5", YELLOW, "D"),
]
# (累计步数, report 目录, 标注)
ROUTE1 = [(0, "lineage-sr1v1-r16-lr5e-5-e2-greedy-t3", "SFT v1"), (100, "lineage-rl-sr1-v1-s100-greedy-t3", "v1-s100"),
          (300, "lineage-rl-sr1-v2-s200-greedy-t3", "v2-s200"), (400, "v3b-s300-greedy-t3", "v3b-s300"),
          (450, "v4-s50-greedy-t3", "v4-s50"), (500, "v4-s100-greedy-t3", "v4-s100"),
          (510, "v4-s110-greedy-t3", "v4-s110")]
LABEL_OFFSETS = {"v4-s110": (16, -26), "s170": (8, 10)}  # 和相邻点的标注错开
ROUTE2 = [(0, "sft2-r16-lr3e-5-e2-greedy-t3", "SFT v2"), (50, "sft2-rl-s50-greedy-t3", "s50"),
          (100, "sft2-rl-s100-greedy-t3", "s100"), (150, "sft2r2-s150-greedy-t3", "s150"),
          (170, "sft2r2-s170-greedy-t3", "s170"), (200, "sft2r2-s200-greedy-t3", "s200")]
BASE = "lineage-base-greedy-t3"
PAPER_7B_PPO = 0.431  # Search-R1 论文 7B Base PPO 的同七集平均
# 榜单对比：只取检索类方法和 RAG 基线；七集 = 去掉 FictionalHot
LB_METHODS = {"ReSeek", "ZeroSearch", "Search-R1", "RAG"}
LB_SETS = ["nq", "triviaqa", "popqa", "hotpotqa", "2wiki", "musique", "bamboogle"]
OURS_BEST = [("v4-s100-greedy-t3", "v4-s100"), ("sft2r2-s150-greedy-t3", "r2-s150")]


def setup_style():
    for path in CJK_FONTS:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def rolling(y, w):
    """只用已有的点做滑动平均（开头不足 w 步就用前几步），短的段不会被截掉"""
    return np.array([y[max(0, i - w + 1): i + 1].mean() for i in range(len(y))])


def load_run(root, name):
    d = os.path.join(root, "runs", name)
    rows = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl"))]
    out = {k: np.array(v, dtype=float) for k, v in {
        "step": [m["step"] for m in rows],
        # 服务端步数：这一步 rollout 用的策略版本 = 训练后发布的步数 - 1（旧版客户端没记 publish 就用 step）
        "srv": [(m.get("publish") or {}).get("step", m["step"] + 1) - 1 for m in rows],
        "acc": [m["acc"] for m in rows],
        "gen": [m["avg_gen_tokens"] for m in rows],
        "fmt": [m["format_ok"] for m in rows],
        "search": [m["avg_search"] for m in rows],
    }.items()}
    exp_path = os.path.join(d, "expected.jsonl")
    if os.path.exists(exp_path):
        exp = {r["step"]: r["exp"] for r in map(json.loads, open(exp_path))}
        out["diff"] = rolling(np.array([m["acc"] - exp[m["step"]] for m in rows]), 5)
    return out


def eval_avg(evals, name):
    rep = json.load(open(os.path.join(evals, name, "report.json")))["report"]
    sets = [k for k in rep if k != "all"]
    return float(np.mean([rep[s]["acc@1(EM)"] for s in sets]))


def finish_panels(fig, axes, handles, labels, suptitle, path, ncol):
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol, frameon=False)
    fig.suptitle(suptitle, color=INK, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("写出", path)


def draw_eval_lineage(evals, path):
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for y, text in [(0.45, "目标 0.45"), (PAPER_7B_PPO, f"Search-R1 论文 7B Base PPO {PAPER_7B_PPO:.3f}")]:
        ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(522, y + 0.002, text, color=INK2, va="bottom", ha="right", fontsize=9)
    for route, color, marker, label, dy in [
        (ROUTE1, BLUE, "o", ROUTE_NAMES[0], 10),
        (ROUTE2, ORANGE, "s", ROUTE_NAMES[1], -24),
    ]:
        xs = [x for x, _, _ in route]
        ys = [eval_avg(evals, n) for _, n, _ in route]
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
        for x, y, (_, _, lab) in zip(xs, ys, route):
            ax.annotate(f"{lab}\n{y:.3f}", (x, y), textcoords="offset points", xytext=LABEL_OFFSETS.get(lab, (0, dy)),
                        ha="center", fontsize=8.5, color=INK)
    ax.set_xlim(-15, 525)
    ax.set_ylim(0.30, 0.47)
    ax.set_xlabel("累计 RL 步数（从各自的 SFT 起点算）")
    ax.set_title(f"榜单协议七集抽样平均 EM（1925 题，贪心；七集平均误差约 ±1.2 点；基础模型 {eval_avg(evals, BASE):.3f}）",
                 color=INK)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("写出", path)


def draw_train_lineage(runs, path):
    panels = [("acc", "训练 acc"), ("gen", "平均生成长度（token）"), ("search", "平均检索次数"), ("fmt", "格式正确率")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 11), sharey="row", constrained_layout=True)
    for col, route_name in enumerate(ROUTE_NAMES):
        for row, (key, title) in enumerate(panels):
            ax = axes[row, col]
            for route, name, label, color, offset in LINEAGE_RUNS:
                if route != col or name not in runs:
                    continue
                r = runs[name]
                keep = np.ones(len(r["srv"]), bool)
                if name == "sft2-grpo":  # 第一段在 100 步手动停，之后的两步被 r2 从 step-100 续训覆盖
                    keep = r["srv"] < 100
                ax.plot(r["srv"][keep] + offset, rolling(r[key][keep], 10), color=color, linewidth=2, label=label,
                        zorder=3)
            ax.set_title(f"{route_name}\n{title}" if row == 0 else title, color=INK)
            ax.set_xlim(-5, 520 if col == 0 else 205)
            if row == 0:  # 放在各列线条空出来的角上
                ax.legend(loc="lower left", bbox_to_anchor=(0.22 if col == 0 else 0, 0), frameon=False, fontsize=9)
            if row == len(panels) - 1:
                ax.set_xlabel("累计 RL 步数（从 SFT 起点算）")
    fig.suptitle("各段 RL 的训练指标（10 步滑动平均；各段题池和检索设置不同，acc 跨段不可比）",
                 color=INK, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("写出", path)


def draw_probe_dynamics(runs, path, max_step=30):
    panels = [("gen", "平均生成长度（token）", None), ("fmt", "格式正确率", None),
              ("diff", "acc - 期望 acc（5 步滑动平均）", 0.05), ("search", "平均检索次数", 0.5)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (key, title, tick) in zip(axes.flat, panels):
        for name, label, color, marker in PROBE_RUNS:
            r = runs[name]
            keep = r["step"] < max_step
            x, y = r["step"][keep], r[key][keep]
            ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=4, markeredgecolor=SURFACE,
                    markeredgewidth=1, label=label, zorder=3)
            if len(x) < max_step:  # 中途停训的探针在末点标 ×
                ax.plot(x[-1], y[-1], marker="X", markersize=10, color=color, markeredgecolor=SURFACE, zorder=4)
        if key == "diff":
            ax.axhline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
        if key == "fmt":
            ax.axhline(0.9, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
            ax.text(max_step - 0.5, 0.905, "停训线 0.9", ha="right", va="bottom", color=INK2, fontsize=9)
        if tick:
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(tick))
        ax.set_title(title, color=INK)
        ax.set_xlim(-0.5, max_step - 0.5)
        ax.set_xlabel("训练步（第 0 步 = 路线二 s100）")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    finish_panels(fig, axes, handles, labels, "10-28 夜间 GRPO 调参探针前 30 步（都从路线二 s100 起，同 seed 同题序）",
                  path, ncol=2)


def draw_leaderboard(evals, lb_dir, path):
    """七集平均对比：榜单各家（leaderboard/ 下从 Space 拉的原始 json）+ 我们两条路线的最好检查点"""
    rows = []
    for f in sorted(os.listdir(lb_dir)):
        method, model = f[:-5].split("_", 1)
        if method not in LB_METHODS:
            continue
        res = json.load(open(os.path.join(lb_dir, f)))["results"]
        size = model.split("-")[1].upper()
        rows.append((f"{method} · Qwen2.5-{size}", np.mean([res[k]["exact_match"] for k in LB_SETS]), False))
    rows += [(f"我们 {lab} · RWKV-7 2.9B", eval_avg(evals, name), True) for name, lab in OURS_BEST]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    ys = np.arange(len(rows))
    ax.barh(ys, [v for _, v, _ in rows], height=0.62, color=[BLUE if ours else "#b9b8b3" for *_, ours in rows], zorder=3)
    for y, (label, v, ours) in zip(ys, rows):
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9, color=INK, fontweight="bold" if ours else None)
    ax.set_yticks(ys, [label for label, *_ in rows])
    for tick, (*_, ours) in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if ours else INK2)
        tick.set_fontweight("bold" if ours else "normal")
    ax.set_xlim(0, 0.46)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("和 SearchAgent 榜单比：七集平均 EM", color=INK)
    ax.set_xlabel("不含 FictionalHot；榜单是完整测试集，我们每集抽 300 题（七集平均误差约 ±1.2 点）")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("写出", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="provenance/rl-lineage-20261029")
    p.add_argument("--evals", default="outputs/upstream-eval")
    a = p.parse_args()
    setup_style()
    names = {r[1] for r in LINEAGE_RUNS} | {r[0] for r in PROBE_RUNS}
    # v1 的指标在另一台机器上，没拉回来时只画评测点
    runs = {n: load_run(a.root, n) for n in names if os.path.exists(os.path.join(a.root, "runs", n, "metrics.jsonl"))}
    out = os.path.join(a.root, "figures")
    os.makedirs(out, exist_ok=True)
    draw_eval_lineage(a.evals, os.path.join(out, "fig1_eval_lineage.png"))
    draw_train_lineage(runs, os.path.join(out, "fig2_train_lineage.png"))
    draw_probe_dynamics(runs, os.path.join(out, "fig3_probe_dynamics.png"))
    draw_leaderboard(a.evals, os.path.join(a.root, "leaderboard"), os.path.join(out, "fig4_leaderboard.png"))


if __name__ == "__main__":
    main()
