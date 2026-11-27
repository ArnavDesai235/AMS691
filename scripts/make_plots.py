"""Figures for the report, drawn from the committed C1-C4 results (no model calls).

    python scripts/make_plots.py                       # -> reports/figures/*.png
    python scripts/make_plots.py --run full_cf0.3_s42  # draw a different results dir for the tuned role

Inputs per run: results/<run>/C{1..4}_leakage.json and C{1..4}_metrics.json.
Design rules: one axis per panel (no dual axes), thin marks, categorical colours in fixed order
(validated palette), original-vs-tuned drawn as grey (context) vs blue (emphasis), text in neutral ink,
a value label on every bar (the orange/aqua steps are below 3:1 on white, so labels are required).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONFIGS = ["C1", "C2", "C3", "C4"]
INK, INK2, GRID, SURFACE = "#191919", "#666666", "#e6e6e6", "#ffffff"
BLUE, ORANGE, AQUA, GREY = "#257de0", "#f47230", "#00b894", "#b5b5b5"


def load(results: Path, run: str, kind: str, c: str) -> dict:
    return json.loads((results / run / f"{c}_{kind}.json").read_text())["summary"]


def style(ax, ylabel: str | None = None, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def bars(ax, xs, ys, width, color, label, errs=None, fmt="{:.2f}", pad=0.01):
    ax.bar(xs, ys, width=width, color=color, label=label, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    if errs:
        for x, y, e in zip(xs, ys, errs):
            if e is not None:
                ax.vlines(x, e[0], e[1], color=INK2, linewidth=1, zorder=4)
    top = max(ys) if ys else 1
    for x, y, e in zip(xs, ys, errs or [None] * len(ys)):
        ax.text(x, (e[1] if e else y) + pad * (top if top > 1 else 1), "0" if y == 0 else fmt.format(y),
                ha="center", va="bottom", fontsize=7.5, color=INK)


def rate(d: dict, key: str):
    r = d[key]
    return r["rate"], (tuple(r["ci95"]) if r.get("ci95") else None)


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, **kw)
    return leg


def fig_leakage(results: Path, run: str, out: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200, facecolor=SURFACE)
    series = [("struct_canary", "Structural canary in context", BLUE),
              ("contam_canary", "Contamination canary in context", ORANGE),
              ("answer_canary", "Canary in answer", AQUA)]
    w = 0.26
    for i, (key, label, col) in enumerate(series):
        xs, ys, es = [], [], []
        for j, c in enumerate(CONFIGS):
            y, e = rate(load(results, run, "leakage", c)["overall"], key)
            xs.append(j + (i - 1) * (w + 0.02)); ys.append(y); es.append(e)
        bars(ax, xs, ys, w, col, label, es)
    ax.set_xticks(range(4), CONFIGS)
    ax.set_ylim(0, 1.15)
    n = load(results, run, "leakage", "C3")["overall"]["n"]
    style(ax, "Share of query-user rows", f"Leakage by configuration ({tag}, cf = 0.3, n = {n} rows with forbidden canaries)")
    legend(ax, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.01, 0.01, "Error bars: 95% CI over rows. C4 is 0 on every measure.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "fig1_leakage_by_config.png")
    plt.close(fig)


def fig_tuning(results: Path, orig: str, tuned: str, out: Path) -> None:
    panels = [("recall@10", "Recall@10", "metrics", None, "{:.2f}", (0, 1.0)),
              ("latency_p95_s", "p95 latency (s)", "metrics", None, "{:.0f}", None),
              ("contam_canary", "Contamination canary in context", "leakage", "overall", "{:.2f}", (0, 1.05)),
              ("answer_canary", "Canary in answer", "leakage", "overall", "{:.2f}", (0, 1.05))]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.5), dpi=200, facecolor=SURFACE)
    w = 0.38
    for ax, (key, title, kind, sub, fmt, ylim) in zip(axes, panels):
        for k, (run, col, lab) in enumerate([(orig, GREY, "Original retrieval"), (tuned, BLUE, "Tuned retrieval")]):
            xs, ys, es = [], [], []
            for j, c in enumerate(CONFIGS):
                d = load(results, run, kind, c)["overall"]
                if kind == "metrics":
                    y, e = d[key], None
                else:
                    y, e = rate(d, key)
                xs.append(j + (k - 0.5) * (w + 0.03)); ys.append(y); es.append(e)
            bars(ax, xs, ys, w, col, lab, es, fmt=fmt, pad=0.02)
        ax.set_xticks(range(4), CONFIGS)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(load(results, orig, "metrics", c)["overall"][key] for c in CONFIGS) * 1.18)
        style(ax, None, title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Effect of tuned retrieval (cf = 0.3, 1120 rows; leakage over the 840 rows with forbidden canaries)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(out / "fig2_original_vs_tuned.png")
    plt.close(fig)


def fig_hub(results: Path, run: str, out: Path, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), dpi=200, facecolor=SURFACE, sharey=True)
    w = 0.36
    for ax, (key, title) in zip(axes, [("contam_canary", "Contamination canary in context"),
                                       ("answer_canary", "Canary in answer")]):
        for k, (reg, col, lab) in enumerate([("hub", BLUE, "Hub (>= 8 source chunks)"),
                                             ("non_hub", ORANGE, "Non-hub (< 8)")]):
            xs, ys, es = [], [], []
            for j, c in enumerate(CONFIGS):
                d = load(results, run, "leakage", c)["by_regime"][reg]
                y, e = rate(d, key)
                xs.append(j + (k - 0.5) * (w + 0.03)); ys.append(y); es.append(e)
            bars(ax, xs, ys, w, col, f"{lab}", es, fmt="{:.2f}")
        ax.set_xticks(range(4), CONFIGS)
        ax.set_ylim(0, 1.1)
        style(ax, None, title)
    axes[0].set_ylabel("Share of query-user rows", color=INK2, fontsize=9)
    n = {r: load(results, run, "leakage", "C3")["by_regime"][r]["n"] for r in ("hub", "non_hub")}
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, fontsize=8.5,
               labelcolor=INK2)
    fig.suptitle(f"By realized source_id regime ({tag}; hub n = {n['hub']}, non-hub n = {n['non_hub']} rows)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.text(0.99, 0.01, "Error bars: 95% CI. Hub sample is small and not controlled for entity or query type.",
             fontsize=7, color=INK2, ha="right")
    fig.tight_layout(rect=(0, 0.13, 1, 0.94))
    fig.savefig(out / "fig3_hub_vs_nonhub.png")
    plt.close(fig)


def fig_unauth(results: Path, run: str, out: Path, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200, facecolor=SURFACE)
    w = 0.36
    for k, (key, col, lab) in enumerate([("answer_unauthorized_fact_rate", GREY, "Raw (substring match)"),
                                         ("answer_unauthorized_fact_rate_adj", BLUE,
                                          "Adjusted (value not stated in user's readable docs)")]):
        xs = [j + (k - 0.5) * (w + 0.03) for j in range(4)]
        ys = [load(results, run, "metrics", c)["overall"][key] for c in CONFIGS]
        bars(ax, xs, ys, w, col, lab, None, fmt="{:.3f}", pad=0.02)
    ax.set_xticks(range(4), CONFIGS)
    ax.set_ylim(0, 0.52)
    style(ax, "Share of rows (n = 1120)", f"Restricted fact repeated in the answer ({tag})")
    legend(ax, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig4_unauthorized_fact_raw_vs_adjusted.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--original", default="full_cf0.3_s42")
    ap.add_argument("--tuned", default="full_cf0.3_s42_tuned")
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_leakage(results, a.tuned, out, "tuned retrieval")
    fig_tuning(results, a.original, a.tuned, out)
    fig_hub(results, a.tuned, out, "tuned retrieval")
    fig_unauth(results, a.tuned, out, "tuned retrieval")
    print("wrote", *sorted(p.name for p in out.glob("fig*.png")))


if __name__ == "__main__":
    main()
