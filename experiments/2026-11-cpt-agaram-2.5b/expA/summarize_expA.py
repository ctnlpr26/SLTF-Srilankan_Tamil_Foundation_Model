"""Experiment A · summary: BPC per checkpoint and set, gains vs base, train-vs-held-out gap.

Confidence intervals: bootstrap over DOCUMENTS (excerpts of one book stay together), paired across
checkpoints (every checkpoint is scored on exactly the same text).
"""
import glob
import json
import os

import numpy as np

SET_LABEL = {"train": "Training docs (seen)", "val": "Held-out docs", "control": "Tamil Wikipedia (control)"}


def load_results(res_dir):
    res = {}
    for p in glob.glob(os.path.join(res_dir, "step*_*.json")):
        r = json.load(open(p, encoding="utf-8"))
        res[(r["step"], r["set"])] = r
    if not res:
        raise SystemExit(f"no results in {res_dir}")
    return res


def _per_doc(r):
    bits, chars = {}, {}
    for row in r["rows"]:
        bits[row["doc_id"]] = bits.get(row["doc_id"], 0.0) + row["bits"]
        chars[row["doc_id"]] = chars.get(row["doc_id"], 0) + row["chars"]
    return bits, chars


def summarize(res_dir, n_boot=2000, seed=0, steps_per_epoch=None, tokens_per_step=32768):
    res = load_results(res_dir)
    steps = sorted({s for s, _ in res})
    sets = [s for s in ("train", "val", "control") if any(k[1] == s for k in res)]
    rng = np.random.default_rng(seed)
    out = {"steps": steps, "sets": sets, "bpc": {}, "ci": {}, "gain": {}, "gain_ci": {}, "n_docs": {},
           "n_chars": {}, "gap": {}, "gap_ci": {}, "epoch2": {}, "per_doc": {}}
    boot_bpc = {}
    for s in sets:
        st = [k for k in steps if (k, s) in res]
        per = {k: _per_doc(res[(k, s)]) for k in st}
        docs = sorted(set.intersection(*(set(per[k][0]) for k in st)))   # docs every checkpoint scored
        chars = np.array([per[st[0]][1][d] for d in docs], dtype=float)
        idx = rng.integers(0, len(docs), size=(n_boot, len(docs)))
        out["n_docs"][s], out["n_chars"][s] = len(docs), int(chars.sum())
        for k in st:
            bits = np.array([per[k][0][d] for d in docs])
            out["bpc"][(k, s)] = float(bits.sum() / chars.sum())
            boot_bpc[(k, s)] = bits[idx].sum(1) / chars[idx].sum(1)
            out["ci"][(k, s)] = tuple(float(x) for x in np.percentile(boot_bpc[(k, s)], [2.5, 97.5]))
            out["per_doc"][(k, s)] = {d: float(per[k][0][d] / per[k][1][d]) for d in docs}
        if 0 in st:
            for k in st:
                if k == 0:
                    continue
                g = 100 * (1 - boot_bpc[(k, s)] / boot_bpc[(0, s)])          # paired: same resampled docs
                out["gain"][(k, s)] = 100 * (1 - out["bpc"][(k, s)] / out["bpc"][(0, s)])
                out["gain_ci"][(k, s)] = tuple(float(x) for x in np.percentile(g, [2.5, 97.5]))
    # train-vs-held-out gap (percentage points), train and val are independent samples
    for k in steps:
        if (k, "train") in out["gain"] and (k, "val") in out["gain"]:
            g_tr = 100 * (1 - boot_bpc[(k, "train")] / boot_bpc[(0, "train")])
            g_va = 100 * (1 - boot_bpc[(k, "val")] / boot_bpc[(0, "val")])
            out["gap"][k] = out["gain"][(k, "train")] - out["gain"][(k, "val")]
            out["gap_ci"][k] = tuple(float(x) for x in np.percentile(g_tr - g_va, [2.5, 97.5]))
    # epoch 2: change from the checkpoint closest to the end of epoch 1 to the last checkpoint
    if steps_per_epoch and len(steps) > 2:
        last = max(steps)
        e1 = min((k for k in steps if 0 < k < last), key=lambda k: abs(k - steps_per_epoch), default=None)
        if e1 is not None and last > steps_per_epoch * 1.5:
            for s in sets:
                if (e1, s) in boot_bpc and (last, s) in boot_bpc:
                    d = 100 * (1 - boot_bpc[(last, s)] / boot_bpc[(e1, s)])
                    out["epoch2"][s] = {"from": e1, "to": last,
                                        "change": 100 * (1 - out["bpc"][(last, s)] / out["bpc"][(e1, s)]),
                                        "ci": tuple(float(x) for x in np.percentile(d, [2.5, 97.5]))}
    out["steps_per_epoch"], out["tokens_per_step"] = steps_per_epoch, tokens_per_step
    return out, res


def _ci(c, fmt="{:.3f}"):
    sep = " to " if "+" in fmt or min(c) < 0 else "–"
    return f"{fmt.format(c[0])}{sep}{fmt.format(c[1])}"


def to_markdown(out, res, titles=None):
    titles = titles or {}
    spe, tps = out["steps_per_epoch"], out["tokens_per_step"]
    L = [f"# Experiment A · {titles.get('__model__', '')} · Training vs held-out next-token prediction", "",
         "Bits per character (BPC, lower is better), same windowing as E2 (window 2048, stride 1024). "
         "Gain = % fewer bits per character than the base model on the same text. "
         "95% CIs: bootstrap over documents, paired across checkpoints.", ""]
    L += ["| Set | Documents | Characters |", "|---|---|---|"]
    L += [f"| {SET_LABEL[s]} | {out['n_docs'][s]} | {out['n_chars'][s]:,} |" for s in out["sets"]]
    hdr = ["Step", "Epoch", "Tokens seen"]
    for s in out["sets"]:
        hdr += [f"{SET_LABEL[s]} BPC", "gain %"]
    if out["gap"]:
        hdr += ["Train − held-out gain (pp)"]
    L += ["", "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for k in out["steps"]:
        ep = f"{k / spe:.2f}" if spe else "—"
        row = ["base" if k == 0 else str(k), "0" if k == 0 else ep, "0" if k == 0 else f"{k * tps / 1e6:.1f}M"]
        for s in out["sets"]:
            if (k, s) not in out["bpc"]:
                row += ["—", "—"]
                continue
            row.append(f"{out['bpc'][(k, s)]:.3f} ({_ci(out['ci'][(k, s)])})")
            row.append("—" if k == 0 or (k, s) not in out["gain"] else
                       f"{out['gain'][(k, s)]:.1f} ({_ci(out['gain_ci'][(k, s)], '{:.1f}')})")
        if out["gap"]:
            row.append(f"{out['gap'][k]:+.1f} ({_ci(out['gap_ci'][k], '{:+.1f}')})" if k in out["gap"] else "—")
        L.append("| " + " | ".join(row) + " |")

    L += ["", "## What the numbers say", ""]
    last = max(out["steps"])
    if (last, "train") in out["gain"] and (last, "val") in out["gain"]:
        L.append(f"- At step {last}: {out['gain'][(last, 'train')]:.1f}% gain on training text vs "
                 f"{out['gain'][(last, 'val')]:.1f}% on held-out text "
                 f"(gap {out['gap'][last]:+.1f} pp, 95% CI {_ci(out['gap_ci'][last], '{:+.1f}')}).")
    for s, e in out["epoch2"].items():
        L.append(f"- Epoch 2 (step {e['from']} → {e['to']}), {SET_LABEL[s].lower()}: BPC changed by "
                 f"{-e['change']:+.1f}% (95% CI {-e['ci'][1]:+.1f} to {-e['ci'][0]:+.1f}).")
    if (last, "control") in out["gain"] and (last, "val") in out["gain"]:
        L.append(f"- Held-out Noolaham text improved {out['gain'][(last, 'val')]:.1f}%, Tamil Wikipedia "
                 f"{out['gain'][(last, 'control')]:.1f}% (95% CI {_ci(out['gain_ci'][(last, 'control')], '{:.1f}')}).")

    if (0, "val") in out["per_doc"] and (last, "val") in out["per_doc"]:
        L += ["", f"## Held-out documents: base vs step {last}", "",
              "| Document | Base BPC | SLTFM BPC | Gain % |", "|---|---|---|---|"]
        b, f = out["per_doc"][(0, "val")], out["per_doc"][(last, "val")]
        for d in sorted(b, key=lambda d: 1 - f[d] / b[d], reverse=True):
            name = str(titles.get(d, d)).replace("|", "\\|")[:70]
            L.append(f"| {name} | {b[d]:.3f} | {f[d]:.3f} | {100 * (1 - f[d] / b[d]):.1f} |")
    return "\n".join(L)


def plot(out, path, log_x=False, title="SLTFM CPT: training vs held-out text, by checkpoint"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"train": "#E39B2B", "val": "#2560B0", "control": "#7A869A"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in out["sets"]:
        ks = [k for k in out["steps"] if (k, s) in out["bpc"]]
        x = [max(k, 1) if log_x else k for k in ks]
        y = [out["bpc"][(k, s)] for k in ks]
        lo = [out["ci"][(k, s)][0] for k in ks]
        hi = [out["ci"][(k, s)][1] for k in ks]
        ax.plot(x, y, marker="o", color=colors[s], label=SET_LABEL[s])
        ax.fill_between(x, lo, hi, color=colors[s], alpha=0.15, linewidth=0)
    if out["steps_per_epoch"]:
        if max(out["steps"]) > out["steps_per_epoch"] * 1.05:      # only for multi-epoch runs
            ax.axvline(out["steps_per_epoch"], color="#888", linestyle="--", linewidth=1)
            ax.text(out["steps_per_epoch"], ax.get_ylim()[1], "  end of epoch 1", va="top", fontsize=9, color="#555")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("Training step (0 = base model)")
    ax.set_ylabel("Bits per character (lower is better)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def to_json(out):
    def key(k):
        return f"{k[0]}:{k[1]}" if isinstance(k, tuple) else str(k)
    return {name: ({key(k): v for k, v in val.items()} if isinstance(val, dict) else val)
            for name, val in out.items()}
