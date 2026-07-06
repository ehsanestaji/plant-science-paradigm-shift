"""
Paper A GPU-enriched figures — publication-quality.

Reads analysis results from results/paper_a/ and produces figures in
results/paper_a/figures/{main,supplementary}/ as PDF + PNG (300 DPI).

Usage:
    python -m src.viz.paper_a_gpu_figures [--results-dir results/paper_a] [--out-dir results/paper_a/figures]

Main figures (M1–M8) and supplementary figures (S1, S2, S3, S6, S9, S10).
Skips gracefully when an input file is missing.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Nature Plants color palette — organism-specific
ORGANISM_COLORS = {
    "arabidopsis":            "#e41a1c",
    "rice":                   "#377eb8",
    "wheat":                  "#ff7f00",
    "maize":                  "#4daf4a",
    "soybean":                "#984ea3",
    "tomato":                 "#a65628",
    "barley":                 "#f781bf",
    "cotton":                 "#999999",
    "potato":                 "#66c2a5",
    "tobacco":                "#8da0cb",
    "other_crop":             "#a6d854",
    "other_model_organism":   "#ffd92f",
    "non_specific":           "#cccccc",
}

TOP6_ORGANISMS = ["arabidopsis", "rice", "wheat", "maize", "soybean", "tomato"]
TOP5_ORGANISMS = ["arabidopsis", "rice", "wheat", "maize", "soybean"]

FALLBACK_PALETTE = sns.color_palette("Set2", 13)


def _org_color(organism):
    """Return color for an organism, falling back to a palette entry."""
    key = str(organism).lower().strip()
    if key in ORGANISM_COLORS:
        return ORGANISM_COLORS[key]
    idx = abs(hash(key)) % len(FALLBACK_PALETTE)
    return FALLBACK_PALETTE[idx]


# ── Helpers ───────────────────────────────────────────────────────────

def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}", flush=True)


def _load(path, label=None):
    if not os.path.exists(path):
        tag = label or os.path.basename(path)
        print(f"  SKIP: {tag} not found ({path})", flush=True)
        return None
    return pd.read_csv(path)


def _fmt_k(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K" if x >= 1000 else f"{x:.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ══════════════════════════════════════════════════════════════════════
# MAIN FIGURES
# ══════════════════════════════════════════════════════════════════════

# ── M1: Stacked area — organism share 1990–2024 ────────────────────────

def fig_m1(res_dir, out_dir):
    """M1_organism_share_timeseries — stacked area chart of organism share."""
    print("M1: Organism share timeseries...", flush=True)
    df = _load(f"{res_dir}/main/organism_timeseries.csv", "organism_timeseries.csv")
    if df is None:
        return

    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)].copy()
    df["organism"] = df["organism"].str.lower().str.strip()

    # Exclude non_specific for clarity
    df = df[df["organism"] != "non_specific"]

    # Pivot to wide form
    pivot = df.pivot_table(index="year", columns="organism", values="share_pct",
                           aggfunc="sum", fill_value=0)
    # Order columns by total share descending
    col_order = pivot.sum().sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]

    colors = [_org_color(c) for c in col_order]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.stackplot(pivot.index, pivot.values.T, labels=col_order,
                 colors=colors, alpha=0.85)

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of plant science papers (%)")
    ax.set_title("Organism Share in Plant Science (1990–2024)\n"
                 "Excluding non-specific papers")
    ax.set_xlim(1990, 2024)
    ax.set_ylim(0, None)

    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left",
              fontsize=7, ncol=2, framealpha=0.8)
    fig.tight_layout()
    _save(fig, "M1_organism_share_timeseries", out_dir)


# ── M2: Arabidopsis vs crops — line chart ─────────────────────────────

def fig_m2(res_dir, out_dir):
    """M2_arabidopsis_vs_crops — top 5 organisms as lines, crossover annotations."""
    print("M2: Arabidopsis vs crops...", flush=True)
    df = _load(f"{res_dir}/main/organism_timeseries.csv", "organism_timeseries.csv")
    if df is None:
        return

    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)].copy()
    df["organism"] = df["organism"].str.lower().str.strip()

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    plotted = {}
    for org in TOP5_ORGANISMS:
        sub = df[df["organism"] == org].sort_values("year")
        if sub.empty:
            continue
        lw = 2.5 if org == "arabidopsis" else 1.8
        ls = "--" if org == "arabidopsis" else "-"
        line, = ax.plot(sub["year"], sub["share_pct"],
                        color=_org_color(org), linewidth=lw, linestyle=ls,
                        label=org.capitalize())
        plotted[org] = sub.set_index("year")["share_pct"]

    # Mark crossover: find first year where another organism overtakes arabidopsis
    if "arabidopsis" in plotted:
        arab = plotted["arabidopsis"]
        for org in TOP5_ORGANISMS[1:]:
            if org not in plotted:
                continue
            other = plotted[org]
            combined = pd.DataFrame({"arab": arab, "other": other}).dropna()
            crossover = combined[combined["other"] >= combined["arab"]]
            if not crossover.empty:
                yr = crossover.index[0]
                y_val = combined.loc[yr, "other"]
                ax.annotate(
                    f"{org.capitalize()}\novertakes ~{yr}",
                    xy=(yr, y_val),
                    xytext=(yr - 4, y_val + 1.5),
                    fontsize=7,
                    arrowprops=dict(arrowstyle="->", color="grey", lw=0.8),
                    ha="center",
                )

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of plant science papers (%)")
    ax.set_title("Arabidopsis Share Declining as Crop Species Rise (1990–2024)")
    ax.set_xlim(1990, 2024)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "M2_arabidopsis_vs_crops", out_dir)


# ── M3: Paradigm shift — applied share per organism ────────────────────

def fig_m3(res_dir, out_dir):
    """M3_paradigm_shift — applied share over time, one line per organism (top 6)."""
    print("M3: Paradigm shift...", flush=True)
    df = _load(f"{res_dir}/main/paradigm_shift_by_organism.csv",
               "paradigm_shift_by_organism.csv")
    if df is None:
        return

    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)].copy()
    df["organism"] = df["organism"].str.lower().str.strip()

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    for org in TOP6_ORGANISMS:
        sub = df[df["organism"] == org].sort_values("year")
        if sub.empty:
            continue
        lw = 2.5 if org == "arabidopsis" else 1.8
        ax.plot(sub["year"], sub["applied_share"] * 100,
                color=_org_color(org), linewidth=lw,
                label=org.capitalize())

    ax.set_xlabel("Year")
    ax.set_ylabel("Applied research share (%)")
    ax.set_title("Paradigm Shift: Applied Research Share by Organism (1990–2024)")
    ax.set_xlim(1990, 2024)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "M3_paradigm_shift", out_dir)


# ── M4: Topic landscape — 2D UMAP scatter ─────────────────────────────

def fig_m4(res_dir, out_dir, project_root):
    """M4_topic_landscape — 2D UMAP scatter colored by organism (50k subsample)."""
    print("M4: Topic landscape UMAP...", flush=True)

    umap_path = os.path.join(project_root, "results/topics/umap_2d.npy")
    ids_path  = os.path.join(project_root, "results/topics/umap_2d_work_ids.npy")
    cls_path  = os.path.join(project_root,
                             "data/processed/classifications/paper_a_organism.csv")

    for p in (umap_path, ids_path, cls_path):
        if not os.path.exists(p):
            print(f"  SKIP M4: {p} not found", flush=True)
            return

    print("  Loading UMAP coordinates...", flush=True)
    umap_xy = np.load(umap_path)
    work_ids = np.load(ids_path, allow_pickle=True).astype(str)

    print("  Loading organism classifications...", flush=True)
    cls_df = pd.read_csv(cls_path)
    cls_df["work_id"] = cls_df["work_id"].astype(str)
    cls_dict = dict(zip(cls_df["work_id"], cls_df["predicted_label"]))

    n_total = len(work_ids)
    N_SAMPLE = 50_000
    rng = np.random.default_rng(42)
    idx = rng.choice(n_total, size=min(N_SAMPLE, n_total), replace=False)
    idx.sort()

    xy_s = umap_xy[idx]
    ids_s = work_ids[idx]
    labels = [cls_dict.get(wid, "non_specific") for wid in ids_s]

    # Remove outliers (>3 IQR)
    q25, q75 = np.percentile(xy_s, [25, 75], axis=0)
    iqr = q75 - q25
    mask = (
        (xy_s[:, 0] >= q25[0] - 3 * iqr[0]) & (xy_s[:, 0] <= q75[0] + 3 * iqr[0]) &
        (xy_s[:, 1] >= q25[1] - 3 * iqr[1]) & (xy_s[:, 1] <= q75[1] + 3 * iqr[1])
    )
    xy_s = xy_s[mask]
    labels = [l for l, m in zip(labels, mask) if m]

    labels_arr = np.array(labels)
    unique_orgs = sorted(set(labels_arr))

    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Draw non_specific first (background)
    for org in ["non_specific"] + [o for o in unique_orgs if o != "non_specific"]:
        m = labels_arr == org
        if not m.any():
            continue
        alpha = 0.15 if org == "non_specific" else 0.35
        size = 0.5 if org == "non_specific" else 1.5
        ax.scatter(xy_s[m, 0], xy_s[m, 1],
                   c=_org_color(org), s=size, alpha=alpha,
                   linewidths=0, rasterized=True,
                   label=org.replace("_", " ").capitalize())

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Plant Science Topic Landscape (SPECTER2 + UMAP)\n"
                 f"n={len(xy_s):,} sampled papers, colored by organism")
    ax.set_aspect("equal", "datalim")

    # Compact legend — skip non_specific
    handles, labels_leg = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels_leg) if "non" not in l.lower()]
    if filtered:
        h_f, l_f = zip(*filtered)
        legend = ax.legend(h_f, l_f, fontsize=7, loc="upper right",
                           markerscale=6, framealpha=0.85, ncol=2)
        for lh in legend.legend_handles:
            lh.set_alpha(1.0)

    fig.tight_layout()
    _save(fig, "M4_topic_landscape", out_dir)


# ── M5: Topic evolution heatmap ────────────────────────────────────────

def fig_m5(res_dir, out_dir):
    """M5_topic_evolution_heatmap — rows=macro-themes, cols=decades, color=growth_rate."""
    print("M5: Topic evolution heatmap...", flush=True)
    df = _load(f"{res_dir}/main/topic_growth_decline.csv",
               "topic_growth_decline.csv")
    if df is None:
        return

    # Use shorter theme names: first 40 chars of 'name'
    if "name" in df.columns:
        df["theme_label"] = df["name"].str[:40]
    elif "macro_theme_id" in df.columns:
        df["theme_label"] = "Theme " + df["macro_theme_id"].astype(str)
    else:
        df["theme_label"] = df.index.astype(str)

    # Pivot: rows=theme, cols=decade
    pivot = df.pivot_table(index="theme_label", columns="decade",
                           values="growth_rate", aggfunc="mean")

    # Keep top 20 most variable rows
    if len(pivot) > 20:
        var = pivot.var(axis=1).nlargest(20).index
        pivot = pivot.loc[var]

    # Sort by mean growth
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    vmax = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 0.1)
    vmin = -vmax

    fig_h = max(5, len(pivot) * 0.35)
    fig, ax = plt.subplots(figsize=(7.2, fig_h))

    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                vmin=vmin, vmax=vmax,
                linewidths=0.3, linecolor="white",
                cbar_kws={"label": "Growth rate", "shrink": 0.7},
                annot=(len(pivot) <= 15), fmt=".2f", annot_kws={"size": 6})

    ax.set_xlabel("Decade")
    ax.set_ylabel("Macro-theme")
    ax.set_title("Topic Growth / Decline by Decade\n(top 20 most variable macro-themes)")
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    _save(fig, "M5_topic_evolution_heatmap", out_dir)


# ── M6: Organism × topic enrichment heatmap ───────────────────────────

def fig_m6(res_dir, out_dir):
    """M6_organism_topic_enrichment — rows=organisms, cols=macro-themes, color=enrichment_log2."""
    print("M6: Organism × topic enrichment...", flush=True)
    df = _load(f"{res_dir}/main/organism_topic_heatmap.csv",
               "organism_topic_heatmap.csv")
    if df is None:
        return

    themes = _load(f"{res_dir}/main/macro_themes.csv", "macro_themes.csv")

    # Build theme name map
    if themes is not None and "macro_theme_id" in themes.columns and "name" in themes.columns:
        theme_names = dict(zip(themes["macro_theme_id"], themes["name"].str[:35]))
    else:
        theme_names = {}

    df["organism"] = df["organism"].str.lower().str.strip()
    df = df[df["organism"] != "non_specific"]

    if "macro_theme_name" in df.columns:
        df["theme_label"] = df["macro_theme_name"].str[:35]
    elif "macro_theme_id" in df.columns:
        df["theme_label"] = df["macro_theme_id"].map(
            lambda x: theme_names.get(x, f"Theme {x}")[:35])
    else:
        print("  SKIP M6: cannot identify theme column", flush=True)
        return

    pivot = df.pivot_table(index="organism", columns="theme_label",
                           values="enrichment_log2", aggfunc="mean", fill_value=0)

    # Keep top 15 themes by variance
    if pivot.shape[1] > 15:
        var = pivot.var(axis=0).nlargest(15).index
        pivot = pivot[var]

    # Order organisms as per TOP6
    present = [o for o in TOP6_ORGANISMS if o in pivot.index]
    rest = [o for o in pivot.index if o not in present]
    pivot = pivot.loc[present + rest]

    vmax = max(abs(pivot.values).max(), 0.1)

    fig_h = max(4, len(pivot) * 0.6)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Enrichment (log₂)", "shrink": 0.7},
                annot=True, fmt=".2f", annot_kws={"size": 7})

    ax.set_xlabel("Macro-theme")
    ax.set_ylabel("Organism")
    ax.set_title("Organism × Topic Enrichment\n(log₂ enrichment vs. corpus baseline)")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    _save(fig, "M6_organism_topic_enrichment", out_dir)


# ── M7: Semantic drift ─────────────────────────────────────────────────

def fig_m7(res_dir, out_dir):
    """M7_semantic_drift — distance_to_arabidopsis over time per organism."""
    print("M7: Semantic drift...", flush=True)
    df = _load(f"{res_dir}/main/semantic_drift_centroids.csv",
               "semantic_drift_centroids.csv")
    if df is None:
        return

    df["organism"] = df["organism"].str.lower().str.strip()

    # Identify time column
    time_col = None
    for c in ("window_start", "year", "decade"):
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        print("  SKIP M7: no time column found", flush=True)
        return

    # Identify distance column
    dist_col = None
    for c in ("distance_to_arabidopsis", "cosine_distance", "distance", "drift"):
        if c in df.columns:
            dist_col = c
            break
    if dist_col is None:
        print(f"  SKIP M7: no distance column; columns={list(df.columns)}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    plotted_any = False
    for org in TOP6_ORGANISMS:
        if org == "arabidopsis":
            continue
        sub = df[df["organism"] == org].sort_values(time_col)
        if sub.empty:
            continue
        ax.plot(sub[time_col], sub[dist_col],
                color=_org_color(org), linewidth=1.8,
                marker="o", markersize=3,
                label=org.capitalize())
        plotted_any = True

    if not plotted_any:
        print("  SKIP M7: no data for target organisms", flush=True)
        plt.close()
        return

    ax.set_xlabel(time_col.replace("_", " ").capitalize())
    ax.set_ylabel("Semantic distance to Arabidopsis")
    ax.set_title("Semantic Drift of Crop Species from Arabidopsis")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _save(fig, "M7_semantic_drift", out_dir)


# ── M8: CRISPR adoption S-curve ────────────────────────────────────────

def fig_m8(res_dir, out_dir):
    """M8_crispr_adoption — cumulative CRISPR share S-curve per organism."""
    print("M8: CRISPR adoption...", flush=True)
    df = _load(f"{res_dir}/main/method_diffusion_by_organism.csv",
               "method_diffusion_by_organism.csv")
    if df is None:
        return

    # Identify method column
    method_col = None
    for c in ("method", "method_name", "Method"):
        if c in df.columns:
            method_col = c
            break
    if method_col is None:
        print(f"  SKIP M8: no method column; columns={list(df.columns)}", flush=True)
        return

    df[method_col] = df[method_col].str.lower().str.strip()
    crispr = df[df[method_col].str.contains("crispr", na=False)].copy()
    if crispr.empty:
        print("  SKIP M8: no CRISPR rows found", flush=True)
        return

    crispr["organism"] = crispr["organism"].str.lower().str.strip()

    # Identify share/count column
    share_col = None
    for c in ("cumulative_share", "cumulative_pct", "share", "n_papers", "count"):
        if c in crispr.columns:
            share_col = c
            break
    if share_col is None:
        print(f"  SKIP M8: no share column; columns={list(crispr.columns)}", flush=True)
        return

    # If count column, compute cumulative share per organism
    if share_col in ("n_papers", "count"):
        def _cum_share(grp):
            grp = grp.sort_values("year")
            total = grp[share_col].sum()
            if total > 0:
                grp["_cum_share"] = grp[share_col].cumsum() / total * 100
            else:
                grp["_cum_share"] = 0.0
            return grp
        crispr = crispr.groupby("organism", group_keys=False).apply(_cum_share)
        share_col = "_cum_share"

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    plotted_any = False
    for org in TOP6_ORGANISMS:
        sub = crispr[crispr["organism"] == org].sort_values("year")
        if sub.empty:
            continue
        ax.plot(sub["year"], sub[share_col],
                color=_org_color(org), linewidth=2,
                marker="o", markersize=3,
                label=org.capitalize())
        plotted_any = True

    if not plotted_any:
        # Plot any organisms present
        for i, (org, grp) in enumerate(crispr.groupby("organism")):
            if i >= 8:
                break
            grp = grp.sort_values("year")
            ax.plot(grp["year"], grp[share_col],
                    color=FALLBACK_PALETTE[i % len(FALLBACK_PALETTE)],
                    linewidth=1.8, label=org.capitalize())

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative CRISPR adoption share (%)")
    ax.set_title("CRISPR Adoption S-Curve by Organism")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "M8_crispr_adoption", out_dir)


# ══════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY FIGURES
# ══════════════════════════════════════════════════════════════════════

# ── S1: Growth rates — grouped bar CAGR by organism by decade ─────────

def fig_s1(res_dir, supp_dir):
    """S1_growth_rates — CAGR by organism by decade."""
    print("S1: Growth rates...", flush=True)
    df = _load(f"{res_dir}/main/organism_growth_rates.csv",
               "organism_growth_rates.csv")
    if df is None:
        return

    df["organism"] = df["organism"].str.lower().str.strip()

    cagr_col = None
    for c in ("cagr", "growth_rate", "CAGR"):
        if c in df.columns:
            cagr_col = c
            break
    if cagr_col is None:
        print(f"  SKIP S1: no CAGR column; columns={list(df.columns)}", flush=True)
        return

    decade_col = None
    for c in ("decade", "period", "decade_label"):
        if c in df.columns:
            decade_col = c
            break
    if decade_col is None:
        print(f"  SKIP S1: no decade column; columns={list(df.columns)}", flush=True)
        return

    # Keep top 8 organisms
    top_orgs = df.groupby("organism")[cagr_col].mean().abs().nlargest(8).index.tolist()
    df = df[df["organism"].isin(top_orgs)]

    pivot = df.pivot_table(index="organism", columns=decade_col,
                           values=cagr_col, aggfunc="mean")

    n_decades = len(pivot.columns)
    n_orgs = len(pivot)
    x = np.arange(n_decades)
    width = 0.8 / n_orgs

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, org in enumerate(pivot.index):
        offsets = x + (i - n_orgs / 2 + 0.5) * width
        vals = pivot.loc[org].values
        bars = ax.bar(offsets, vals, width=width * 0.9,
                      color=_org_color(org), alpha=0.85,
                      label=org.capitalize())

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("Decade")
    ax.set_ylabel("CAGR")
    ax.set_title("Compound Annual Growth Rate by Organism and Decade")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    _save(fig, "S1_growth_rates", supp_dir)


# ── S2: Paradigm citations — bar of median citations ──────────────────

def fig_s2(res_dir, supp_dir):
    """S2_paradigm_citations — median citations per paradigm × organism."""
    print("S2: Paradigm citation impact...", flush=True)
    df = _load(f"{res_dir}/supplementary/paradigm_citation_impact.csv",
               "paradigm_citation_impact.csv")
    if df is None:
        return

    df["organism"] = df["organism"].str.lower().str.strip()
    df = df[df["organism"].isin(TOP6_ORGANISMS)]

    # Shorten paradigm labels
    if "paradigm" in df.columns:
        df["paradigm_short"] = df["paradigm"].str[:30]
    else:
        print(f"  SKIP S2: no paradigm column; columns={list(df.columns)}", flush=True)
        return

    cite_col = "median_citations" if "median_citations" in df.columns else "mean_citations"

    pivot = df.pivot_table(index="paradigm_short", columns="organism",
                           values=cite_col, aggfunc="mean", fill_value=0)
    # Keep only organisms in TOP6 that are present
    present = [o for o in TOP6_ORGANISMS if o in pivot.columns]
    pivot = pivot[present]

    n_paradigms = len(pivot)
    n_orgs = len(present)
    x = np.arange(n_paradigms)
    width = 0.8 / n_orgs

    fig, ax = plt.subplots(figsize=(10, max(5, n_paradigms * 0.55)))

    for i, org in enumerate(present):
        offsets = x + (i - n_orgs / 2 + 0.5) * width
        ax.bar(offsets, pivot[org].values, width=width * 0.9,
               color=_org_color(org), alpha=0.85, label=org.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right", fontsize=7)
    ax.set_xlabel("Research paradigm")
    ax.set_ylabel(f"{cite_col.replace('_', ' ').capitalize()}")
    ax.set_title("Citation Impact by Paradigm and Organism")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, "S2_paradigm_citations", supp_dir)


# ── S3: Topic diversity — Shannon entropy ─────────────────────────────

def fig_s3(res_dir, supp_dir):
    """S3_topic_diversity — Shannon entropy over years."""
    print("S3: Topic diversity...", flush=True)
    df = _load(f"{res_dir}/supplementary/topic_diversity_by_year.csv",
               "topic_diversity_by_year.csv")
    if df is None:
        return

    if "shannon_entropy" not in df.columns:
        # Try to find the right column
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if "year" in numeric_cols:
            numeric_cols.remove("year")
        if not numeric_cols:
            print("  SKIP S3: no entropy column found", flush=True)
            return
        entropy_col = numeric_cols[0]
    else:
        entropy_col = "shannon_entropy"

    df = df[df["year"].between(1990, 2024)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4))

    ax.plot(df["year"], df[entropy_col], color="#4393c3", linewidth=2)
    # Rolling smoothing
    roll = df.set_index("year")[entropy_col].rolling(5, center=True, min_periods=2).mean()
    ax.plot(roll.index, roll.values, color="#d6604d", linewidth=2,
            linestyle="--", label="5-yr rolling mean")

    ax.set_xlabel("Year")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Topic Diversity in Plant Science Literature (1990–2024)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "S3_topic_diversity", supp_dir)


# ── S6: Disruption index — bar of mean CD per organism ────────────────

def fig_s6(res_dir, supp_dir):
    """S6_disruption_index — mean CD index by organism (latest decade)."""
    print("S6: Disruption index...", flush=True)
    df = _load(f"{res_dir}/supplementary/organism_disruption_index.csv",
               "organism_disruption_index.csv")
    if df is None:
        return

    df["organism"] = df["organism"].str.lower().str.strip()

    # Latest decade
    if "decade" in df.columns:
        latest = df["decade"].max()
        df = df[df["decade"] == latest]

    cd_col = "mean_cd" if "mean_cd" in df.columns else "cd_index"
    if cd_col not in df.columns:
        print(f"  SKIP S6: no CD column; columns={list(df.columns)}", flush=True)
        return

    df = df.sort_values(cd_col, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [_org_color(o) for o in df["organism"]]
    bars = ax.bar(df["organism"], df[cd_col], color=colors, alpha=0.85)

    for bar, val in zip(bars, df[cd_col]):
        ha = "center"
        va = "bottom" if val >= 0 else "top"
        offset = 0.002 if val >= 0 else -0.002
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:.3f}", ha=ha, va=va, fontsize=7)

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Organism")
    ax.set_ylabel("Mean CD Index")
    decade_label = f" ({latest})" if "decade" in df.columns else ""
    ax.set_title(f"Disruption Index by Organism{decade_label}\n"
                 "Positive = disruptive; Negative = consolidating")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    _save(fig, "S6_disruption_index", supp_dir)


# ── S9: Cross-citation matrix heatmap ─────────────────────────────────

def fig_s9(res_dir, supp_dir):
    """S9_cross_citation — heatmap of organism cross-citation matrix."""
    print("S9: Cross-citation matrix...", flush=True)
    df = _load(f"{res_dir}/supplementary/organism_cross_citation.csv",
               "organism_cross_citation.csv")
    if df is None:
        return

    # Expect: citing_organism, cited_organism, n_citations (or share)
    if "citing_organism" not in df.columns or "cited_organism" not in df.columns:
        print(f"  SKIP S9: expected citing/cited columns; cols={list(df.columns)}", flush=True)
        return

    val_col = "share" if "share" in df.columns else "n_citations"
    pivot = df.pivot_table(index="citing_organism", columns="cited_organism",
                           values=val_col, aggfunc="sum", fill_value=0)

    # Exclude non_specific rows/cols
    for dim in (pivot.index, pivot.columns):
        pass  # handled below
    excl = "non_specific"
    pivot = pivot.loc[
        [i for i in pivot.index if i != excl],
        [c for c in pivot.columns if c != excl],
    ]

    fig_sz = max(6, len(pivot) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_sz, fig_sz * 0.85))

    fmt = ".2f" if val_col == "share" else ".0f"
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": val_col.replace("_", " ").capitalize(), "shrink": 0.7},
                annot=True, fmt=fmt, annot_kws={"size": 8})

    ax.set_xlabel("Cited organism")
    ax.set_ylabel("Citing organism")
    ax.set_title("Cross-Citation Network Between Organisms\n"
                 "(row = citing, column = cited)")
    ax.tick_params(axis="x", rotation=40, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    _save(fig, "S9_cross_citation", supp_dir)


# ── S10: International collaboration rate ────────────────────────────

def fig_s10(res_dir, supp_dir):
    """S10_collaboration — intl collab rate per organism over time."""
    print("S10: Collaboration...", flush=True)
    df = _load(f"{res_dir}/supplementary/organism_collaboration.csv",
               "organism_collaboration.csv")
    if df is None:
        return

    df["organism"] = df["organism"].str.lower().str.strip()

    collab_col = None
    for c in ("intl_collab_rate", "international_collab_rate", "collab_rate"):
        if c in df.columns:
            collab_col = c
            break
    if collab_col is None:
        print(f"  SKIP S10: no collab column; columns={list(df.columns)}", flush=True)
        return

    df = df[df["year"].between(1990, 2024)].copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    for org in TOP6_ORGANISMS:
        sub = df[df["organism"] == org].sort_values("year")
        if sub.empty:
            continue
        ax.plot(sub["year"], sub[collab_col] * 100,
                color=_org_color(org), linewidth=1.8,
                label=org.capitalize())

    ax.set_xlabel("Year")
    ax.set_ylabel("International collaboration rate (%)")
    ax.set_title("International Collaboration Rate by Organism (1990–2024)")
    ax.set_xlim(1990, 2024)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, "S10_collaboration", supp_dir)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Generate Paper A GPU-enriched publication figures."
    )
    ap.add_argument("--results-dir", default="results/paper_a",
                    help="Paper A results directory (default: results/paper_a)")
    ap.add_argument("--out-dir", default="results/paper_a/figures",
                    help="Output base directory (default: results/paper_a/figures)")
    args = ap.parse_args()

    res_dir  = args.results_dir
    out_dir  = args.out_dir
    main_dir = os.path.join(out_dir, "main")
    supp_dir = os.path.join(out_dir, "supplementary")

    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(supp_dir, exist_ok=True)

    # Project root = two levels above this file's package (src/viz/)
    project_root = str(Path(__file__).resolve().parents[2])

    print(f"Results dir : {res_dir}", flush=True)
    print(f"Output dir  : {out_dir}", flush=True)
    print("Generating Paper A GPU-enriched figures...\n", flush=True)

    # ── Main figures ──────────────────────────────────────────────────
    fig_m1(res_dir, main_dir)
    fig_m2(res_dir, main_dir)
    fig_m3(res_dir, main_dir)
    fig_m4(res_dir, main_dir, project_root)
    fig_m5(res_dir, main_dir)
    fig_m6(res_dir, main_dir)
    fig_m7(res_dir, main_dir)
    fig_m8(res_dir, main_dir)

    # ── Supplementary figures ─────────────────────────────────────────
    fig_s1(res_dir, supp_dir)
    fig_s2(res_dir, supp_dir)
    fig_s3(res_dir, supp_dir)
    fig_s6(res_dir, supp_dir)
    fig_s9(res_dir, supp_dir)
    fig_s10(res_dir, supp_dir)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
