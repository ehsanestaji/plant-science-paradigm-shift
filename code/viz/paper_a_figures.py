"""
Figure generation for Paper A: Nature Plants — paradigm shift narrative.

Reads analysis results from results/ and produces publication-quality
figures in figures/ as PDF + PNG (300 DPI).

Usage:
    python -m src.viz.paper_a_figures [--results-dir results --out-dir figures]

Figures:
    FA1  Growth of plant science with logistic inflection & plateau
    FA2  Model organisms — Arabidopsis peak vs crop species rising
    FA3  Sleeping beauty citation trajectories with 2019-2022 awakening band
    FA4  Method diffusion curves with CRISPR/ML/RNA-seq highlighted
    FA5  Concept marriages — epoch bands + marriages per year
    FA6  Orphan organisms — horizontal attention gap bar chart
    FA7  Cross-field comparison — two-panel growth & model-organism trajectories
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style (mirrors generate_all_figures.py) ───────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("Set2", 12)


# ── Helpers ───────────────────────────────────────────────────────────

def _save(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_dir}/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}", flush=True)


def _load(path):
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found", flush=True)
        return None
    return pd.read_csv(path)


# ── FA1: Growth with logistic inflection & plateau ────────────────────

def FA1_growth_logistic(res_dir, out_dir):
    """Bar chart of papers/year 1960-2024 with the fitted logistic curve
    overlaid, inflection at 2006, and carrying capacity K=217,781.

    Numbers match the main-text claim and SI S7 AICc comparison, derived
    from the works_clean logistic fit in
    src/analysis/paper_a/hardening/sensitivity_logistic.py.
    """
    print("FA1: Growth logistic...", flush=True)
    df = _load(f"{res_dir}/temporal/papers_per_year.csv")
    if df is None:
        return

    df = df[(df["year"] >= 1960) & (df["year"] <= 2024)].copy()
    if df.empty:
        print("  SKIP: no data in range 1960-2024", flush=True)
        return

    fig, ax = plt.subplots(figsize=(13, 6))

    # Bars coloured by period: pre-1990 pre-genomics (muted grey-green);
    # 1990-2006 acceleration (teal); 2006-2024 post-inflection approach to
    # carrying capacity K (green).
    colours = []
    for y in df["year"]:
        if y < 1990:
            colours.append(PALETTE[7])   # muted grey-green
        elif y <= 2006:
            colours.append(PALETTE[0])   # teal (acceleration)
        else:
            colours.append(PALETTE[2])   # green (post-inflection)

    ax.bar(df["year"], df["n_papers"], color=colours, width=0.85, alpha=0.85,
           zorder=2)

    # Fitted logistic curve — values from logistic_sensitivity.csv
    # (works_clean, 1990-2024 truncation): K=217,781, r=0.1131, t_mid=2006.05
    K_FIT, R_FIT, T_MID = 217781.0, 0.1131, 2006.05
    years_smooth = np.linspace(1990, 2024, 200)
    logistic_y = K_FIT / (1.0 + np.exp(-R_FIT * (years_smooth - T_MID)))
    ax.plot(years_smooth, logistic_y, color="black", linewidth=2.4,
            linestyle="-", zorder=4,
            label=f"Logistic fit (K={K_FIT/1000:.0f}K, $t_{{\\mathrm{{mid}}}}$=2006)")

    # Carrying-capacity horizontal reference
    ax.axhline(K_FIT, color="black", linestyle=":", linewidth=1, alpha=0.6,
               zorder=1)
    ax.text(1962, K_FIT + 3000, f"K = {K_FIT:,.0f}",
            fontsize=9, color="black", alpha=0.7)

    # Inflection annotation (2006)
    y_inflect = df.loc[df["year"] == 2006, "n_papers"]
    if not y_inflect.empty:
        yv = y_inflect.values[0]
        ax.annotate(
            "Logistic inflection\n$t_{\\mathrm{mid}} = 2006$",
            xy=(2006, yv),
            xytext=(1993, yv * 2.2),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            ha="center",
        )
    ax.axvline(2006, color="black", linestyle="--", linewidth=1, alpha=0.5,
               zorder=1)

    # Plateau shading (2015-2024) — approach to K
    ax.axvspan(2015, 2024.5, alpha=0.08, color=PALETTE[2],
               label="Plateau period (2015–2024)")
    ax.text(2019, df["n_papers"].max() * 0.97,
            f"Peak 199K\n(\u2248 91\% of K)",
            ha="center", va="top", fontsize=9, color="darkgreen")

    # Pre-1990 label
    ax.axvspan(1960, 1989.5, alpha=0.04, color=PALETTE[7])
    ax.text(1974, df["n_papers"].max() * 0.10, "Pre-genomics era",
            ha="center", va="bottom", fontsize=8, color="grey")

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of publications")
    ax.set_title("Growth of Plant Science Research (1960–2024)\n"
                 "with Logistic Fit and Inflection at 2006")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.set_xlim(1959, 2025)
    ax.set_ylim(0, K_FIT * 1.08)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "FA1_growth_logistic", out_dir)


# ── FA2: Model organisms — Arabidopsis peak vs rising crops ───────────

def FA2_model_organisms(res_dir, out_dir):
    """Line plot: Arabidopsis (dashed bold) peaked ~2012 vs rice/wheat/maize
    continuing to rise.  Falls back to organism_paper_counts.csv if the
    dynamics/model_organisms.csv time-series is absent."""
    print("FA2: Model organisms...", flush=True)

    df = _load(f"{res_dir}/dynamics/model_organisms.csv")
    if df is None:
        print("  Trying fallback: novel/organism_paper_counts.csv", flush=True)
        df = _load(f"{res_dir}/novel/organism_paper_counts.csv")
        if df is None:
            return
        # organism_paper_counts has no year column — can't make a line plot
        print("  SKIP FA2: organism_paper_counts has no year dimension", flush=True)
        return

    # Normalise column names
    df = df.rename(columns=str.lower)
    if "organism" not in df.columns or "year" not in df.columns:
        print("  SKIP FA2: expected columns 'organism' and 'year'", flush=True)
        return

    df = df[(df["year"] >= 1980) & (df["year"] <= 2024)].copy()

    # Organisms of interest
    ARABIDOPSIS_KEYS = ["arabidopsis", "Arabidopsis"]
    CROP_KEYS = ["rice", "wheat", "maize", "soybean", "tomato", "potato",
                 "barley", "cassava", "sorghum"]

    fig, ax = plt.subplots(figsize=(13, 7))

    plotted = set()
    n_paper_col = "n_papers" if "n_papers" in df.columns else "n"

    # Arabidopsis — dashed, bold, highlighted
    for key in ARABIDOPSIS_KEYS:
        sub = df[df["organism"].str.lower() == key.lower()]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub[n_paper_col],
                color="black", linewidth=3, linestyle="--",
                label="Arabidopsis", zorder=5)
        # Annotate peak
        peak_row = sub.loc[sub[n_paper_col].idxmax()]
        ax.annotate(
            f"Peak {int(peak_row['year'])}",
            xy=(peak_row["year"], peak_row[n_paper_col]),
            xytext=(peak_row["year"] - 4, peak_row[n_paper_col] * 1.08),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=1),
        )
        plotted.add(key.lower())
        break  # only first match

    # Crop species
    for i, crop in enumerate(CROP_KEYS):
        sub = df[df["organism"].str.lower() == crop.lower()]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("year")
        ax.plot(sub["year"], sub[n_paper_col],
                color=PALETTE[i % len(PALETTE)], linewidth=2, alpha=0.9,
                label=crop.capitalize())
        plotted.add(crop.lower())

    # Any remaining organisms not yet plotted
    remaining = [o for o in df["organism"].unique()
                 if o.lower() not in plotted]
    for i, org in enumerate(remaining[:4]):
        sub = df[df["organism"] == org].sort_values("year")
        ax.plot(sub["year"], sub[n_paper_col],
                color=PALETTE[(i + len(CROP_KEYS)) % len(PALETTE)],
                linewidth=1.5, alpha=0.6, linestyle=":", label=org)

    ax.axvline(2012, color="grey", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(2012.5, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1000,
            "~2012", fontsize=8, color="grey", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of papers")
    ax.set_title("Model Organism Trajectories in Plant Science\n"
                 "Arabidopsis peaked ~2012; crop species continue rising")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    _save(fig, "FA2_model_organisms", out_dir)


# ── FA3: Sleeping beauties — citation trajectories + gold awakening band

def FA3_sleeping_beauties(res_dir, out_dir):
    """Top-8 sleeping beauty citation trajectories (paper age on x-axis)
    with a gold band marking the 2019-2022 awakening cluster."""
    print("FA3: Sleeping beauties...", flush=True)
    scores = _load(f"{res_dir}/novel/sleeping_beauty_scores.csv")
    ts     = _load(f"{res_dir}/novel/sleeping_beauty_timeseries.csv")
    if scores is None or ts is None:
        return

    # Take top 8 by B score
    top8 = scores.nlargest(8, "B").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 7))

    # Gold awakening band (calendar years 2019-2022 will be converted to
    # paper-age per paper, so we shade in paper-age space per paper instead;
    # here we draw a background annotation block)
    ax.axvspan(15, 45, ymin=0, ymax=1, alpha=0.08, color="#F5A623",
               label="Typical awakening window (~15–45 yrs)")
    ax.text(30, 0, "Awakening\ncluster", ha="center", va="bottom",
            fontsize=8, color="#C68A00", transform=ax.get_xaxis_transform())

    GOLD = "#F5A623"
    colors = sns.color_palette("tab10", 8)

    for i, (_, sb) in enumerate(top8.iterrows()):
        paper_ts = ts[ts["focal_id"] == sb["work_id"]].sort_values("paper_age")
        if len(paper_ts) == 0:
            continue

        label = f"{sb.get('crop', 'unknown').capitalize()} B={sb['B']:.0f}"
        ax.plot(paper_ts["paper_age"], paper_ts["n_new_citations"],
                color=colors[i], linewidth=2, alpha=0.85, label=label)

        # Mark awakening star
        awake_yr = sb.get("awakening_year")
        pub_yr   = sb.get("pub_year")
        if pd.notna(awake_yr) and pd.notna(pub_yr):
            age_awake = int(awake_yr) - int(pub_yr)
            row_at = paper_ts[paper_ts["paper_age"] == age_awake]
            if len(row_at):
                ax.scatter([age_awake], [row_at["n_new_citations"].values[0]],
                           marker="*", s=180, color=colors[i],
                           edgecolors=GOLD, linewidths=1.5, zorder=6)

    ax.set_xlabel("Paper age (years since publication)")
    ax.set_ylabel("Annual new citations")
    ax.set_title("Sleeping Beauty Papers in Plant Science\n"
                 "Top 8 by Beauty coefficient B  (★ = awakening year)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_xlim(left=0)
    fig.tight_layout()
    _save(fig, "FA3_sleeping_beauties", out_dir)


# ── FA4: Method diffusion — CRISPR/ML/RNA-seq highlighted ─────────────

def FA4_method_diffusion(res_dir, out_dir):
    """Cumulative adoption S-curves for all tracked methods; CRISPR,
    Machine learning and RNA-seq are drawn bold and labelled."""
    print("FA4: Method diffusion...", flush=True)
    ts = _load(f"{res_dir}/novel/method_adoption_timeseries.csv")
    if ts is None:
        return

    # Normalise method column (strip extra whitespace)
    ts["method"] = ts["method"].str.strip()

    HIGHLIGHT = {
        "CRISPR":           (PALETTE[3], 3.0),
        "Machine learning": (PALETTE[0], 3.0),
        "RNA-seq":          (PALETTE[1], 3.0),
    }

    fig, ax = plt.subplots(figsize=(13, 7))

    methods = ts["method"].unique()

    for method in methods:
        mdf = ts[ts["method"] == method].sort_values("year").copy()
        if len(mdf) < 3:
            continue
        origin = mdf["origin_year"].iloc[0] if "origin_year" in mdf.columns else mdf["year"].min()
        cumul = mdf["n_papers"].cumsum()
        peak  = cumul.iloc[-1]
        if peak == 0:
            continue
        pct   = cumul / peak * 100
        years_since = mdf["year"] - origin

        if method in HIGHLIGHT:
            color, lw = HIGHLIGHT[method]
            ax.plot(years_since, pct, color=color, linewidth=lw,
                    label=method, zorder=5)
            # Label at 50 % adoption mark
            idx_50 = (pct - 50).abs().idxmin()
            ax.text(years_since.loc[idx_50],
                    pct.loc[idx_50] + 3,
                    method, fontsize=8, color=color,
                    fontweight="bold", ha="left")
        else:
            ax.plot(years_since, pct, color="lightgrey", linewidth=1,
                    alpha=0.7, zorder=2)

    ax.axhline(50, color="grey", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(-4.5, 51, "50 %", fontsize=8, color="grey", va="bottom")

    ax.set_xlabel("Years since method origin")
    ax.set_ylabel("Cumulative adoption (% of peak)")
    ax.set_title("Method Diffusion S-Curves in Plant Science\n"
                 "CRISPR, Machine learning & RNA-seq highlighted")
    ax.set_xlim(-5, 45)
    ax.set_ylim(0, 105)

    # Build legend for highlights + grey background guide
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=lw, label=m)
        for m, (c, lw) in HIGHLIGHT.items()
    ]
    handles.append(plt.Line2D([0], [0], color="lightgrey",
                              linewidth=1, label="Other methods"))
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "FA4_method_diffusion", out_dir)


# ── FA5: Concept marriages — epoch bands + bars per year ──────────────

def FA5_concept_marriages(res_dir, out_dir):
    """Two-panel: (A) marriage count per year coloured by epoch band,
    (B) epoch summary bar chart."""
    print("FA5: Concept marriages...", flush=True)
    marriages = _load(f"{res_dir}/novel/concept_marriages.csv")
    epochs_df = _load(f"{res_dir}/novel/concept_epochs.csv")
    if marriages is None:
        return

    # Filter to plant-science-relevant concepts
    off_topic = [
        "economics", "finance", "law", "criminol", "astronomy", "astrophysics",
        "particle physics", "quantum", "nuclear", "geology", "civil engineer",
        "literature", "aesthetics", "music", "religion", "philosophy",
        "psychology", "marketing", "accounting", "management", "sociology",
    ]
    def _is_relevant(name):
        n = str(name).lower()
        return not any(kw in n for kw in off_topic)

    m = marriages[
        marriages["name_a"].apply(_is_relevant) &
        marriages["name_b"].apply(_is_relevant)
    ].copy()

    EPOCH_ORDER = [
        "1960s–1974 (Classical Era)",
        "1975–1984 (Molecular Biology)",
        "1985–1994 (Genomics Dawn)",
        "1995–2004 (Bioinformatics)",
        "2005–2014 (Omics Revolution)",
        "2015–2024 (AI & Climate Era)",
    ]
    # Epoch date ranges for band shading
    EPOCH_SPANS = [
        (1960, 1974),
        (1975, 1984),
        (1985, 1994),
        (1995, 2004),
        (2005, 2014),
        (2015, 2024),
    ]
    epoch_colors = {e: PALETTE[i] for i, e in enumerate(EPOCH_ORDER)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10),
                                    gridspec_kw={"height_ratios": [2, 1]})

    # ── Panel A: bars per year, coloured by epoch ─────────────────────
    if "marriage_year" in m.columns:
        yearly = m.groupby("marriage_year").size().reset_index(name="n_marriages")
        yearly = yearly[(yearly["marriage_year"] >= 1960) &
                        (yearly["marriage_year"] <= 2024)]
        # Assign epoch colour to each bar
        def _epoch_color(yr):
            for (start, end), epoch in zip(EPOCH_SPANS, EPOCH_ORDER):
                if start <= yr <= end:
                    return epoch_colors[epoch]
            return "grey"
        bar_colors = [_epoch_color(y) for y in yearly["marriage_year"]]
        ax1.bar(yearly["marriage_year"], yearly["n_marriages"],
                color=bar_colors, width=0.85, alpha=0.85)

        # Epoch band overlays (light vertical spans)
        for (start, end), epoch in zip(EPOCH_SPANS, EPOCH_ORDER):
            ax1.axvspan(start, end, alpha=0.05,
                        color=epoch_colors[epoch])
            ax1.text((start + end) / 2,
                     yearly["n_marriages"].max() * 0.97,
                     epoch.split(" (")[0],
                     ha="center", va="top", fontsize=7,
                     color=epoch_colors[epoch], fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "marriage_year column not found",
                 transform=ax1.transAxes, ha="center")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("New concept marriages")
    ax1.set_title("A — Annual Concept Marriages in Plant Science Knowledge Space")

    # ── Panel B: epoch summary (n_marriages from epochs_df or computed) ─
    if epochs_df is not None and "n_marriages" in epochs_df.columns:
        present = [e for e in EPOCH_ORDER if e in epochs_df["epoch"].values]
        epoch_counts = epochs_df.set_index("epoch").reindex(present)["n_marriages"]
    elif "epoch" in m.columns:
        present = [e for e in EPOCH_ORDER if e in m["epoch"].values]
        epoch_counts = m.groupby("epoch").size().reindex(present, fill_value=0)
    else:
        ax2.set_visible(False)
        fig.tight_layout(pad=2)
        _save(fig, "FA5_concept_marriages", out_dir)
        return

    short_labels = [e.split(" (")[0] for e in present]
    colors_bar = [epoch_colors[e] for e in present]
    bars = ax2.bar(short_labels, epoch_counts.values, color=colors_bar, alpha=0.85)
    for bar, val in zip(bars, epoch_counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1, str(int(val)),
                 ha="center", va="bottom", fontsize=9)
    ax2.set_xlabel("Intellectual Epoch")
    ax2.set_ylabel("Concept marriages")
    ax2.set_title("B — Total Concept Marriages per Epoch")
    ax2.tick_params(axis="x", rotation=25)

    fig.tight_layout(pad=2)
    _save(fig, "FA5_concept_marriages", out_dir)


# ── FA6: Orphan organisms — horizontal attention gap chart ────────────

def FA6_orphan_organisms(res_dir, out_dir):
    """Diverging horizontal bar chart: attention_gap_log2 per organism.
    Negative = under-researched relative to caloric/food importance."""
    print("FA6: Orphan organisms...", flush=True)
    df = _load(f"{res_dir}/novel/research_attention_gap.csv")
    if df is None:
        return

    if "attention_gap_log2" not in df.columns:
        print("  SKIP FA6: attention_gap_log2 column not found", flush=True)
        return

    gap = df.dropna(subset=["attention_gap_log2"]).copy()
    gap = gap.sort_values("attention_gap_log2")

    # Drop rows with NaN organism
    gap = gap.dropna(subset=["organism"])

    colors = ["#d62728" if v < 0 else "#2ca02c"
              for v in gap["attention_gap_log2"]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(gap) * 0.45)))

    bars = ax.barh(gap["organism"], gap["attention_gap_log2"],
                   color=colors, height=0.65, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.9)

    # Value labels
    for bar, val in zip(bars, gap["attention_gap_log2"]):
        ha = "left" if val >= 0 else "right"
        offset = 0.03 if val >= 0 else -0.03
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=8)

    # Legend patches
    under = mpatches.Patch(color="#d62728", label="Under-researched (negative gap)")
    over  = mpatches.Patch(color="#2ca02c", label="Over-researched (positive gap)")
    ax.legend(handles=[under, over], fontsize=9, loc="lower right")

    ax.set_xlabel("Attention Gap Index (log₂ ratio vs expected from food importance)")
    ax.set_title("Orphan Organisms in Plant Science\n"
                 "Negative = under-researched relative to caloric/agricultural importance")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    _save(fig, "FA6_orphan_organisms", out_dir)


# ── FA7: Cross-field comparison — two-panel ───────────────────────────

def FA7_cross_field(res_dir, out_dir):
    """Two-panel:
    (a) Normalized growth trajectories: plant science vs other fields
        (uses cross_field_growth.csv if present, falls back to papers_per_year)
    (b) Model organisms across biology
        (uses cross_field_model_organisms.csv if present)
    """
    print("FA7: Cross-field comparison...", flush=True)

    cf_growth = _load(f"{res_dir}/temporal/cross_field_growth.csv")
    cf_orgs   = _load(f"{res_dir}/temporal/cross_field_model_organisms.csv")

    # Determine which panels we can draw
    has_growth = cf_growth is not None
    has_orgs   = cf_orgs is not None

    if not has_growth and not has_orgs:
        # Fallback: single-panel using papers_per_year as plant science only
        print("  No cross-field CSVs found — using fallback single-field view",
              flush=True)
        df = _load(f"{res_dir}/temporal/papers_per_year.csv")
        if df is None:
            return
        fig, ax = plt.subplots(figsize=(11, 6))
        df = df[(df["year"] >= 1960) & (df["year"] <= 2024)].copy()
        # Normalize to 1990 baseline
        baseline_val = df.loc[df["year"] == 1990, "n_papers"]
        if baseline_val.empty:
            baseline_val = df["n_papers"].iloc[0]
        else:
            baseline_val = baseline_val.values[0]
        if baseline_val > 0:
            df["norm"] = df["n_papers"] / baseline_val
        else:
            df["norm"] = df["n_papers"]
        ax.plot(df["year"], df["norm"], color=PALETTE[0], linewidth=2.5,
                label="Plant science")
        ax.axhline(1, color="grey", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized publication count (1990 = 1.0)")
        ax.set_title("FA7 — Plant Science Growth (cross-field data not yet available)")
        ax.legend()
        fig.tight_layout()
        _save(fig, "FA7_cross_field", out_dir)
        return

    # Full two-panel version
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Panel A: normalized growth trajectories ───────────────────────
    ax1 = axes[0]
    if has_growth:
        field_col = None
        for candidate in ("field", "discipline", "category", "subject"):
            if candidate in cf_growth.columns:
                field_col = candidate
                break
        year_col = "year" if "year" in cf_growth.columns else cf_growth.columns[0]
        val_col  = None
        for candidate in ("n_papers", "n", "count", "normalized"):
            if candidate in cf_growth.columns:
                val_col = candidate
                break

        if field_col is None or val_col is None:
            ax1.text(0.5, 0.5,
                     "Cannot identify field/value columns\nin cross_field_growth.csv",
                     transform=ax1.transAxes, ha="center", fontsize=9)
        else:
            baseline_year = 1990
            for i, (field, grp) in enumerate(cf_growth.groupby(field_col)):
                grp = grp.sort_values(year_col)
                bv_row = grp.loc[grp[year_col] == baseline_year, val_col]
                bv = bv_row.values[0] if len(bv_row) else grp[val_col].iloc[0]
                if bv == 0:
                    continue
                norm = grp[val_col] / bv
                is_plant = "plant" in str(field).lower()
                lw  = 3.0 if is_plant else 1.5
                ls  = "-" if is_plant else "--"
                col = PALETTE[0] if is_plant else PALETTE[i % len(PALETTE)]
                ax1.plot(grp[year_col], norm, color=col,
                         linewidth=lw, linestyle=ls,
                         label=field, alpha=0.9 if is_plant else 0.7, zorder=5 if is_plant else 2)

            ax1.axhline(1, color="grey", linewidth=0.7, linestyle=":")
            ax1.axvline(2012, color="grey", linewidth=0.7, linestyle=":", alpha=0.5)
            ax1.set_xlabel("Year")
            ax1.set_ylabel(f"Normalized publications ({baseline_year} = 1.0)")
            ax1.set_title("A — Normalized Growth Trajectories Across Biology")
            ax1.legend(fontsize=8, ncol=1, loc="upper left")
    else:
        ax1.text(0.5, 0.5, "cross_field_growth.csv not available",
                 transform=ax1.transAxes, ha="center", fontsize=9)
        ax1.set_title("A — Cross-field Growth (data pending)")

    # ── Panel B: model organisms across biology ───────────────────────
    ax2 = axes[1]
    if has_orgs:
        org_col = None
        for candidate in ("organism", "model_organism", "species"):
            if candidate in cf_orgs.columns:
                org_col = candidate
                break
        year_col2 = "year" if "year" in cf_orgs.columns else cf_orgs.columns[0]
        val_col2  = None
        for candidate in ("n_papers", "n", "count"):
            if candidate in cf_orgs.columns:
                val_col2 = candidate
                break
        field_col2 = None
        for candidate in ("field", "discipline", "biology_field"):
            if candidate in cf_orgs.columns:
                field_col2 = candidate
                break

        if org_col is None or val_col2 is None:
            ax2.text(0.5, 0.5,
                     "Cannot identify organism/value columns\nin cross_field_model_organisms.csv",
                     transform=ax2.transAxes, ha="center", fontsize=9)
        else:
            group_col = field_col2 if field_col2 else org_col
            for i, (grp_name, grp) in enumerate(cf_orgs.groupby(group_col)):
                grp = grp.sort_values(year_col2)
                ax2.plot(grp[year_col2], grp[val_col2],
                         color=PALETTE[i % len(PALETTE)], linewidth=2,
                         label=grp_name, alpha=0.8)
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Number of papers")
            ax2.set_title("B — Model Organism Usage Across Biology Fields")
            ax2.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
            ax2.legend(fontsize=8, ncol=1, loc="upper left")
    else:
        ax2.text(0.5, 0.5, "cross_field_model_organisms.csv not available",
                 transform=ax2.transAxes, ha="center", fontsize=9)
        ax2.set_title("B — Model Organisms Across Biology (data pending)")

    fig.suptitle("Cross-Field Comparison: Plant Science in the Broader Biology Landscape",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "FA7_cross_field", out_dir)


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Paper A (Nature Plants) figures."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory (default: results)")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory for figures (default: figures)")
    args = parser.parse_args()

    res_dir = args.results_dir
    out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    print(f"Results dir : {res_dir}", flush=True)
    print(f"Output dir  : {out_dir}", flush=True)
    print("Generating Paper A figures...", flush=True)

    FA1_growth_logistic(res_dir, out_dir)
    FA2_model_organisms(res_dir, out_dir)
    FA3_sleeping_beauties(res_dir, out_dir)
    FA4_method_diffusion(res_dir, out_dir)
    FA5_concept_marriages(res_dir, out_dir)
    FA6_orphan_organisms(res_dir, out_dir)
    FA7_cross_field(res_dir, out_dir)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
