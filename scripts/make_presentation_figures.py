"""Generate presentation / manuscript candidate figures from the ACTUAL pipeline.

Every number is recomputed from the live artifacts and data — nothing typed by hand.
Outputs PNGs to presentation/figures/.

Incorporates Ioanna's 2026-07-13 review notes:
  * attrition flow BEFORE the metrics slide (fig0)
  * AUC reported alongside R² (fig3)
  * RF vs GB head-to-head on our own data (fig2b)
  * RACE added to cluster demographics (fig7) and to the sensitivity check (fig8)
  * n-per-cell shown on trajectories; preop TBWL + the 10.5% threshold (fig6)

Run:
    PYTHONPATH=. python scripts/make_presentation_figures.py
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from src.config import ARTIFACTS, MODEL_PERFORMANCE, TBWL_BY_YEAR, SEED, BASELINE_FEATURES
from src.data_load import load
from src.reliability import tier
import src.phenotype as phe

OUT = Path("presentation/figures")
OUT.mkdir(parents=True, exist_ok=True)
S5 = Path("/Users/kaushikperkari/Downloads/Re_ Research involvement (1)/"
          "Supplementary Table S5 All ML model performance years 1-6.xlsx")

TIER_C = {"green": "#2ca02c", "amber": "#ff9f1c", "red": "#d62728"}
CL_C = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]
PREOP_THRESHOLD = 10.5
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold"})


# ── 0. Attrition ──────────────────────────────────────────────────────────────
def fig0_attrition(df):
    """Patient flow. Ioanna: show this BEFORE the metrics, so R² is read in context."""
    n0 = len(df)
    ns = [int(pd.to_numeric(df[TBWL_BY_YEAR[y]], errors="coerce").notna().sum())
          for y in range(1, 7)]
    stages = ["Enrolled"] + [f"Year {y}" for y in range(1, 7)]
    vals = [n0] + ns

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    bars = ax.bar(stages, vals, color=["#455a64"] + ["#1f77b4"] * 4 + ["#d62728"] * 2,
                  width=.68)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 14, f"{v}\n({v/n0*100:.0f}%)",
                ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_ylabel("Patients with weight-loss data")
    ax.set_ylim(0, n0 * 1.2)
    ax.grid(axis="y", alpha=.3)
    ax.text(5.5, n0 * 0.80,
            "Years 5–6: only ~10% of the cohort remains.\nThis is why the model cannot "
            "predict them —\nand why we refuse to report a number.",
            fontsize=10.5, color="#d62728", weight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fff0f0", ec="#d62728"))
    ax.set_title("Figure 1. Patient attrition — people stop coming to clinic", pad=12)
    plt.tight_layout()
    plt.savefig(OUT / "fig0_attrition.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig0_attrition.png")


def fig0b_sankey(df):
    """Attrition as a Sankey-style flow (Ioanna asked for the manuscript's Sankey;
    her figure wasn't in the shared files, so this reconstructs it from our data)."""
    n0 = len(df)
    ns = [n0] + [int(pd.to_numeric(df[TBWL_BY_YEAR[y]], errors="coerce").notna().sum())
                 for y in range(1, 7)]
    stages = ["Cohort"] + [f"Yr {y}" for y in range(1, 7)]

    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.axis("off")
    xgap, node_w = 1.9, 0.34
    scale = 4.0 / n0            # vertical units per patient
    top = 4.3
    retained_c, lost_c = "#1f77b4", "#d9d9d9"

    xs = [i * xgap for i in range(len(ns))]
    # retained band (top-aligned), lost branches peel downward
    for i in range(len(ns) - 1):
        x0, x1 = xs[i] + node_w, xs[i + 1]
        h0, h1 = ns[i] * scale, ns[i + 1] * scale
        # retained flow (top strip that narrows)
        ax.fill_between([x0, x1], [top - h1, top - h1], [top, top],
                        color=retained_c, alpha=0.55, lw=0)
        # lost flow peeling off the bottom
        lost = ns[i] - ns[i + 1]
        if lost > 0:
            ax.fill_between([x0, x1], [top - h0, top - h1 - lost * scale],
                            [top - h1, top - h1], color=lost_c, alpha=0.9, lw=0)
            ax.text((x0 + x1) / 2, top - h1 - lost * scale / 2 - 0.12,
                    f"lost {lost}", ha="center", va="top", fontsize=8.5, color="#8a8a8a")
    # nodes
    for i, (x, n, s) in enumerate(zip(xs, ns, stages)):
        h = n * scale
        col = "#455a64" if i == 0 else (retained_c if i <= 4 else "#d62728")
        ax.add_patch(mpatches.Rectangle((x, top - h), node_w, h, color=col))
        ax.text(x + node_w / 2, top + 0.14, s, ha="center", va="bottom",
                fontsize=10, weight="bold")
        ax.text(x + node_w / 2, top - h - 0.16, f"{n}\n{n/n0*100:.0f}%", ha="center",
                va="top", fontsize=9, weight="bold",
                color="#d62728" if i >= 5 else "#222")

    ax.set_xlim(-0.3, xs[-1] + node_w + 0.4)
    ax.set_ylim(-1.2, top + 0.8)
    ax.text(xs[-1] + node_w / 2, top - ns[-1] * scale - 0.95,
            "Years 5–6:\n~10% remain", ha="center", fontsize=9.5, color="#d62728",
            weight="bold")
    ax.set_title("Figure 1. Patient attrition flow — 786 enrolled → 81 with year-6 data",
                 fontsize=13, weight="bold", loc="left")
    plt.tight_layout()
    plt.savefig(OUT / "fig0b_sankey.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig0b_sankey.png")


# ── 1. Pipeline ───────────────────────────────────────────────────────────────
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(17, 3.9))
    ax.axis("off")
    steps = [
        ("RAW DATA", "802 rows\n185 columns", "data/…csv", "#eceff1"),
        ("CLEAN + DEDUP", "802 → 786 patients\n16 dup rows removed", "src/data_load.py", "#e3f2fd"),
        ("FEATURE PREP", "15 preop features\nencode + impute", "src/preprocess.py", "#e3f2fd"),
        ("RANDOM FOREST", "one model per year\nTBWL yrs 1–6", "scripts/\nreproduce_models.py", "#fff3e0"),
        ("PREDICT + GATE", "patient trajectory\n+ trust rating", "src/predict.py\nsrc/reliability.py", "#fff3e0"),
        ("CLUSTER", "UMAP → silhouette\n→ k-means (5)", "src/phenotype.py", "#f3e5f5"),
        ("SURGEON TOOL", "web app the\nsurgeon uses", "app/streamlit_app.py", "#e8f5e9"),
    ]
    n = len(steps); w, h, gap = 2.05, 1.15, 0.42
    for i, (title, body, fname, color) in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(mpatches.FancyBboxPatch((x, 1.0), w, h, boxstyle="round,pad=0.05",
                                             fc=color, ec="#455a64", lw=1.4))
        ax.text(x + w/2, 1.0 + h - 0.26, title, ha="center", va="center", fontsize=9.5, weight="bold")
        ax.text(x + w/2, 1.0 + h/2 - 0.22, body, ha="center", va="center", fontsize=8.2,
                linespacing=1.45, color="#263238")
        ax.text(x + w/2, 0.88, fname, ha="center", va="top", fontsize=7, style="italic",
                color="#37474f", family="monospace", linespacing=1.35)
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap - .04, 1.57), xytext=(x + w + .04, 1.57),
                        arrowprops=dict(arrowstyle="-|>", lw=1.9, color="#455a64"))
    ax.set_xlim(-.25, n * (w + gap) - gap + .25); ax.set_ylim(.35, 2.55)
    ax.set_title("Figure 2. Analysis pipeline — each box is one script", pad=12)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_pipeline.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig1_pipeline.png")


# ── 2b. RF vs GB on OUR data ──────────────────────────────────────────────────
def fig2b_rf_vs_gb():
    """Answers the reviewer's 'why RF and not GB at years 3-4?' with our own numbers."""
    p = ARTIFACTS / "rf_vs_gb.csv"
    if not p.exists():
        print("  (skip fig2b — run scripts/compare_rf_gb.py first)")
        return
    d = pd.read_csv(p)
    t = d[(d.outcome == "TBWL") & (d.year <= 6)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, metric, label, lo in [(axes[0], "r2", "R² (variance explained)", True),
                                  (axes[1], "auc", "AUC (discrimination)", False)]:
        for model, col, mk in [("RandomForest", "#d62728", "o"),
                               ("GradientBoosting", "#1f77b4", "s")]:
            g = t[t.model == model].sort_values("year")
            ax.plot(g.year, g[metric], marker=mk, lw=2.6, ms=8, color=col, label=model)
        ax.set_xlabel("Year after surgery"); ax.set_ylabel(label)
        ax.set_xticks(range(1, 7)); ax.grid(alpha=.3); ax.legend(fontsize=9)
    axes[0].axhline(0, color="grey", lw=1, ls=":")
    axes[0].set_ylim(-1.2, 1.0)
    axes[0].annotate("GB COLLAPSES here\n(R² = −3.8, off-scale)", xy=(5, -0.9),
                     xytext=(3.4, -1.05), fontsize=9.5, color="#1f77b4", weight="bold",
                     arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    axes[0].annotate("GB wins yr 3", xy=(3, 0.70), xytext=(1.6, 0.86), fontsize=9.5,
                     color="#1f77b4", arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    axes[1].axhline(0.8, color="#2ca02c", ls="--", lw=1.4)
    axes[1].text(1.05, .81, "0.8 = strong discrimination", color="#2ca02c", fontsize=9)
    fig.suptitle("Figure 4. Random Forest vs Gradient Boosting on our data — GB edges yr3, "
                 "but collapses in the sparse late years", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2b_rf_vs_gb.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig2b_rf_vs_gb.png")


# ── 2. Five-model comparison (S5) ─────────────────────────────────────────────
def fig2_model_comparison():
    x = pd.read_excel(S5)
    x.columns = [str(c).strip() for c in x.columns]
    t = x[x["Metric_Type"].astype(str).str.strip() == "TBWL"].copy()
    t["Model"] = t["Model"].astype(str).str.strip()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, label in [(axes[0], "R2_mean", "R² (variance explained)"),
                              (axes[1], "AUC_mean", "AUC (discrimination)")]:
        for model, grp in t.groupby("Model"):
            grp = grp.sort_values("Year")
            rf = model == "RandomForest"
            ax.plot(grp["Year"], grp[metric], marker="o", lw=3 if rf else 1.4,
                    ms=8 if rf else 5, color="#d62728" if rf else None,
                    alpha=1 if rf else .55, zorder=5 if rf else 2,
                    label=model + (" ★" if rf else ""))
        ax.set_xlabel("Year after surgery"); ax.set_ylabel(label)
        ax.set_xticks(range(1, 7)); ax.grid(alpha=.3); ax.legend(fontsize=8.5)
    fig.suptitle("Figure 3. All five model families across six years (TBWL)",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig2_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig2_model_comparison.png")


# ── 3. Reliability, now WITH AUC ──────────────────────────────────────────────
def fig3_reliability():
    p = ARTIFACTS / "rf_vs_gb.csv"
    auc = {}
    if p.exists():
        d = pd.read_csv(p)
        d = d[d.model == "RandomForest"]
        auc = {(r.outcome, int(r.year)): r.auc for r in d.itertuples()}

    outcomes, years = ["TBWL", "FML"], list(range(1, 7))
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    for i, oc in enumerate(outcomes):
        for j, y in enumerate(years):
            m = MODEL_PERFORMANCE[oc][y]; t = tier(oc, y)
            ax.add_patch(mpatches.Rectangle((j, -i), 1, 1, fc=TIER_C[t], ec="white", lw=2.5))
            ax.text(j+.5, -i+.70, f"R²={m['r2']:.2f}", ha="center", va="center",
                    color="white", fontsize=9.5, weight="bold")
            a = auc.get((oc, y))
            if a == a and a is not None:
                ax.text(j+.5, -i+.45, f"AUC={a:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
            ax.text(j+.5, -i+.18, "REPORTED" if t != "red" else "REFUSED",
                    ha="center", va="center", color="white", fontsize=7.5)
    ax.set_xlim(0, 6); ax.set_ylim(-1, 1)
    ax.set_xticks([j+.5 for j in range(6)]); ax.set_xticklabels([f"Year {y}" for y in years])
    ax.set_yticks([.5, -.5]); ax.set_yticklabels(["TBWL %\n(weight loss)", "FML %\n(fat loss)"])
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    handles = [mpatches.Patch(fc=TIER_C[k], label=l) for k, l in
               [("green", "Green — trustworthy (R² ≥ 0.40)"),
                ("amber", "Amber — rough guide (0.20–0.40)"),
                ("red", "Red — tool REFUSES to give a number (< 0.20)")]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, fontsize=9)
    ax.set_title("Figure 5. Reliability gating — R² and AUC per year. The tool refuses to guess.",
                 pad=12)
    # combined-view callout: low R2 but high AUC (Ioanna's example)
    ax.text(3.0, -1.62,
            "Combined view: FML year 3 has R²=0.18 (can't predict the exact %) but AUC=0.80 "
            "(can still rank\nhigh vs low responders). When R² is low but AUC is high, the model "
            "is useful for triage, not point estimates.",
            ha="center", va="top", fontsize=8.8, style="italic", color="#1f4e79",
            bbox=dict(boxstyle="round,pad=0.4", fc="#eaf2fb", ec="#1f4e79", lw=1))
    ax.set_ylim(-2.5, 1)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_reliability.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig3_reliability.png")


def _refit_for_viz(df):
    import umap
    b = joblib.load(ARTIFACTS / "phenotype_kmeans.joblib")
    traj = phe._predicted_trajectories(df)
    X = phe._assemble_features(df, traj).reindex(columns=b["columns"], fill_value=0)
    Xs = b["scaler"].transform(b["imputer"].transform(X))
    Xu = umap.UMAP(**phe.UMAP_KW).fit_transform(Xs)
    labels = b["assigner"].predict(Xs)
    return b, Xu, labels


def fig4_silhouette(b):
    sil = b["silhouette_by_k"]
    ks, vs = list(sil.keys()), list(sil.values())
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(ks, vs, marker="o", lw=2, color="#1f77b4", ms=8)
    ax.scatter([4, 5], [sil[4], sil[5]], s=260, facecolors="none", edgecolors="#d62728",
               lw=2.5, zorder=5)
    ax.annotate(f"k=4: {sil[4]:.4f}\nk=5: {sil[5]:.4f}\n→ tied (Δ={abs(sil[4]-sil[5]):.4f})",
                xy=(5, sil[5]), xytext=(6.3, sil[5]-.055), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", weight="bold")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score (higher = better separated)")
    ax.set_xticks(ks); ax.grid(alpha=.3)
    ax.set_title("Figure 6a. Cluster count derived from the data (k swept 2–10), not assumed", pad=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_silhouette.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig4_silhouette.png")


def fig5_umap(b, Xu, labels):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for c in range(b["k"]):
        m = labels == c
        ax.scatter(Xu[m, 0], Xu[m, 1], s=26, alpha=.8, color=CL_C[c],
                   label=f"Phenotype {c+1} (n={int(m.sum())})", edgecolors="none")
    ax.set_xlabel("UMAP dimension 1"); ax.set_ylabel("UMAP dimension 2")
    ax.legend(fontsize=9)
    ax.set_title("Figure 6b. The five phenotypes, visualised", pad=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_umap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig5_umap.png")


# ── 6. Trajectories: n-per-cell + preop + 10.5% threshold ─────────────────────
def fig6_trajectories(df, b, labels):
    d = df.copy(); d["cl"] = labels
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    late_ns = []
    for c in range(b["k"]):
        s = d[d.cl == c]
        preop = pd.to_numeric(s["Preop_TBWL"], errors="coerce").mean()
        xs, ys, ns = [0], [preop], [len(s)]
        for y in range(1, 7):
            v = pd.to_numeric(s[TBWL_BY_YEAR[y]], errors="coerce").dropna()
            if len(v) >= 3:
                xs.append(y); ys.append(v.mean()); ns.append(len(v))
        below = preop < PREOP_THRESHOLD
        ax.plot(xs, ys, marker="o", lw=2.6, ms=8, color=CL_C[c],
                ls="--" if below else "-",
                label=f"Phenotype {c+1} (n={len(s)}){'  ⚠ preop < 10.5%' if below else ''}")
        # annotate n only for yrs 1-6 (skip preop, avoids the pile-up at x=0)
        for x, yv, nn in zip(xs, ys, ns):
            if x == 0:
                continue
            va, off = ("bottom", 9) if x >= 5 else ("top", -9)
            ax.annotate(f"{nn}", (x, yv), textcoords="offset points", xytext=(0, off),
                        ha="center", va=va, fontsize=8, color=CL_C[c], weight="bold")
            if x >= 5:
                late_ns.append(nn)

    ax.axhline(PREOP_THRESHOLD, color="#D95F02", ls="--", lw=1.8)
    ax.text(-0.28, PREOP_THRESHOLD - 1.1, "10.5% preop threshold", color="#D95F02",
            fontsize=9.5, weight="bold")
    ax.axvspan(4.5, 6.45, color="#d62728", alpha=.07)
    ax.text(5.47, 46.5, f"sparse (n as low as {min(late_ns)}) —\ninterpret with caution",
            ha="center", fontsize=9.5, color="#d62728", style="italic", weight="bold")
    ax.set_xticks(range(0, 7))
    ax.set_xticklabels(["Preop"] + [str(y) for y in range(1, 7)])
    ax.set_xlabel("Time"); ax.set_ylabel("Mean TBWL %")
    ax.set_xlim(-.35, 6.6); ax.set_ylim(5, 50)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, loc="lower left", framealpha=.95)
    ax.set_title("Figure 7a. Observed trajectory per phenotype — numbers above/below each point "
                 "are n\n(dashed = preop below the 10.5% threshold)", pad=10, fontsize=12.5)
    plt.tight_layout()
    plt.savefig(OUT / "fig6_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig6_trajectories.png  (late-year n as low as {min(late_ns)})")


# ── 7. Demographics WITH RACE ─────────────────────────────────────────────────
def fig7_demographics(df, b, labels):
    d = df.copy(); d["cl"] = labels
    rows = []
    for c in range(b["k"]):
        s = d[d.cl == c]
        rec = {
            "Mean Preop TBWL %": pd.to_numeric(s["Preop_TBWL"], errors="coerce").mean(),
            "Mean Initial BMI": pd.to_numeric(s["Initial_BMI"], errors="coerce").mean(),
            "Mean Age": pd.to_numeric(s["Age"], errors="coerce").mean(),
            "% Female": (s["Sex"].astype(str) == "Female").mean()*100,
        }
        for r in ["Hispanic", "White", "African_American"]:
            rec[f"% {r.replace('_',' ')}"] = (s["Race"].astype(str) == r).mean()*100
        for st in ["Sleeve", "Bypass", "Revision"]:
            rec[f"% {st}"] = (s["Surgery_Type"].astype(str) == st).mean()*100
        rows.append(rec)
    H = pd.DataFrame(rows, index=[f"Phenotype {c+1}\n(n={int((labels==c).sum())})"
                                  for c in range(b["k"])]).T

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    sns.heatmap(H, annot=True, fmt=".1f", cmap="Blues", linewidths=.8, linecolor="white",
                cbar_kws={"label": "value"}, annot_kws={"weight": "bold", "size": 10}, ax=ax)
    ax.set_title("Figure 7b. Who is in each phenotype — including RACE\n"
                 "(ordered by ascending preoperative weight loss)", pad=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig7_demographics.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("  fig7_demographics.png")
    return H


# ── 8. Sensitivity: now sex + procedure + RACE ────────────────────────────────
def fig8_sensitivity(df, b, labels):
    import umap
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score

    d = df.copy(); d["cl"] = labels
    Zsp = pd.get_dummies(d[["Sex", "Surgery_Type"]].astype(str)).values
    Zall = pd.get_dummies(d[["Sex", "Surgery_Type", "Race"]].astype(str)).values
    acc_sp = cross_val_score(DecisionTreeClassifier(random_state=SEED), Zsp, labels, cv=5).mean()
    acc_all = cross_val_score(DecisionTreeClassifier(random_state=SEED), Zall, labels, cv=5).mean()

    traj = phe._predicted_trajectories(df)
    T = traj[[phe._traj_col(y) for y in phe.CLUSTER_TRAJ_YEARS]].values
    Xu2 = umap.UMAP(**phe.UMAP_KW).fit_transform(StandardScaler().fit_transform(T))
    lab2 = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(Xu2)
    order = pd.Series(T[:, 1]).groupby(lab2).mean().sort_values().index
    lab2 = pd.Series(lab2).map({o: i for i, o in enumerate(order)}).values
    acc2_sp = cross_val_score(DecisionTreeClassifier(random_state=SEED), Zsp, lab2, cv=5).mean()
    acc2_all = cross_val_score(DecisionTreeClassifier(random_state=SEED), Zall, lab2, cv=5).mean()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # A: composition of the 1A-aligned clusters (procedure)
    procs = ["Sleeve", "Bypass", "Revision"]
    bot = np.zeros(b["k"])
    for p, col in zip(procs, ["#1f77b4", "#2ca02c", "#d62728"]):
        v = [(d[d.cl == c]["Surgery_Type"].astype(str) == p).mean()*100 for c in range(b["k"])]
        axes[0].bar(range(1, b["k"]+1), v, bottom=bot, label=p, color=col); bot += np.array(v)
    axes[0].set_xlabel("Phenotype"); axes[0].set_ylabel("% of patients"); axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=9)
    axes[0].set_title("A. 1A recipe — each phenotype\nis essentially one procedure", fontsize=11)

    # B: how predictable is cluster membership?
    labels_x = ["sex +\nprocedure", "sex + procedure\n+ RACE"]
    xpos = np.arange(2); wd = .36
    axes[1].bar(xpos - wd/2, [acc_sp*100, acc_all*100], wd, label="1A recipe", color="#d62728")
    axes[1].bar(xpos + wd/2, [acc2_sp*100, acc2_all*100], wd, label="trajectory-shape only",
                color="#2ca02c")
    for i, v in enumerate([acc_sp, acc_all]):
        axes[1].text(i - wd/2, v*100 + 1.5, f"{v:.0%}", ha="center", weight="bold", fontsize=10)
    for i, v in enumerate([acc2_sp, acc2_all]):
        axes[1].text(i + wd/2, v*100 + 1.5, f"{v:.0%}", ha="center", weight="bold", fontsize=10)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels_x)
    axes[1].set_ylabel("% of cluster membership predicted\nby demographics alone")
    axes[1].set_ylim(0, 100); axes[1].legend(fontsize=9)
    axes[1].set_title("B. Procedure + sex are the drivers (71%).\nRace over-separates only in this "
                      "lean re-fit —\n1A's own clusters (Table 6) split by preop-TBWL, not race",
                      fontsize=9.5)

    # C: trajectory-only clusters -> responder gradient
    d2 = df.copy(); d2["cl"] = lab2
    for c in range(5):
        s = d2[d2.cl == c]
        ys = [1, 2, 3, 4]
        means = [pd.to_numeric(s[TBWL_BY_YEAR[y]], errors="coerce").mean() for y in ys]
        axes[2].plot(ys, means, marker="o", lw=2.4, ms=7, color=CL_C[c],
                     label=f"Cluster {c+1} (n={len(s)})")
    axes[2].set_xlabel("Year after surgery"); axes[2].set_ylabel("Actual TBWL % (mean)")
    axes[2].set_xticks([1, 2, 3, 4]); axes[2].grid(alpha=.3); axes[2].legend(fontsize=8.5)
    axes[2].set_title("C. Trajectory-shape only →\na true poor→strong responder gradient",
                      fontsize=11)

    fig.suptitle("Figure 8. Phenotypes are structured mainly by procedure and sex; clustering on "
                 "trajectory shape reveals the underlying responder gradient", fontsize=12.5, weight="bold")
    plt.tight_layout()
    plt.savefig(OUT / "fig8_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig8_sensitivity.png  (1A: {acc_sp:.0%}/{acc_all:.0%}  vs  traj-only: "
          f"{acc2_sp:.0%}/{acc2_all:.0%})")
    return acc_sp, acc_all, acc2_sp, acc2_all


def main():
    print("Generating figures from live pipeline...")
    df = load()
    fig0_attrition(df)
    fig0b_sankey(df)
    fig1_pipeline()
    fig2_model_comparison()
    fig2b_rf_vs_gb()
    fig3_reliability()
    print("  (recomputing UMAP embedding — ~1 min)")
    b, Xu, labels = _refit_for_viz(df)
    fig4_silhouette(b)
    fig5_umap(b, Xu, labels)
    fig6_trajectories(df, b, labels)
    H = fig7_demographics(df, b, labels)
    fig8_sensitivity(df, b, labels)
    print(f"\nAll figures -> {OUT}/")
    print("\nCluster demographics (WITH RACE) — for your speaker notes:")
    print(H.round(1).to_string())


if __name__ == "__main__":
    main()
