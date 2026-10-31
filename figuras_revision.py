#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figures for the revised Schubert manuscript.

Figures that need no new data (rendered directly):
  fig_schema            Fig. 2  - Schubert class of a coordinate subspace as a rank statistic
  fig_diagnostic        Fig. S1 - raw codimension vs exchange-interface size
                                  (values transcribed from Tables 5 and 7 of the
                                  submitted version)

Figures that need the user's data:
  fig_nu_species        Fig. 4  - normalized codimension nu and dimension d per species
  fig_h1a               Fig. 5  - H1a before/after removing the d dependence
  fig_bottleneck        Fig. 7  - H1 bottleneck distances (clustered)

Usage
  python figuras_revision.py --static --out figs/
  python figuras_revision.py --muestras muestras.csv --tolerancia tolerancia.csv \
        --bottleneck bottleneck.csv --out figs/
  python figuras_revision.py --demo --out figs_demo/     # illustrative data, watermarked

muestras.csv   : especie, escenario, codim, d   (or a 'lambda' column -> --col-lambda)
tolerancia.csv : especie, tolerancia
bottleneck.csv : square matrix, first column = species, header = species

Species names must match the keys of TIER below (abbreviated form, e.g. "Sal. ruber").
Python >= 3.9.
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

M_DEFAULT = 552

# ----------------------------------------------------------------------
# Tiers (Section 4.2 of the manuscript) and an ordinal blue ramp:
# light -> dark follows the salinity ordering T0 < T1 < T2.
# ----------------------------------------------------------------------
TIER = {
    "Ros. halotolerans": "T0", "Gra. halophila": "T0",
    "Hrr. salinarum": "T1", "Nmn. salina": "T1", "Spb. salinus": "T1",
    "Ocn. granulosus": "T1", "Arh. aquaeolei": "T1", "Pcc. vermicola": "T1",
    "Hqr. walsbyi": "T2", "Sal. ruber": "T2", "Hmc. mukohataei": "T2",
}
TIER_COLOR = {"T0": "#6da7ec", "T1": "#2a78d6", "T2": "#104281"}
TIER_LABEL = {"T0": "$T_0$ (SS13 only)", "T1": "$T_1$ (decreasing)",
              "T2": "$T_2$ (increasing)"}
NEUTRAL = "#8a8986"
INK = "#0b0b0b"
INK2 = "#52514e"
NULL_COLOR = "#e34948"
SCEN_MARKER = {"SS19": "o", "SS37": "s"}

MM = 1 / 25.4
W2 = 174 * MM          # double column
W1 = 84 * MM           # single column


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e1",
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 600,
    })


def panel_label(ax, letter, dx=-26, dy=6):
    """(A), (B)... outside the axes, upper-left corner."""
    ax.annotate("(%s)" % letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(dx, dy), textcoords="offset points",
                fontsize=9, fontweight="bold", color=INK,
                ha="left", va="bottom", annotation_clip=False)


def watermark(fig):
    fig.text(0.5, 0.5, "ILLUSTRATIVE DATA - NOT RESULTS", rotation=25,
             fontsize=26, color="#e34948", alpha=0.18, ha="center",
             va="center", fontweight="bold", zorder=100)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "%s.%s" % (name, ext)),
                    bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def species_order(names):
    order = {"T0": 0, "T1": 1, "T2": 2}
    return sorted(names, key=lambda s: (order.get(TIER.get(s, "T9"), 9), s))


def nu_z(codim, d, m):
    codim = np.asarray(codim, float)
    d = np.asarray(d, float)
    rect = d * (m - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(rect > 0, codim / rect, np.nan)
        z = np.where(rect > 0, (codim - rect / 2) / np.sqrt(rect * (m + 1) / 12), np.nan)
    return nu, z


def spearman_perm(x, y, n_mc=100000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = len(x)
    rx = stats.rankdata(x) - (n + 1) / 2
    ry = stats.rankdata(y) - (n + 1) / 2
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    rho = (rx * ry).sum() / den
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_mc, n)), axis=1)
    null = (ry[idx] * rx).sum(axis=1) / den
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean()), n


def italic(s):
    # species names are drawn as plain text with fontstyle="italic"
    return s


def fmt_p(p):
    return "$p_{\\mathrm{perm}}<0.001$" if p < 0.001 else "$p_{\\mathrm{perm}}=%.3f$" % p


# ----------------------------------------------------------------------
# Fig. 2 - schema
# ----------------------------------------------------------------------
def fig_schema(out):
    m, active = 16, [2, 5, 6, 11, 15]
    d = len(active)
    tiers = ["T0"] * 3 + ["T1"] * 6 + ["T2"] * 4 + [None] * 3
    lam = [(m - d) + l - p for l, p in enumerate(active, start=1)]
    codim = sum(lam)

    fig = plt.figure(figsize=(W2, 2.75))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1.6],
                          width_ratios=[1.2, 1.0, 1.05],
                          hspace=0.55, wspace=0.45)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[1, 2])

    # (A) flag order with tiers and active reactions
    axA.set_xlim(0.3, m + 0.7)
    axA.set_ylim(-1.0, 0.95)
    axA.axis("off")
    for j in range(1, m + 1):
        t = tiers[j - 1]
        edge = TIER_COLOR[t] if t else NEUTRAL
        is_act = j in active
        axA.add_patch(Rectangle((j - 0.42, 0), 0.84, 0.84,
                                facecolor=edge if is_act else "white",
                                edgecolor=edge, linewidth=1.1))
        axA.text(j, 0.42, "%d" % j, ha="center", va="center", fontsize=6,
                 color="white" if is_act else INK2)
    spans = [("T0", 1, 3), ("T1", 4, 9), ("T2", 10, 13), (None, 14, 16)]
    for t, a, b in spans:
        c = TIER_COLOR[t] if t else NEUTRAL
        axA.plot([a - 0.4, b + 0.4], [-0.18, -0.18], color=c, lw=1.6,
                 solid_capstyle="round")
        axA.text((a + b) / 2, -0.55,
                 {"T0": "$T_0$", "T1": "$T_1$", "T2": "$T_2$", None: "unassigned"}[t],
                 ha="center", va="center", fontsize=7, color=INK)
    axA.annotate("", xy=(m + 0.6, 0.42), xytext=(m + 0.48, 0.42),
                 arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7))
    axA.text(m / 2 + 0.5, 0.92, "flag order $F_1\\subset F_2\\subset\\cdots\\subset F_m$;"
             "  filled squares = active exchange reactions $R(\\varepsilon)$",
             ha="center", va="bottom", fontsize=6.6, color=INK2)
    panel_label(axA, "A", dx=-22, dy=-12)

    # (B) delta_j staircase
    j = np.arange(0, m + 1)
    delta = np.array([sum(1 for p in active if p <= k) for k in j])
    axB.step(j, delta, where="post", color=TIER_COLOR["T1"], lw=1.4)
    for l, p in enumerate(active, start=1):
        axB.plot(p, l, "o", ms=4, color=TIER_COLOR["T2"], mec="white", mew=0.8)
        axB.annotate("$p_%d=%d$" % (l, p), (p, l), xytext=(3, -9),
                     textcoords="offset points", fontsize=6, color=INK2)
    axB.set_xlabel("flag step $j$")
    axB.set_ylabel(r"$\delta_j=\dim(P\cap F_j)$")
    axB.set_xlim(0, m)
    axB.set_ylim(0, d + 0.6)
    axB.set_xticks([0, 4, 8, 12, 16])
    axB.set_yticks(range(0, d + 1))
    panel_label(axB, "B")

    # (C) lambda_l = inactive reactions after p_l
    ys = np.arange(1, d + 1)
    axC.barh(ys, lam, height=0.55, color=TIER_COLOR["T1"], edgecolor="white")
    for y, v in zip(ys, lam):
        axC.text(v + 0.25, y, str(v), va="center", fontsize=6.5, color=INK)
    axC.set_yticks(ys)
    axC.set_yticklabels(["$\\lambda_%d$" % l for l in ys])
    axC.invert_yaxis()
    axC.set_xlim(0, m - d + 0.8)
    axC.set_xlabel("inactive reactions placed after $p_l$")
    axC.grid(axis="y", visible=False)
    panel_label(axC, "C")

    # (D) Young diagram in d x (m-d)
    axD.set_xlim(-0.2, m - d + 0.2)
    axD.set_ylim(d + 0.2, -0.2)
    axD.set_aspect("equal")
    axD.axis("off")
    for r in range(d):
        for c in range(m - d):
            filled = c < lam[r]
            axD.add_patch(Rectangle((c, r), 1, 1,
                                    facecolor=TIER_COLOR["T1"] if filled else "white",
                                    edgecolor="white" if filled else "#d6d5d0",
                                    linewidth=0.6))
    axD.add_patch(Rectangle((0, 0), m - d, d, fill=False, edgecolor=INK2, lw=0.7))
    axD.text((m - d) / 2, d + 0.7,
             "$|\\lambda|=%d$ = Mann-Whitney $U$\n$\\nu=|\\lambda|/[d(m-d)]=%.2f$"
             % (codim, codim / (d * (m - d))),
             ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.5)
    axD.text((m - d) / 2, -0.55, "$d\\times(m-d)=%d\\times%d$ rectangle" % (d, m - d),
             ha="center", va="bottom", fontsize=6.3, color=INK2)
    axD.annotate("(D)", xy=(0, 1), xycoords=(axD.transAxes, axB.transAxes),
                 xytext=(-14, 6), textcoords="offset points", fontsize=9,
                 fontweight="bold", color=INK, ha="left", va="bottom",
                 annotation_clip=False)

    save(fig, out, "fig2_schema_rank_statistic")


# ----------------------------------------------------------------------
# Fig. S1 - diagnostic from the submitted Tables 5 and 7
# ----------------------------------------------------------------------
SUBMITTED = pd.DataFrame({
    "especie": ["Hrr. salinarum", "Nmn. salina", "Spb. salinus", "Ocn. granulosus",
                "Arh. aquaeolei", "Pcc. vermicola", "Hqr. walsbyi", "Sal. ruber",
                "Hmc. mukohataei"],
    "interfaz": [132, 105, 9, 7, 21, 9, 94, 165, 7],
    "codim": [6501.61, 6347.83, 2578.62, 2133.95, 3967.00, 3060.66, 6351.70,
              8172.48, 2138.17],
})


def fig_diagnostic(out):
    df = SUBMITTED
    rho, p, n = spearman_perm(df["interfaz"], df["codim"])
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    for t in ("T1", "T2"):
        s = df[df["especie"].map(TIER) == t]
        ax.scatter(s["interfaz"], s["codim"], s=30, color=TIER_COLOR[t],
                   edgecolor="white", linewidth=0.8, zorder=3, label=TIER_LABEL[t])
    place = {"Sal. ruber": (-6, 0, "right", "center"),
             "Hrr. salinarum": (0, 7, "center", "bottom"),
             "Nmn. salina": (2, -8, "center", "top"),
             "Hqr. walsbyi": (-6, 0, "right", "center"),
             "Arh. aquaeolei": (6, 0, "left", "center"),
             "Pcc. vermicola": (6, 0, "left", "center"),
             "Spb. salinus": (6, 0, "left", "center")}
    for _, r in df.iterrows():
        if r["especie"] not in place:
            continue
        dx, dy, ha, va = place[r["especie"]]
        ax.annotate(r["especie"], (r["interfaz"], r["codim"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6, color=INK,
                    ha=ha, va=va, fontstyle="italic")
    # Ocn. granulosus and Hmc. mukohataei coincide (d = 7, codim ~ 2135)
    ax.annotate("Ocn. granulosus,\nHmc. mukohataei", (7, 2136), xytext=(6, -1),
                textcoords="offset points", fontsize=6, color=INK, ha="left",
                va="center", fontstyle="italic", linespacing=1.1)
    ax.set_xscale("log")
    ax.set_xlim(5, 260)
    ax.set_xlabel("Exchange-interface size (reactions, log scale)")
    ax.set_ylabel("Mean Schubert codimension $|\\lambda|$")
    ax.text(0.03, 0.97, "Spearman $\\rho=%.2f$\n%s ($n=%d$)" % (rho, fmt_p(p), n),
            transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK,
            linespacing=1.4)
    ax.set_ylim(1500, 8800)
    ax.legend(loc="lower right", handletextpad=0.3)
    save(fig, out, "figS1_codim_vs_interface_submitted")
    return rho, p


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_samples(path, col_lambda=None, m=M_DEFAULT):
    df = pd.read_csv(path)
    if col_lambda:
        lam = df[col_lambda].map(lambda s: [int(v) for v in re.findall(r"-?\d+", str(s))])
        df["d"] = lam.map(len)
        df["codim"] = lam.map(sum)
    df = df[(df["d"] > 0) & (df["d"] < m)].copy()
    df["nu"], df["z"] = nu_z(df["codim"], df["d"], m)
    return df


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (B, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


# ----------------------------------------------------------------------
# Fig. 4 - nu and d per species and scenario
# ----------------------------------------------------------------------
def fig_nu_species(df, out, demo=False):
    sp = species_order(df["especie"].unique())
    scen = [s for s in ("SS19", "SS37") if s in set(df["escenario"])] or sorted(df["escenario"].unique())
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(W2, 0.27 * len(sp) + 0.9),
                                   sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.12))
    y0 = np.arange(len(sp))
    off = {s: (k - (len(scen) - 1) / 2) * 0.22 for k, s in enumerate(scen)}
    for i, s in enumerate(sp):
        c = TIER_COLOR.get(TIER.get(s), NEUTRAL)
        for sc in scen:
            x = df[(df["especie"] == s) & (df["escenario"] == sc)]
            if x.empty:
                continue
            y = i + off[sc]
            lo, hi = boot_ci(x["nu"])
            axA.plot([lo, hi], [y, y], color=c, lw=1.2, solid_capstyle="round")
            mk = SCEN_MARKER.get(sc, "o")
            filled = sc == scen[-1]
            axA.plot(x["nu"].mean(), y, mk, ms=4.5, color=c,
                     mfc=c if filled else "white", mec=c, mew=1.0)
            axB.plot(x["d"].mean(), y, mk, ms=4.5, color=c,
                     mfc=c if filled else "white", mec=c, mew=1.0)
    axA.axvline(0.5, color=NULL_COLOR, lw=0.8, ls=(0, (3, 2)))
    axA.text(0.5, -0.9, " random-flag expectation", color=NULL_COLOR,
             fontsize=6, va="center", ha="left")
    axA.set_yticks(y0)
    axA.set_yticklabels(sp, fontstyle="italic")
    axA.invert_yaxis()
    axA.set_xlabel("Normalized codimension $\\nu$ (mean, 95% bootstrap CI)")
    axB.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FormatStrFormatter
    axB.xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500]))
    axB.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    axB.xaxis.set_minor_locator(NullLocator())
    lo_d, hi_d = df["d"].min(), df["d"].max()
    axB.set_xlim(lo_d / 1.4, hi_d * 1.4)
    axB.set_xlabel("Mean subspace dimension $d$ (log)")
    for ax in (axA, axB):
        ax.grid(axis="y", visible=False)
    # tier separators
    tiers = [TIER.get(s) for s in sp]
    for i in range(1, len(sp)):
        if tiers[i] != tiers[i - 1]:
            for ax in (axA, axB):
                ax.axhline(i - 0.5, color="#d6d5d0", lw=0.6)
    handles = [plt.Line2D([], [], ls="", marker="s", ms=6, color=TIER_COLOR[t],
                          label=TIER_LABEL[t]) for t in ("T0", "T1", "T2")]
    handles += [plt.Line2D([], [], ls="", marker=SCEN_MARKER.get(sc, "o"), ms=4.5,
                           color=INK2, mfc=INK2 if sc == scen[-1] else "white",
                           label="scenario %s" % sc) for sc in scen]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.2, columnspacing=1.2)
    panel_label(axA, "A", dx=-80, dy=4)
    panel_label(axB, "B", dx=-10, dy=4)
    if demo:
        watermark(fig)
    save(fig, out, "fig4_normalized_codimension" + ("_DEMO" if demo else ""))


# ----------------------------------------------------------------------
# Fig. 5 - H1a, raw vs normalized
# ----------------------------------------------------------------------
def fig_h1a(df, tol, m, out, demo=False):
    g = df.groupby("especie").agg(d=("d", "mean"), codim=("codim", "mean"),
                                  nu=("nu", "mean")).reset_index()
    g = g.merge(tol, on="especie", how="inner")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.8),
                             gridspec_kw=dict(wspace=0.45))
    a = axes[0]
    dd = np.linspace(1, df["d"].max() * 1.05, 300)
    mu = dd * (m - dd) / 2
    sd = np.sqrt(dd * (m - dd) * (m + 1) / 12)
    a.fill_between(dd, mu - 2 * sd, mu + 2 * sd, color=NULL_COLOR, alpha=0.13, lw=0)
    a.plot(dd, mu, color=NULL_COLOR, lw=1.0, label="random flag (mean $\\pm$2 SD)")
    a.plot(dd, dd * (m - dd), color=INK2, lw=0.7, ls=(0, (3, 2)), label="maximum $d(m-d)$")
    for t in ("T0", "T1", "T2"):
        s = df[df["especie"].map(TIER) == t]
        if len(s):
            a.scatter(s["d"], s["codim"], s=5, color=TIER_COLOR[t], alpha=0.35,
                      lw=0, label=TIER_LABEL[t])
    a.set_xlabel("Subspace dimension $d$")
    a.set_ylabel("Schubert codimension $|\\lambda|$")
    leg = a.legend(loc="upper left", markerscale=2.0, handlelength=1.2, fontsize=5.6, borderaxespad=0.2)
    for h in leg.legend_handles:
        h.set_alpha(1)
    panel_label(a, "A", dy=15)

    for ax, col, xl, lab in [(axes[1], "codim", "Mean raw codimension $|\\lambda|$", "B"),
                             (axes[2], "nu", "Mean normalized codimension $\\nu$", "C")]:
        texts = []
        for _, r in g.iterrows():
            ax.scatter(r[col], r["tolerancia"], s=26,
                       color=TIER_COLOR.get(TIER.get(r["especie"]), NEUTRAL),
                       edgecolor="white", lw=0.8, zorder=3)
            texts.append(ax.text(r[col], r["tolerancia"], r["especie"], fontsize=5.3,
                                 color=INK2, fontstyle="italic"))
        ax.margins(x=0.12, y=0.1)
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, expand=(1.3, 1.9), force_text=(0.4, 0.8),
                        arrowprops=dict(arrowstyle="-", color="#b5b4af", lw=0.4))
        except ImportError:
            pass
        rho, p, n = spearman_perm(g[col], g["tolerancia"])
        ax.set_title("$\\rho_S=%.2f$, %s, $n=%d$" % (rho, fmt_p(p), n),
                     loc="left", fontsize=6.3, color=INK, pad=3)
        ax.set_xlabel(xl)
        ax.set_ylabel("NaCl tolerance breadth")
        panel_label(ax, lab, dy=15)
    axes[2].axvline(0.5, color=NULL_COLOR, lw=0.8, ls=(0, (3, 2)))
    if demo:
        watermark(fig)
    save(fig, out, "fig5_h1a_raw_vs_normalized" + ("_DEMO" if demo else ""))


# ----------------------------------------------------------------------
# Fig. 7 - bottleneck heatmap, clustered, lower triangle
# ----------------------------------------------------------------------
def fig_bottleneck(D, out, demo=False):
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from matplotlib.colors import LinearSegmentedColormap
    names = list(D.index)
    Z = linkage(squareform(D.values, checks=False), method="average")
    order = leaves_list(Z)
    D = D.iloc[order, order]
    names = list(D.index)
    n = len(names)
    cmap = LinearSegmentedColormap.from_list(
        "seqblue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    vals = D.values.astype(float)
    sub = vals[1:, :-1]                       # drop empty first row / last column
    mask = np.triu(np.ones_like(sub, bool), k=1)
    show = np.ma.array(sub, mask=mask)
    vmax = float(np.nanmax(sub[~mask]))
    fig, ax = plt.subplots(figsize=(W1 * 1.2, W1 * 1.05))
    im = ax.pcolormesh(show, cmap=cmap, vmin=0, vmax=vmax, edgecolors="white",
                       linewidth=1.0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    for i in range(n - 1):
        for j in range(i + 1):
            v = sub[i, j]
            ax.text(j + 0.5, i + 0.5, "%.0f" % v if vmax >= 100 else "%.2g" % v,
                    ha="center", va="center", fontsize=5.4,
                    color="white" if v > 0.55 * vmax else INK)
    ax.set_xticks(np.arange(n - 1) + 0.5)
    ax.set_yticks(np.arange(n - 1) + 0.5)
    ax.set_xticklabels(names[:-1], rotation=45, ha="right", fontstyle="italic")
    ax.set_yticklabels(names[1:], fontstyle="italic")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Bottleneck distance ($\\mathcal{H}_1$)")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6)
    if demo:
        watermark(fig)
    save(fig, out, "fig7_bottleneck_H1" + ("_DEMO" if demo else ""))


# ----------------------------------------------------------------------
# Demo data (clearly illustrative)
# ----------------------------------------------------------------------
def demo_data(m=M_DEFAULT, seed=3):
    rng = np.random.default_rng(seed)
    dmap = {"Ros. halotolerans": 12, "Gra. halophila": 8, "Hrr. salinarum": 60,
            "Nmn. salina": 48, "Spb. salinus": 7, "Ocn. granulosus": 6,
            "Arh. aquaeolei": 15, "Pcc. vermicola": 7, "Hqr. walsbyi": 45,
            "Sal. ruber": 80, "Hmc. mukohataei": 6}
    rows = []
    for s, d in dmap.items():
        bias = rng.normal(0, 0.6)
        for sc, shift in (("SS19", 0.0), ("SS37", 0.25)):
            w = np.exp(-(bias + shift) * np.linspace(0, 3, m))
            w /= w.sum()
            for _ in range(60):
                dd = max(2, d + int(rng.integers(-3, 4)))
                pos = np.sort(rng.choice(np.arange(1, m + 1), dd, replace=False, p=w))
                c = dd * (m - dd) + dd * (dd + 1) / 2 - pos.sum()
                rows.append(dict(especie=s, escenario=sc, codim=c, d=dd))
    df = pd.DataFrame(rows)
    df["nu"], df["z"] = nu_z(df["codim"], df["d"], m)
    tol = pd.DataFrame(dict(especie=list(dmap), tolerancia=rng.uniform(10, 30, len(dmap)).round(1)))
    sp = [s for s in dmap if TIER[s] != "T0"]
    X = rng.uniform(0, 1, (len(sp), 2))
    Dm = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)) * 9000
    D = pd.DataFrame(Dm, index=sp, columns=sp)
    return df, tol, D


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="figs")
    ap.add_argument("--m", type=int, default=M_DEFAULT)
    ap.add_argument("--static", action="store_true", help="schema + diagnostic only")
    ap.add_argument("--demo", action="store_true", help="illustrative, watermarked previews")
    ap.add_argument("--muestras")
    ap.add_argument("--tolerancia")
    ap.add_argument("--bottleneck")
    ap.add_argument("--col-lambda", default=None)
    a = ap.parse_args()
    style()

    fig_schema(a.out)
    rho, p = fig_diagnostic(a.out)
    print("Fig. S1 (submitted Tables 5/7): Spearman(interface, codim) = %.3f, p = %.4f" % (rho, p))
    if a.static:
        return
    if a.demo:
        df, tol, D = demo_data(a.m)
        fig_nu_species(df, a.out, demo=True)
        fig_h1a(df, tol, a.m, a.out, demo=True)
        fig_bottleneck(D, a.out, demo=True)
        return
    if a.muestras:
        df = load_samples(a.muestras, a.col_lambda, a.m)
        fig_nu_species(df, a.out)
        if a.tolerancia:
            fig_h1a(df, pd.read_csv(a.tolerancia), a.m, a.out)
    if a.bottleneck:
        D = pd.read_csv(a.bottleneck, index_col=0)
        fig_bottleneck(D, a.out)


if __name__ == "__main__":
    main()
