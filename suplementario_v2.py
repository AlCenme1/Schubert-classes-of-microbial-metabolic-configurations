# -*- coding: utf-8 -*-
"""
Builds supplementary_v2.tex (converted to appendices by a_apendice.py) and
Fig. S3 for the curated 17-taxon community (model v2).
Run after analisis.py / figuras_run.py (and after the curation); every table
that depends on results is regenerated from the CSVs, so re-running the
pipeline updates the supplement automatically.

Inputs
  resultados/           (analisis.py, sintetico.py, control_run.py)
  ../curacion/          (curar_modelo.py outputs)
Outputs
  supplementary_v2.tex, figs/figS3_curation.pdf/.png
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figuras_revision as fr

R = "resultados"
C = "curacion_previa"        # curation of the earlier community
V2 = "curacion_v2"           # reconstruction and curation of the inserted models
OUT = "figs"
M = json.load(open(os.path.join(R, "meta.json")))["m"]

ABREV = {"Chromohalobacter": "Chb. salexigens", "Haloarcula": "Har. halophila",
         "Halobacterium": "Hbt. salinarum", "Haloferax": "Hfx. volcanii",
         "Halomonas": "Hal. elongata", "Haloquadratum": "Hqr. walsbyi",
         "Halorubrum": "Hrr. salinarum", "Natronomonas": "Nmn. salina",
         "salinibacter": "Sal. ruber", "Tetragenococcus": "Tet. halophilus",
         "halomicrobium": "Hmc. mukohataei", "Spiribacter": "Spb. salinus",
         "Arhodomonas": "Arh. aquaeolei", "Gracilimonas": "Gra. halophila",
         "Oceanicola": "Ocn. granulosus", "Puniceicoccus": "Pcc. vermicola",
         "Roseavarius": "Ros. halotolerans", "Roseovarius": "Ros. halotolerans",
         "Halomicrobium": "Hmc. mukohataei", "Salinibacter": "Sal. ruber",
         "iSRU780": "Sal. ruber", "iHQW-TK": "Hqr. walsbyi"}
RETENIDAS = ["Chromohalobacter", "Haloarcula", "Halobacterium", "Haloferax", "Halomonas",
             "Halorubrum", "Natronomonas", "Tetragenococcus"]
NUEVAS = ["Arhodomonas", "Gracilimonas", "Oceanicola", "Puniceicoccus", "Roseavarius",
          "Spiribacter", "Halomicrobium", "iSRU780", "iHQW-TK"]
VIEJO = {"Halomicrobium": "halomicrobium", "iSRU780": "salinibacter", "iHQW-TK": "Haloquadratum"}
VALID = list(fr.TIER)
TIER = fr.TIER


def em(s):
    return r"\emph{%s}" % s


def num(x, d=3):
    if pd.isna(x):
        return "--"
    s = ("%." + str(d) + "f") % x
    return "$%s$" % s if x < 0 else s


def pval(p):
    return "$<$0.001" if p < 0.001 else "%.3f" % p


def orden(lista):
    return fr.species_order(list(lista))


# ----------------------------------------------------------------------
# Fig. S3: curation
# ----------------------------------------------------------------------
def fig_curacion():
    """(A) reversible reactions in the earlier draft/CarveMe models vs the inserted
    models; (B) growth in isolation of the inserted models before and after curation."""
    import pickle
    fr.style()
    rev = pd.read_csv(os.path.join(C, "reversibilidad.csv"))
    rev = rev[rev.modelo == "original"].set_index("especie")
    ins = pd.read_csv(os.path.join(V2, "inspeccion_individuales.csv")).set_index("modelo")
    res = pickle.load(open(os.path.join(V2, "curar_indiv2_res.pkl"), "rb"))

    fig, (a, b) = plt.subplots(1, 2, figsize=(fr.W2, 3.0), gridspec_kw=dict(wspace=0.62))
    ords = sorted(NUEVAS, key=lambda k: -rev.loc[VIEJO.get(k, k), "pct_rev"])
    y = np.arange(len(ords))
    for i, k in enumerate(ords):
        a.plot([ins.loc[k, "pct_rev"], rev.loc[VIEJO.get(k, k), "pct_rev"]], [i, i],
               color="#cfcfca", lw=1.4, zorder=1)
    a.scatter([rev.loc[VIEJO.get(k, k), "pct_rev"] for k in ords], y, s=22, color="#eb6834",
              zorder=2, label="earlier model")
    a.scatter([ins.loc[k, "pct_rev"] for k in ords], y, s=22, color=fr.TIER_COLOR["T1"],
              zorder=3, label="inserted model")
    car = rev[rev.grupo == "CarveMe"].pct_rev
    a.axvspan(car.min(), car.max(), color=fr.TIER_COLOR["T0"], alpha=0.15, lw=0)
    a.set_yticks(y)
    a.set_yticklabels([ABREV[k] for k in ords], fontstyle="italic", fontsize=6.3)
    a.invert_yaxis()
    a.set_xlabel("Reversible reactions (%)")
    a.grid(axis="y", visible=False)
    a.legend(loc="lower right", fontsize=6.2, handletextpad=0.2)
    fr.panel_label(a, "A", dx=-82, dy=4)

    piso = 1e-3
    ordg = sorted(NUEVAS, key=lambda k: -max(res[k]["crec_antes"], piso))
    yy = np.arange(len(ordg))
    h = 0.36
    antes = np.array([max(res[k]["crec_antes"], piso) for k in ordg])
    despues = np.array([max(res[k]["crec"], piso) for k in ordg])
    b.barh(yy - h / 2, antes, height=h, color="#eb6834", label="before curation")
    b.barh(yy + h / 2, despues, height=h, color=fr.TIER_COLOR["T1"], label="after curation")
    for i, k in enumerate(ordg):
        sc = ins.loc[k, "crec_sin_C"]
        if sc > 1e-6:
            b.text(antes[i] * 1.15, i - h / 2, "no carbon: %.2f" % sc, va="center",
                   fontsize=5.6, color="#eb6834")
        if res[k]["crec"] <= 1e-6:
            b.text(piso * 1.3, i, "no growth in isolation", va="center", fontsize=5.8,
                   color=fr.INK2, fontstyle="italic")
    b.set_xscale("log")
    b.set_xlim(piso, 60)
    b.set_yticks(yy)
    b.set_yticklabels([ABREV[k] for k in ordg], fontstyle="italic", fontsize=6.3)
    b.invert_yaxis()
    b.set_xlabel("Growth in isolation, M9 + glycerol (h$^{-1}$, log)")
    b.grid(axis="y", visible=False)
    b.legend(loc="lower right", fontsize=6.2)
    fr.panel_label(b, "B", dx=-82, dy=4)
    fr.save(fig, OUT, "figS3_curation")


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def tabla_sintetico():
    s = pd.read_csv(os.path.join(R, "sintetico.csv"), keep_default_na=False)
    filas = []
    for n in sorted(s.n_s.unique()):
        t = s[s.n_s == n].set_index("regimen")
        celdas = []
        for reg in ("positional", "confounded", "null"):
            r = t.loc[reg]
            celdas += ["%.3f" % r.rechazo_raw, "%.3f" % r.rechazo_nu,
                       num(r.rho_raw_mediana, 2), num(r.rho_nu_mediana, 2)]
        filas.append("%d & %s \\\\" % (n, " & ".join(celdas)))
    return "\n".join(filas)


def tabla_descriptores():
    mu = pd.read_csv(os.path.join(R, "muestras_todas.csv"))
    g = mu.groupby("especie").agg(d=("d", "mean"), d_sd=("d", "std"), codim=("codim", "mean"),
                                  nu=("nu", "mean"), nu_sd=("nu", "std"), z=("z", "mean"))
    especies = orden(g.index)
    filas = []
    for s in especies:
        r = g.loc[s]
        t = TIER.get(s, "--")
        filas.append("%s & %s & %.1f (%.1f) & %.0f & %.3f (%.3f) & %.2f \\\\" % (
            em(s), "$%s_%s$" % (t[0], t[1]) if t != "--" else "--", r.d, r.d_sd, r.codim, r.nu,
            r.nu_sd, r.z))
    return "\n".join(filas)


def tabla_correlaciones():
    filas = []
    etiquetas = {"tolerancia ~ codim cruda": r"Raw $|\lambda|$",
                 "tolerancia ~ d (dimensión)": "$d$",
                 "tolerancia ~ AUC = codim/[d(m-d)]": r"$\nu$",
                 "tolerancia ~ z vs bandera aleatoria": "$z$",
                 "tolerancia ~ AUC | d (parcial)": r"$\nu\mid d$ (partial)",
                 "codim cruda ~ d": r"Raw $|\lambda|$ vs $d$"}
    cs = {k: pd.read_csv(os.path.join(R, "paso1_%s" % k, "correlaciones.csv"))
          for k in ("todas", "tex")}
    for prueba, lab in etiquetas.items():
        celdas = []
        for k in ("todas", "tex"):
            for sub in ("todas con tolerancia", "alta confianza"):
                r = cs[k][(cs[k].prueba == prueba) & (cs[k].subconjunto == sub)]
                celdas.append("--" if r.empty else "%s (%s)" % (num(r.rho.iloc[0], 2),
                                                                 pval(r.p_perm.iloc[0])))
        filas.append("%s & %s \\\\" % (lab, " & ".join(celdas)))
    return "\n".join(filas)


def tabla_epsilon():
    e = pd.read_csv(os.path.join(R, "barrido_epsilon.csv"))
    t = e[e.especie.isin(VALID)].groupby(["especie", "eps"])["nu_media"].mean().unstack()
    cab = " & ".join("%g" % x for x in t.columns)
    filas = ["%s & %s \\\\" % (em(s), " & ".join("%.3f" % v for v in t.loc[s].values))
             for s in orden(t.index)]
    return cab, len(t.columns), "\n".join(filas)


def tabla_variantes():
    v = pd.read_csv(os.path.join(R, "nu_por_variante.csv")).set_index("especie")
    filas = []
    for s in orden(v.index):
        r = v.loc[s]
        filas.append("%s & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            em(s), r.original, r.solo_frecuencia, r.invertida, r.permutada, r.aleatoria))
    me = v.mean()
    filas.append(r"\midrule" + "\nMean & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
        me.original, me.solo_frecuencia, me.invertida, me.permutada, me.aleatoria))
    frac = (me.solo_frecuencia - me.aleatoria) / (me.original - me.aleatoria)
    return "\n".join(filas), frac


def tabla_control():
    c = pd.read_csv(os.path.join(R, "regresion_ordinal_control.csv")).set_index("especie")
    reps = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, "control_rep*.csv")))])
    reps.loc[(reps.ic_sup - reps.ic_inf) > 50, "beta"] = np.nan   # quasi-separation
    rg = reps.groupby("especie").beta.agg(["min", "max", "std"])
    filas = []
    for s in orden(c.index):
        r = c.loc[s]
        if pd.isna(r.beta) or s not in rg.index:
            continue
        q = rg.loc[s]
        filas.append("%s & %s & [%s, %s] & %s & %s & [%s, %s] & %.1f \\\\" % (
            em(s), num(r.beta, 2), num(r.ic_inf, 2), num(r.ic_sup, 2), pval(r.p),
            num(r.beta_ctrl, 2), num(q["min"], 2), num(q["max"], 2), r.p_ctrl))
    return "\n".join(filas), reps.rep.nunique()


def tabla_especies_modelo():
    """All 17 taxa: origin, reactions, reversible %, bounds changed, EGC, growth."""
    import pickle
    rev = pd.read_csv(os.path.join(C, "reversibilidad.csv"))
    cu = rev[rev.modelo == "curated"].set_index("especie")
    g0 = pd.read_csv(os.path.join(C, "diag_especies_aisladas.csv")).set_index("especie")
    cam = pd.read_csv(os.path.join(C, "cambios.csv")).groupby("especie").size()
    ins = pd.read_csv(os.path.join(V2, "inspeccion_individuales.csv")).set_index("modelo")
    res = pickle.load(open(os.path.join(V2, "curar_indiv2_res.pkl"), "rb"))
    v2 = pd.read_csv(os.path.join(V2, "v2_crecimiento_especies.csv")).set_index("especie")
    tok = {"Roseavarius": "Roseovarius", "iSRU780": "Salinibacter", "iHQW-TK": "Haloquadratum"}
    filas = []
    for k in RETENIDAS:
        filas.append("%s & CarveMe (retained) & %d & %.1f & %d & %s & %.2f & %.2f \\\\" % (
            em(ABREV[k]), v2.loc[k, "n_rxn"], cu.loc[k, "pct_rev"], cam.get(k, 0),
            "yes" if g0.loc[k, "ATP_sin_entradas"] > 0 else "no",
            max(0.0, v2.loc[k, "crec_medio"]), max(0.0, v2.loc[k, "crec_sin_carbono"])))
    for k in NUEVAS:
        t = tok.get(k, k)
        origen = "curated (%s)" % k if k in ("iSRU780", "iHQW-TK") else "CarveMe (new)"
        egc = ins.loc[k, "EGC"]
        filas.append("%s & %s & %d & %.1f & %d & %s & %.2f & %.2f \\\\" % (
            em(ABREV[k]), origen, v2.loc[t, "n_rxn"], ins.loc[k, "pct_rev"],
            len(res[k]["cambios"]), "no" if egc == "none" else ("PMF" if egc == "PMF" else "%d metabolites" % len(egc.split(","))),
            max(0.0, v2.loc[t, "crec_medio"]), max(0.0, v2.loc[t, "crec_sin_carbono"])))
    return "\n".join(filas)


def tabla_cambios_nuevos():
    import pickle
    res = pickle.load(open(os.path.join(V2, "curar_indiv2_res.pkl"), "rb"))
    filas = []
    for k in NUEVAS:
        for c in res[k]["cambios"]:
            filas.append("%s & \\texttt{%s} & %s & [%g, %g] $\\rightarrow$ [%g, %g] \\\\" % (
                em(ABREV[k]), c["reaccion"], c["regla"], c["antes"][0], c["antes"][1],
                c["despues"][0], c["despues"][1]))
    return "\n".join(filas)


def tabla_cambios_tipo():
    cam = pd.read_csv(os.path.join(C, "cambios.csv"))
    cam = cam[cam.especie.isin(RETENIDAS)]
    out = {}
    for regla, sub in cam.groupby("regla"):
        cnt = sub.base.value_counts()
        out[regla] = ", ".join(r"\texttt{%s} (%d)" % (b.replace("_", r"\_"), n)
                               for b, n in cnt.head(30).items())
        out[regla + "#n"] = len(sub)
        out[regla + "#tipos"] = len(cnt)
    return out


def tabla_ph():
    ph = pd.read_csv(os.path.join(R, "homologia_persistente.csv")).set_index("especie")
    filas = []
    for s in orden(ph.index):
        r = ph.loc[s]
        filas.append("%s & %s & %s & %d & %.1f & %.3f \\\\" % (
            em(s), "{:,.0f}".format(r.persistencia_H0), "{:,.0f}".format(r.persistencia_H1),
            r.n_ciclos_H1, r.d, r.nu))
    return "\n".join(filas)


MET = {"dha": "dihydroxyacetone", "galct__D": "galactitol", "6pgc": "6-phospho-D-gluconate",
       "thr__L": "L-threonine", "lys__L": "L-lysine", "xan": "xanthine", "ura": "uracil",
       "arg__L": "L-arginine", "orn": "ornithine", "hxan": "hypoxanthine",
       "phe__L": "L-phenylalanine", "met__L": "L-methionine", "no3": "nitrate",
       "tyr__L": "L-tyrosine", "gthrd": "glutathione", "trp__L": "L-tryptophan",
       "acald": "acetaldehyde", "gly": "glycine", "ile__L": "L-isoleucine",
       "man1p": "D-mannose 1-phosphate", "s": "sulfur", "h2s": "hydrogen sulfide",
       "etoh": "ethanol", "lac__L": "L-lactate", "ala__L": "L-alanine", "pea": "phenethylamine",
       "pacald": "phenylacetaldehyde", "hom__L": "L-homoserine", "asp__D": "D-aspartate",
       "succ": "succinate", "pac": "phenylacetate", "glcn": "D-gluconate",
       "2dhglcn": "2-dehydro-D-gluconate", "leu__L": "L-leucine", "val__L": "L-valine",
       "pyovd_kt": "pyoverdine", "fe3pyovd_kt": "Fe(III)-pyoverdine", "asn__L": "L-asparagine",
       "sheme": "siroheme", "nh4": "ammonium", "co2": "CO$_2$", "gua": "guanine",
       "ade": "adenine", "csn": "cytosine", "thymd": "thymidine", "uri": "uridine",
       "for": "formate", "ac": "acetate", "pyr": "pyruvate", "glyc3p": "glycerol 3-phosphate"}


def met(m):
    b = m[:-2] if m.endswith("_e") else m
    return MET.get(b, "\\texttt{%s}" % b.replace("_", "\\_"))


def ab(t):
    import pipeline_schubert as ps
    return em(ps.NOMBRES.get(t, t))


def tabla_eco_crecimiento():
    iso = pd.read_csv(os.path.join(V2, "v2_crecimiento_especies.csv")).set_index("especie")
    cre = pd.read_csv(os.path.join(R, "eco_crecimiento.csv")).set_index("especie")
    filas = []
    for t in cre.sort_values("max_en_comunidad", ascending=False).index:
        r = cre.loc[t]
        filas.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
            ab(t), max(0.0, iso.loc[t, "crec_medio"]), r.max_en_comunidad, r.max_al_90,
            max(0.0, r.mu_pfba)))
    return "\n".join(filas)


def tabla_eco_dependencias():
    d = pd.read_csv(os.path.join(R, "eco_dependencias.csv"))
    filas = []
    for _, r in d.iterrows():
        filas.append("%s & %s & %s & %s \\\\" % (
            ab(r.especie), met(r.metabolito), "%.3f" % r.max_libre,
            "infeasible" if pd.isna(r.max_al_90) else "%.3f" % r.max_al_90))
    return "\n".join(filas)


def tabla_eco_hqr():
    h = pd.read_csv(os.path.join(R, "eco_hqr_importa.csv"))
    h = h[~h.en_medio]
    filas = []
    for _, r in h.iterrows():
        don = "; ".join("%s %.3f" % (ab(x.split(":")[0]), float(x.split(":")[1]))
                        for x in str(r.donantes).split(";") if ":" in x)
        filas.append("%s & %.3f & %s \\\\" % (met(r.metabolito), -r.flujo, don))
    return "\n".join(filas)


def tabla_eco_aristas(k=25):
    a = pd.read_csv(os.path.join(R, "intercambio_aristas.csv"))
    medio = {"glyc_e", "o2_e", "nh4_e", "co2_e", "h2o_e", "h_e", "pi_e", "so4_e", "k_e",
             "na1_e", "cl_e", "ca2_e", "mg2_e", "mn2_e", "zn2_e", "cu2_e", "cobalt2_e",
             "ni2_e", "mobd_e", "fe2_e", "fe3_e", "asn__L_e", "sheme_e"}
    a = a[~a.metabolito.isin(medio)]
    g = a.groupby(["donante", "receptor", "metabolito"]).agg(
        frac=("frac_estados", "mean"), flujo=("flujo_medio", "mean")).reset_index()
    g = g.sort_values(["frac", "flujo"], ascending=False).head(k)
    filas = ["%s & %s & %s & %.2f & %.3f \\\\" % (ab(r.donante), ab(r.receptor), met(r.metabolito),
                                                 r.frac, r.flujo) for r in g.itertuples()]
    return "\n".join(filas)


# ----------------------------------------------------------------------
def construir():
    import glob as _g
    fig_curacion()
    eps_cab, eps_n, eps_filas = tabla_epsilon()
    var_filas, frac = tabla_variantes()
    tiene_ctrl = os.path.exists(os.path.join(R, "regresion_ordinal_control.csv"))
    ctrl_filas, n_ctrl = tabla_control() if tiene_ctrl else ("\\multicolumn{7}{c}{pending} \\\\", 0)
    tipos = tabla_cambios_tipo()
    k_ox = "oxidase/oxygenase irreversible"
    k_imb = [k for k in tipos if k.startswith("reversible and imbalanced") and "#" not in k][0]
    tex = TEMPLATE
    rep = {
        "@@SINT@@": tabla_sintetico(),
        "@@DESC@@": tabla_descriptores(),
        "@@CORR@@": tabla_correlaciones(),
        "@@EPSCAB@@": eps_cab, "@@EPSCOLS@@": "c" * eps_n, "@@EPS@@": eps_filas,
        "@@VAR@@": var_filas, "@@FRAC@@": "%.0f" % (100 * frac),
        "@@CTRL@@": ctrl_filas, "@@NCTRL@@": str(n_ctrl),
        "@@ESPMOD@@": tabla_especies_modelo(),
        "@@CAMNUEVOS@@": tabla_cambios_nuevos(),
        "@@OXLIST@@": tipos[k_ox], "@@NOX@@": str(tipos[k_ox + "#n"]),
        "@@IMBLIST@@": tipos[k_imb], "@@NIMB@@": str(tipos[k_imb + "#n"]),
        "@@PH@@": tabla_ph(),
        "@@M@@": str(M),
        "@@ECOCRE@@": tabla_eco_crecimiento(),
        "@@ECODEP@@": tabla_eco_dependencias(),
        "@@ECOHQR@@": tabla_eco_hqr(),
        "@@ECOARI@@": tabla_eco_aristas(),
    }
    for k, x in rep.items():
        tex = tex.replace(k, x)
    open("supplementary_v2.tex", "w").write(tex)
    print("written supplementary_v2.tex")


TEMPLATE = r"""\documentclass[a4paper,11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx,booktabs,amsmath,amssymb,amsthm,longtable,array}
\usepackage[authoryear]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage{algorithm,algpseudocode}
\usepackage[section]{placeins}
\usepackage{float}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\theequation}{S\arabic{equation}}
\newtheorem{proposition}{Proposition}
\renewcommand{\theproposition}{S\arabic{proposition}}
\newtheorem{corollary}[proposition]{Corollary}
\newtheorem{lemma}[proposition]{Lemma}
\theoremstyle{remark}
\newtheorem{remark}[proposition]{Remark}
\setlength{\parskip}{3pt}

\title{Supplementary Material\\[6pt]\large Schubert classes of microbial metabolic configurations: an exact rank-statistic representation, its null model, and what it reveals in a hypersaline community model}
\author{Alex Centeno-Mejía}
\date{}

\begin{document}
\maketitle
\tableofcontents
\vspace{1em}

\noindent All tables in Sections~S5--S7 are generated by \texttt{suplementario.py} from the result files of the analysis pipeline; numbering of main-text items is preceded by ``main''.

%% ======================================================================
\section{Mathematical results}
\label{sec:S-math}

Throughout, $E=\mathrm{span}\{e_1,\dots,e_m\}$, $\sigma$ is a permutation of $\{1,\dots,m\}$ defining the coordinate flag $F_j=\mathrm{span}\{e_{\sigma(1)},\dots,e_{\sigma(j)}\}$, and $P=\mathrm{span}\{e_r:r\in R\}$ with $|R|=d$, $0<d<m$. We write $p_1<\dots<p_d$ for the positions $\sigma^{-1}(R)$, and $\delta_j=\dim(P\cap F_j)$.

\subsection{Well-definedness of Algorithm 1}

\begin{lemma}
\label{lem:delta}
$\delta_0=0$, $\delta_m=d$, and $\delta_{j}-\delta_{j-1}\in\{0,1\}$ for $j=1,\dots,m$. Hence for each $l\in\{1,\dots,d\}$ the index $j_l=\min\{j:\delta_j=l\}$ exists, $j_1<\dots<j_d$, and $\delta_{j_l}-\delta_{j_l-1}=1$.
\end{lemma}
\begin{proof}
For any subspaces, $\dim(P\cap F_j)\le\dim(P\cap F_{j-1})+1$ because $F_j=F_{j-1}\oplus\langle e_{\sigma(j)}\rangle$; monotonicity is immediate. The rest follows from $\delta$ being a nondecreasing integer sequence from $0$ to $d$ with unit steps.
\end{proof}

\begin{proposition}
\label{prop:S-bounds}
The partition returned by Algorithm 1 satisfies $m-d\ge\lambda_1\ge\lambda_2\ge\dots\ge\lambda_d\ge0$, so $\lambda$ fits in the $d\times(m-d)$ rectangle and $0\le|\lambda|\le d(m-d)$.
\end{proposition}
\begin{proof}
By Lemma~\ref{lem:delta}, $j_{l+1}\ge j_l+1$, so $\lambda_{l+1}-\lambda_l=1-(j_{l+1}-j_l)\le0$. Since $j_l\ge l$, $\lambda_l=(m-d)+l-j_l\le m-d$; since $j_d\le m$ and the $d-l$ jumps after $j_l$ need $d-l$ distinct indices in $(j_l,m]$, $j_l\le m-d+l$, i.e.\ $\lambda_l\ge0$.
\end{proof}

\subsection{Closed form and the Mann--Whitney identity}

Proposition~1 of the main text states $\lambda_l=(m-d)+l-p_l$ and $|\lambda|=U$, where $U$ counts pairs (active, inactive) in which the active coordinate precedes the inactive one along $\sigma$. An equivalent expression in terms of ranks is useful computationally. Let $\mathbf a\in\{0,1\}^m$ be the indicator of $R$ and $\boldsymbol\rho\in\{1,\dots,m\}^m$ the rank of each coordinate along $\sigma$ ($\rho_r=\sigma^{-1}(r)$). Then
\begin{equation}
|\lambda|=d(m-d)+\frac{d(d+1)}{2}-\mathbf a^{\top}\boldsymbol\rho ,
\label{eq:matrix}
\end{equation}
which is linear in $\mathbf a$: for $N$ states stacked as rows of $A\in\{0,1\}^{N\times m}$, all codimensions are obtained with one matrix--vector product $A\boldsymbol\rho$. This is the implementation used in the function \texttt{descriptores} of the pipeline; it was checked against a literal implementation of Algorithm~1 on 300 random subspaces ($m=552$, $1\le d<200$) with exact agreement.

\subsection{Null distribution}

\begin{proposition}[Moments under a random reference]
\label{prop:S-null}
If $\sigma$ is uniform on the symmetric group, then $\mathbb E|\lambda|=d(m-d)/2$ and $\mathrm{Var}|\lambda|=d(m-d)(m+1)/12$.
\end{proposition}
\begin{proof}
Under a uniform $\sigma$, $\{p_1,\dots,p_d\}$ is a uniformly random $d$-subset of $\{1,\dots,m\}$, so $W=\sum_lp_l$ is the Wilcoxon rank-sum statistic of a sample of size $d$ among $m$, with $\mathbb EW=d(m+1)/2$ and $\mathrm{Var}\,W=d(m-d)(m+1)/12$ \citep{Mann1947}. Since $|\lambda|=d(m-d)+d(d+1)/2-W$ is an affine function of $W$ with slope $-1$, $\mathbb E|\lambda|=d(m-d)+d(d+1)/2-d(m+1)/2=d(m-d)/2$ and the variance is unchanged.
\end{proof}

The moments were verified by Monte Carlo ($2\times10^4$ random references for $d\in\{7,20,100,165\}$, $m=552$): relative errors below 5\% for the variance and below $0.05$ standard deviations for the mean. As $d(m-d)\to\infty$, $z$ is asymptotically standard normal; for the exact distribution at small $d$, the Mann--Whitney recursion applies.

\subsection{Properties of the normalized codimension}

\begin{proposition}
\label{prop:S-nu}
Let $\nu=|\lambda|/[d(m-d)]$.
\begin{enumerate}
\item $\nu\in[0,1]$; $\nu=1$ iff the active coordinates occupy the first $d$ positions of $\sigma$, and $\nu=0$ iff they occupy the last $d$.
\item $\nu$ equals the probability that a uniformly chosen active coordinate precedes a uniformly chosen inactive coordinate (the AUC of the ``active vs inactive'' classifier that scores coordinates by earliness along $\sigma$; \citealp{Hanley1982}).
\item Under a uniformly random $\sigma$, $\mathbb E\nu=1/2$ and $\mathrm{Var}\,\nu=(m+1)/[12\,d(m-d)]$.
\item (Reversal) If $\bar\sigma(j)=\sigma(m+1-j)$ is the reversed ordering, then $\nu_{\bar\sigma}=1-\nu_\sigma$.
\item (Complement) If $P^{\perp}=\mathrm{span}\{e_r:r\notin R\}$, then $\nu(P^{\perp})=1-\nu(P)$.
\end{enumerate}
\end{proposition}
\begin{proof}
(1)--(2) follow from $|\lambda|=U$ and the definition of $U$ as a count over the $d(m-d)$ (active, inactive) pairs. (3) follows from Proposition~\ref{prop:S-null}. (4) Reversal turns every ordered pair (active before inactive) into (inactive before active), so $U_{\bar\sigma}=d(m-d)-U_\sigma$. (5) Exchanging the roles of active and inactive coordinates has the same effect.
\end{proof}

\begin{remark}
Item (4) gives a reference point for the ``tiers reversed'' analysis of the main text. Reversing only the order of the tier blocks, while keeping the within-block order, is \emph{not} a full reversal; if the descriptor were driven by the tier order, $\nu$ would move towards $1-\nu$; if it were driven by the within-block (frequency) order, it would remain close to $\nu$. The observed behaviour (main Table~7) corresponds to the second case.
\end{remark}

\subsection{Schubert containment for coordinate subspaces}

\begin{proposition}
\label{prop:S-order}
Let $P,Q$ be coordinate subspaces of the same dimension $d$ with positions $p_1<\dots<p_d$ and $q_1<\dots<q_d$ along $\sigma$, and partitions $\lambda(P)$, $\lambda(Q)$. Then $\lambda(P)\supseteq\lambda(Q)$ (as Young diagrams) if and only if $p_l\le q_l$ for every $l$. In that case the Schubert variety of $P$ is contained in that of $Q$, $X_{\lambda(P)}\subseteq X_{\lambda(Q)}$.
\end{proposition}
\begin{proof}
$\lambda_l(P)\ge\lambda_l(Q)$ iff $(m-d)+l-p_l\ge(m-d)+l-q_l$ iff $p_l\le q_l$. The containment of Schubert varieties reverses the inclusion of partitions \citep{Fulton1997}.
\end{proof}

Thus the partial order induced by Schubert containment on configurations of equal dimension is the componentwise (Gale) order of their sorted positions: $P$ is ``below'' $Q$ when each of its active coordinates is at least as early in the reference as the corresponding active coordinate of $Q$. The scalar $|\lambda|$ is a linear extension of this order; two configurations with equal codimension may be incomparable.

\subsection{Resolution of exact permutation tests}

\begin{lemma}
\label{lem:resol}
For $n$ species with distinct values, the exact two-sided permutation $p$-value of the Spearman correlation is at least $2/n!$.
\end{lemma}
\begin{proof}
The identity permutation and its reversal attain $|\rho|=1$; at least these two among the $n!$ permutations satisfy $|\rho_\pi|\ge|\rho_{\mathrm{obs}}|$.
\end{proof}
For $n=6$ the floor is $0.0028$; for $n=9$, $5.5\times10^{-6}$. Non-significant results in the main text are therefore not artefacts of test resolution.

\subsection{Why partial correlations of raw codimension are not used}

Across states, raw codimension and $d$ are almost collinear (main Section~6.3; $\rho_S\approx0.97$). Residualizing the ranks of $|\lambda|$ on the ranks of $d$ leaves residuals dominated by ties and by the curvature of $d(m-d)/2$, not by positional information. In a synthetic community in which tolerance depends only on $d$ and positions are uniformly random ($n=11$), the partial Spearman correlation of raw codimension with tolerance given $d$ was $0.71$ (permutation $p=0.029$), a false positive, whereas the partial correlation of $\nu$ given $d$ was $-0.43$ ($p=0.19$). The main text therefore reports the partial correlation of $\nu$, which is already free of the mechanical dependence on $d$.

%% ======================================================================
\section{Synthetic validation}
\label{sec:S-synth}

\paragraph{Design.} For each replicate, a latent trait $z_i\sim U(0,1)$ was drawn for $n_s$ species, with $m=553$. Each species contributed 20 states. In the \emph{positional} regime, $d_i\sim U\{8,\dots,149\}$ independently of $z_i$, and the active positions of each state were sampled without replacement with weights $w_j\propto\exp\{0.35\,z_i(1-j/m)\}$, so that larger $z_i$ places active coordinates earlier. In the \emph{confounded} regime, $d_i=\mathrm{round}(8+140z_i+\epsilon_i)$, $\epsilon_i\sim N(0,10^2)$, truncated to $[3,m-3]$, and positions were uniform. In the \emph{null} regime, $d_i$ was independent of $z_i$ and positions were uniform. Species means of raw $|\lambda|$ and of $\nu$ were correlated with $z_i$ (Spearman, $\alpha=0.05$); 1000 replicates per cell.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figS2_synthetic.pdf}
\caption{Rejection rate of the Spearman test ($\alpha=0.05$, 1000 replicates) for raw codimension and normalized codimension $\nu$ under (A) positional signal with $d$ independent of the trait, (B) $d$ increasing with the trait without positional signal, and (C) the null regime. Dashed line: nominal level.}
\label{fig:S-synth}
\end{figure}

\begin{table}[H]
\centering
\caption{Synthetic validation: rejection rate ($r$) and median Spearman correlation ($\tilde\rho$) for raw codimension $|\lambda|$ and normalized codimension $\nu$.}
\label{tab:S-synth}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}c cccc cccc cccc@{}}
\toprule
 & \multicolumn{4}{c}{Positional} & \multicolumn{4}{c}{Confounded} & \multicolumn{4}{c}{Null} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
$n_s$ & $r_{|\lambda|}$ & $r_\nu$ & $\tilde\rho_{|\lambda|}$ & $\tilde\rho_\nu$ & $r_{|\lambda|}$ & $r_\nu$ & $\tilde\rho_{|\lambda|}$ & $\tilde\rho_\nu$ & $r_{|\lambda|}$ & $r_\nu$ & $\tilde\rho_{|\lambda|}$ & $\tilde\rho_\nu$ \\
\midrule
@@SINT@@
\bottomrule
\end{tabular}
\end{table}

%% ======================================================================
\section{Diagnostic of the submitted analysis}
\label{sec:S-submitted}

\begin{figure}[H]
\centering
\includegraphics[width=0.55\linewidth]{figS1_codim_vs_interface_submitted.pdf}
\caption{Mean raw Schubert codimension against the number of periplasm--extracellular transport reactions for the nine $T_1$/$T_2$ taxa, as reported in the submitted version (values transcribed from its Tables~5 and~7). Spearman correlation with exact permutation $p$-value. The near-perfect association is the mechanical dependence of Remark~1 (main text).}
\label{fig:S-submitted}
\end{figure}

%% ======================================================================
\section{Community model: species, interfaces and medium}
\label{sec:S-model}

\begin{table}[H]
\centering
\caption{Medium used in all simulations (lower bounds of exchange reactions, mmol gDW$^{-1}$ h$^{-1}$). All other exchanges were closed for uptake; secretion was unrestricted.}
\label{tab:S-medium}
\small
\begin{tabular}{@{}p{11cm}c@{}}
\toprule
Compound(s) & Lower bound \\
\midrule
Glycerol (sole carbon source) & $-10$ \\
Fe$^{2+}$, Fe$^{3+}$ & $-1$ \\
L-asparagine, siroheme (traces required by iSRU780) & $-0.1$ \\
H$_2$O, H$^+$, O$_2$, CO$_2$, NH$_4^+$, P$_i$, SO$_4^{2-}$, K$^+$, Na$^+$, Cl$^-$, Ca$^{2+}$, Mg$^{2+}$, Mn$^{2+}$, Zn$^{2+}$, Cu$^{2+}$, Co$^{2+}$, Ni$^{2+}$, MoO$_4^{2-}$ & $-1000$ \\
\bottomrule
\end{tabular}
\end{table}

The ambient space has $m=@@M@@$ coordinates, one per community exchange reaction; the species interfaces are listed in main Table~5. With complete reconstructions, 584 shared extracellular metabolites are exchanged, against 553 in the earlier version of the model. Net exchange of species $i$ with metabolite $j$ is $f_{ij}=\sum_r S_{jr}v_r$ over the reactions of species $i$ that consume or produce the shared extracellular metabolite $j$. The definition used in the submitted version (\texttt{tex} reactions only) is retained as a sensitivity analysis (Table~\ref{tab:S-corr}).

%% ======================================================================
\section{Curation of the community model}
\label{sec:S-curation}

\subsection{Diagnosis of the earlier community}

The earlier version of the community model predicted a growth rate of 24.1~h$^{-1}$ on 10~mmol~gDW$^{-1}$~h$^{-1}$ glycerol and 23.4~h$^{-1}$ without glycerol, with only three taxa growing. Closing all exchanges abolished growth, so the model did not create mass; it created \emph{energy and reducing power}. Testing each species in isolation identified five causes: (1) 4,078 of 15,079 metabolites lacked \texttt{fbc:charge}, which COBRApy reads as zero, leaving 18.8\% of reactions charge-imbalanced; (2) reversible oxidases, oxygenases and peroxidases, which run backwards produce O$_2$ or oxidize water; (3) seven taxa represented by small draft models (317--430 metabolic reactions) with 53--75\% reversible reactions, against 28--31\% in CarveMe reconstructions (Fig.~\ref{fig:S-curation}A); (4) energy-generating cycles \citep{Fritzemeier2017} in four draft models and in \emph{Hal. elongata}; and (5) no coupling between species, so that any single taxon could reach the community optimum on precursors supplied by others that were not required to grow.

\subsection{Retained CarveMe reconstructions}

Eight taxa were kept from the earlier community and curated by \texttt{curar\_modelo.curar\_completo}: (i) the SBML was repaired (an undeclared biomass intermediate was declared and \texttt{boundaryCondition} was cleared on 554 species); (ii) unknown charges, assumed equal across compartments and species for the same metabolite, were inferred iteratively from reactions in which a single charge was unknown (accepted when at least 60\% of such reactions agreed) and the remainder by integer least squares, which recovered 609 of 629 metabolites and reduced charge-imbalanced reactions to 1.6\%; (iii) reversible non-transport reactions with O$_2$ or H$_2$O$_2$ on only one side were restricted to the oxidant-consuming direction (@@NOX@@ reactions in these taxa: @@OXLIST@@); (iv) reversible reactions that remained charge-imbalanced were restricted to their written direction (@@NIMB@@: @@IMBLIST@@); and (v) \texttt{GCDH}, which formed a cycle with \texttt{GLUTCOADHc} in \emph{Hal. elongata}, was set to its physiological direction. Every change is recorded in \texttt{cambios.csv}.

\subsection{Inserted reconstructions}

The seven draft models were replaced by new CarveMe reconstructions (Gram-negative universe for the six bacteria, archaeal universe for \emph{Hmc. mukohataei}), and the CarveMe models of \emph{Sal. ruber} and \emph{Hqr. walsbyi} by the manually curated iSRU780 and iHQW-TK. Each model was first inspected in isolation (\texttt{inspeccion.py}): reactions, reversibility, missing and imbalanced charges, glycerol pathway, growth on M9 with glycerol and without carbon, and energy-generating cycles for twelve energy metabolites. The rule restricting reversible imbalanced reactions to their written direction, appropriate for the earlier community after charge repair, was not applied: in complete CarveMe reconstructions about 20\% of reactions are charge-imbalanced because metabolites are annotated with an explicit charge of zero, and applying the rule abolished growth on minimal medium. Curation was therefore limited to the oxidase rule and to cycles detected by the energy-generating-cycle test (\texttt{curar\_indiv2.py}); all changes are listed in Table~\ref{tab:S-changes}. Before curation, \emph{Arh. aquaeolei}, \emph{Gra. halophila}, \emph{Ocn. granulosus} and \emph{Ros. halotolerans} regenerated NADH, NADPH, FADH$_2$, ubiquinol-8, menaquinol-8 and glutamate in a closed system through reversible \texttt{GCDH}, and \emph{Arh. aquaeolei} and \emph{Ros. halotolerans} grew at 6.9 and 3.2~h$^{-1}$ without carbon (Fig.~\ref{fig:S-curation}B); iHQW-TK regenerated the periplasmic proton gradient through reversible \texttt{F4H2O}.

\begin{table}[H]
\centering
\caption{Bounds changed in the nine inserted models.}
\label{tab:S-changes}
\small
\begin{tabular}{@{}llll@{}}
\toprule
Species & Reaction & Rule & Bounds \\
\midrule
@@CAMNUEVOS@@
\bottomrule
\end{tabular}
\end{table}

Each curated model was inserted into the community (\texttt{ensamblar.py}) after removing the taxon it replaced: its extracellular metabolites were mapped onto the shared pool (with a new community exchange reaction, closed for uptake, for metabolites not yet in the pool), its other metabolites and compartments and its reactions and genes were suffixed with the taxon name, its own exchange reactions were removed, and its biomass reaction was linked to \texttt{community\_biomass}. iSRU780 does not grow alone on M9 with glycerol: it lacks asparagine synthetase and the terminal steps of siroheme synthesis, which no other taxon can export, and it cannot synthesize lysine on this medium. Asparagine and siroheme were added to the medium as traces (Table~\ref{tab:S-medium}); lysine was left to cross-feeding (Appendix~\ref{sec:S-eco}).

\subsection{Coupling and validation}

For every reaction $r$ of species $i$ exchanging a shared extracellular metabolite,
\begin{equation}
-(c\,\mu_i+u)\le v_r\le c\,\mu_i+u,\qquad c=400,\;u=0.01,
\end{equation}
where $\mu_i$ is the flux of the biomass reaction of species $i$ \citep{Heinken2013}. This adds 13,742 constraints on 6,871 reactions; they are not stored in the SBML and are applied on loading (\texttt{curar\_modelo.acoplar}). After assembly, the energy-generating-cycle test found no cycle in any of the $17\times12$ species--metabolite combinations; no taxon grew without an organic carbon source; and the coupled community grew at 0.780~h$^{-1}$ with 10~mmol~gDW$^{-1}$~h$^{-1}$ glycerol and 44.6~mmol~gDW$^{-1}$~h$^{-1}$ O$_2$ (Table~\ref{tab:S-species}).

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{figS3_curation.pdf}
\caption{Reconstruction and curation of the inserted models. (A) Percentage of reversible reactions in the model representing each taxon in the earlier community (orange) and in the inserted model (blue); the shaded band is the range of the CarveMe reconstructions of the earlier community. (B) Growth in isolation on M9 with 10~mmol~gDW$^{-1}$~h$^{-1}$ glycerol before and after curation (log scale; zero shown at the axis floor); labels give growth without any carbon source before curation.}
\label{fig:S-curation}
\end{figure}

\begin{table}[H]
\centering
\caption{The seventeen models of the curated community. $n$: reactions; Rev.: reversible reactions (\%); Chg.: bounds changed by curation; EGC: energy metabolites regenerated in a closed system before curation (six metabolites: NADH, NADPH, FADH$_2$, ubiquinol-8, menaquinol-8 and glutamate; PMF: periplasmic proton gradient); growth in isolation (h$^{-1}$) after curation on M9 with glycerol (Gly) and without organic carbon (no C).}
\label{tab:S-species}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}llccccc c@{}}
\toprule
Species & Model & $n$ & Rev. & Chg. & EGC & Gly & no C \\
\midrule
@@ESPMOD@@
\bottomrule
\end{tabular}
\end{table}

%% ======================================================================
\section{Schubert descriptors per species}
\label{sec:S-desc}

\begin{table}[H]
\centering
\caption{Mean (SD) dimension, raw codimension, normalized codimension and $z$ per species over the sampled states in which each taxon was active ($\varepsilon=0.02$; leave-one-out Santa Pola reference).}
\label{tab:S-desc}
\small
\begin{tabular}{@{}llcccc@{}}
\toprule
Species & Tier & $d$ & $|\lambda|$ & $\nu$ & $z$ \\
\midrule
@@DESC@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Spearman correlations with NaCl tolerance breadth (exact permutation $p$-values) under the two interface definitions.}
\label{tab:S-corr}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
 & \multicolumn{2}{c}{All interface reactions} & \multicolumn{2}{c}{\texttt{tex} reactions only} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Descriptor & $n_s=9$ & high conf.\ $n_s=6$ & $n_s=9$ & high conf.\ $n_s=6$ \\
\midrule
@@CORR@@
\bottomrule
\end{tabular}
\end{table}

%% ======================================================================
\section{Robustness}
\label{sec:S-robust}

\subsection{Activity threshold}

\begin{table}[H]
\centering
\caption{Mean $\nu$ of the validated taxa (both scenarios) as a function of the activity threshold $\varepsilon$.}
\label{tab:S-eps}
\small
\begin{tabular}{@{}l@@EPSCOLS@@@{}}
\toprule
Species & @@EPSCAB@@ \\
\midrule
@@EPS@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Reference ordering}

\begin{table}[H]
\centering
\caption{Mean $\nu$ per validated taxon under alternative references: Santa Pola tiers (original), usage frequency without tiers, tiers reversed, within-block random permutations (mean of 200) and uniformly random references (mean of 200). Ordering by usage frequency alone reproduces @@FRAC@@\% of the departure of the original reference from $1/2$.}
\label{tab:S-variants}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Species & Original & Frequency only & Reversed & Within-block & Random \\
\midrule
@@VAR@@
\bottomrule
\end{tabular}
\end{table}

\subsection{Scenario effect and random-perturbation control}

\begin{table}[H]
\centering
\caption{Proportional-odds coefficient of $\nu$-tertile on scenario (SS37 vs SS19; $N=150$ states per scenario) and its distribution under @@NCTRL@@ control replicates restricting the same numbers of randomly chosen exchange coordinates ($N=30$ per scenario). $p_{\mathrm{ctrl}}$: fraction of replicates with $|\hat\beta_{\mathrm{ctrl}}|\ge|\hat\beta|$. Replicate coefficients with a 95\% interval wider than 50 (quasi-complete separation with $N=30$) were discarded.}
\label{tab:S-control}
\small
\begin{tabular}{@{}lcccccc@{}}
\toprule
Species & $\hat\beta$ & 95\% CI & $p$ & mean $\hat\beta_{\mathrm{ctrl}}$ & range $\hat\beta_{\mathrm{ctrl}}$ & $p_{\mathrm{ctrl}}$ \\
\midrule
@@CTRL@@
\bottomrule
\end{tabular}
\end{table}

%% ======================================================================
\section{Persistent homology}
\label{sec:S-ph}

Vietoris--Rips filtrations were computed with \texttt{ripser} on the 300 net-exchange vectors ($\varepsilon=0.02$) of each $T_1$/$T_2$ species, after removing coordinates with zero variance; bottleneck distances between $\mathcal H_1$ diagrams were computed with \texttt{persim}. Infinite $\mathcal H_0$ bars were excluded from total persistence.

\begin{table}[H]
\centering
\caption{Total persistence and mean Schubert descriptors per species.}
\label{tab:S-ph}
\small
\begin{tabular}{@{}lccccc@{}}
\toprule
Species & $\mathcal H_0$ & $\mathcal H_1$ & loops & $d$ & $\nu$ \\
\midrule
@@PH@@
\bottomrule
\end{tabular}
\end{table}

%% ======================================================================
\section{Cross-feeding in the community model}
\label{sec:S-eco}

All results in this section were computed on the coupled community at the medium of Table~\ref{tab:S-medium} (\texttt{ecologia.py}, \texttt{ecologia\_dep.py}, \texttt{intercambio.py}). Growth inside the community was maximized for each taxon with the other taxa unconstrained and with community growth fixed at $\ge90\%$ of its optimum (0.780~h$^{-1}$). The parsimonious solution was computed by pFBA at 90\% of the community optimum; dependence on a metabolite was tested by forbidding the taxon's net uptake of it.

\begin{table}[H]
\centering
\caption{Growth of each taxon (h$^{-1}$): alone on M9 with glycerol, maximal inside the community (unconstrained and at $\ge90\%$ of the community optimum), and in the parsimonious solution at 90\% of the optimum.}
\label{tab:S-eco-growth}
\small
\begin{tabular}{@{}lcccc@{}}
\toprule
Species & Alone & Community max & At 90\% & pFBA \\
\midrule
@@ECOCRE@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Maximal biomass flux (h$^{-1}$) of \emph{Sal. ruber} and \emph{Hqr. walsbyi} when their net uptake of one metabolite received from other taxa is forbidden. Unrestricted maxima: \emph{Sal. ruber} 0.415 (0.415 at 90\%); \emph{Hqr. walsbyi} 0.364 (0.131 at 90\%). ``Infeasible'': the community cannot reach 90\% of its optimum without growth of \emph{Sal. ruber} (optimum without it: 0.607~h$^{-1}$).}
\label{tab:S-eco-dep}
\small
\begin{tabular}{@{}llcc@{}}
\toprule
Species & Metabolite blocked & Max (free) & Max (90\%) \\
\midrule
@@ECODEP@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Metabolites taken up by \emph{Hqr. walsbyi} from other taxa when it grows at its community maximum (pFBA with its biomass flux fixed), and secretion by each donor (mmol gDW$^{-1}$ h$^{-1}$).}
\label{tab:S-eco-hqr}
\small
\begin{tabular}{@{}lcp{8cm}@{}}
\toprule
Metabolite & Uptake & Donors \\
\midrule
@@ECOHQR@@
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Most frequent donor--receiver links across the 300 sampled states (metabolites not supplied by the medium): fraction of states in which the link carried more than $\varepsilon=0.02$ and mean allocated flux (mmol gDW$^{-1}$ h$^{-1}$), averaged over the two scenarios.}
\label{tab:S-eco-links}
\small
\begin{tabular}{@{}lllcc@{}}
\toprule
Donor & Receiver & Metabolite & Fraction & Flux \\
\midrule
@@ECOARI@@
\bottomrule
\end{tabular}
\end{table}

The asparagine trace set the maximal growth of \emph{Sal. ruber}: with 0.2~mmol~gDW$^{-1}$~h$^{-1}$ it reached 0.604~h$^{-1}$, and with 1~mmol~gDW$^{-1}$~h$^{-1}$ of both traces 0.647~h$^{-1}$, while the community optimum changed from 0.780 to 0.787 and 0.842~h$^{-1}$. Siroheme was not limiting at 0.1~mmol~gDW$^{-1}$~h$^{-1}$.

%% ======================================================================
\section{Software and reproducibility}
\label{sec:S-software}

All analyses were run in Python with COBRApy \citep{Ebrahim2013}, optlang and the HiGHS solver, NumPy, SciPy, pandas, statsmodels, ripser and persim. The pipeline consists of:
\begin{itemize}
\item \texttt{curar\_modelo.py}, \texttt{cargas.py}, \texttt{reglas.py}: curation of the earlier community (Section~\ref{sec:S-curation});
\item \texttt{inspeccion.py}, \texttt{curar\_indiv2.py}, \texttt{egc.py}, \texttt{ensamblar.py}: inspection, curation and insertion of the new models;
\item \texttt{ecologia.py}, \texttt{ecologia\_dep.py}, \texttt{intercambio.py}, \texttt{fig\_ecologia.py}: cross-feeding analyses (Section~\ref{sec:S-eco});
\item \texttt{pipeline\_schubert.py}: exchange space, reference flags, random-objective sampling with on-disk caching, Schubert descriptors, sensitivity analyses, ordinal models, control, FVA envelopes and persistent homology;
\item \texttt{paso1\_codim\_normalizada.py}: H1a tests with exact permutation $p$-values;
\item \texttt{sintetico.py}: synthetic study;
\item \texttt{figuras\_revision.py}, \texttt{figuras\_run.py}, \texttt{suplementario.py}: figures, tables and this document;
\item \texttt{ejecutar\_pipeline.ipynb}: notebook running every step in order.
\end{itemize}
Random-objective FBA required about 20~s per state with HiGHS on the coupled model (35,853 reactions); states were generated in batches of 40 in fresh processes to bound memory use. Linear programs with an added uptake constraint were solved with HiGHS's interior-point method, because the simplex method stalled on them. Random seeds are fixed in every script. Flux variability envelopes are implemented (\texttt{envolvente\_fva}) but were not computed.

\begin{thebibliography}{9}
\bibitem[{Ebrahim et~al.(2013)}]{Ebrahim2013} Ebrahim, A., Lerman, J.A., Palsson, B.O., Hyduke, D.R., 2013. COBRApy: COnstraints-Based Reconstruction and Analysis for Python. BMC Systems Biology 7, 74.
\bibitem[{Fritzemeier et~al.(2017)}]{Fritzemeier2017} Fritzemeier, C.J., Hartleb, D., Szappanos, B., Papp, B., Lercher, M.J., 2017. Erroneous energy-generating cycles in published genome scale metabolic networks: identification and removal. PLoS Computational Biology 13, e1005494.
\bibitem[{Fulton(1997)}]{Fulton1997} Fulton, W., 1997. Young Tableaux, with Applications to Representation Theory and Geometry. Cambridge University Press.
\bibitem[{Hanley and McNeil(1982)}]{Hanley1982} Hanley, J.A., McNeil, B.J., 1982. The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology 143, 29--36.
\bibitem[{Heinken et~al.(2013)}]{Heinken2013} Heinken, A., Sahoo, S., Fleming, R.M.T., Thiele, I., 2013. Systems-level characterization of a host-microbe metabolic symbiosis in the mammalian gut. Gut Microbes 4, 28--40.
\bibitem[{Lewis et~al.(2010)}]{Lewis2010} Lewis, N.E., Hixson, K.K., Conrad, T.M., et al., 2010. Omic data from evolved E. coli are consistent with computed optimal growth from genome-scale models. Molecular Systems Biology 6, 390.
\bibitem[{Machado et~al.(2018)}]{Machado2018} Machado, D., Andrejev, S., Tramontano, M., Patil, K.R., 2018. Fast automated reconstruction of genome-scale metabolic models for microbial species and communities. Nucleic Acids Research 46, 7542--7553.
\bibitem[{Mann and Whitney(1947)}]{Mann1947} Mann, H.B., Whitney, D.R., 1947. On a test of whether one of two random variables is stochastically larger than the other. Annals of Mathematical Statistics 18, 50--60.
\end{thebibliography}

\end{document}
"""

if __name__ == "__main__":
    construir()
