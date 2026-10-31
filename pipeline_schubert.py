# -*- coding: utf-8 -*-
"""
pipeline_schubert.py  --  complete pipeline for the revised Schubert manuscript.

From the community GSMM (SBML) to every table and CSV used in the paper:

  1. cargar_modelo / construir_mapa     species -> exchange-space interface
  2. orden_bandera                      reference flag (leave-one-out, variants)
  3. muestrear                          random-objective FBA, cached per scenario
  4. tabla_muestras                     one row per (state, species): d, |lambda|, nu, z
  5. sensibilidad_bandera               within-tier / reversed / random flags   (Reviewer 1)
  6. barrido_epsilon                    activity-threshold sweep                (H1b-ii)
  7. regresion_ordinal, control_escenarios   scenario effect + random control
  8. envolvente_fva                     FVA envelope per species                (Reviewer 3)
  9. homologia_persistente              H0/H1 persistence + bottleneck matrix
 10. plantilla_tolerancia               template for the NaCl tolerance table

Every expensive step writes to disk (cache/) and is reused on re-runs.
Requires: cobra, numpy, pandas, scipy, statsmodels, ripser, persim.
Python >= 3.9.
"""
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats

# ----------------------------------------------------------------------
# Configuration defaults (override from the notebook)
# ----------------------------------------------------------------------
# Token used to recognise each species inside reaction / compartment ids,
# mapped to the abbreviation used in the manuscript. Matching is
# case-insensitive and requires the token to be delimited by _ or
# non-alphanumeric characters.
NOMBRES = {
    "Haloquadratum": "Hqr. walsbyi",
    "Salinibacter": "Sal. ruber",
    "Halomicrobium": "Hmc. mukohataei",
    "Halorubrum": "Hrr. salinarum",
    "Natronomonas": "Nmn. salina",
    "Spiribacter": "Spb. salinus",
    "Oceanicola": "Ocn. granulosus",
    "Roseovarius": "Ros. halotolerans",
    "Roseavarius": "Ros. halotolerans",   # spelling used in community12.xml
    "Gracilimonas": "Gra. halophila",
    "Arhodomonas": "Arh. aquaeolei",
    "Puniceicoccus": "Pcc. vermicola",
    "Halomonas": "Hal. elongata",
    "Chromohalobacter": "Chb. salexigens",
    "Haloferax": "Hfx. volcanii",
    "Haloarcula": "Har. halophila",
    "Halobacterium": "Hbt. salinarum",
    "Tetragenococcus": "Tet. halophilus",
}

TIERS = {
    "T0": ["Ros. halotolerans", "Gra. halophila"],
    "T1": ["Hrr. salinarum", "Nmn. salina", "Spb. salinus",
           "Ocn. granulosus", "Arh. aquaeolei", "Pcc. vermicola"],
    "T2": ["Hqr. walsbyi", "Sal. ruber", "Hmc. mukohataei"],
}

VALIDADAS = TIERS["T0"] + TIERS["T1"] + TIERS["T2"]

# Scenario definition (Section 4.5 of the manuscript):
#   SS19 -> restrict exchange reactions associated with T2
#   SS37 -> restrict exchange reactions associated with T1
ESCENARIOS = {"SS19": "T2", "SS37": "T1"}


def _log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ----------------------------------------------------------------------
# 1. Model and species -> exchange map
# ----------------------------------------------------------------------
def cargar_modelo(ruta_sbml, solver=None):
    """Read the SBML model. solver: 'gurobi', 'cplex', 'glpk' or None (best available)."""
    import cobra
    _log("reading %s ..." % ruta_sbml)
    model = cobra.io.read_sbml_model(ruta_sbml)
    candidatos = [solver] if solver else ["gurobi", "cplex", "glpk"]
    for s in candidatos:
        try:
            model.solver = s
            break
        except Exception:
            continue
    _log("model: %d reactions, %d metabolites, solver = %s"
         % (len(model.reactions), len(model.metabolites),
            model.solver.interface.__name__.split(".")[-1]))
    return model


def _patron(token):
    return re.compile(r"(?i)(?:^|[^a-z0-9])%s(?:$|[^a-z0-9])" % re.escape(token))


class Mapa(object):
    """Exchange space E and species interfaces.

    exch_ids[j]   community exchange reaction of coordinate j
    met_e[j]      shared extracellular metabolite of coordinate j
    especies      species tokens found in the model (order = rows of F)
    interfaz[i]   set of coordinates j touched by species i
    W             sparse (n_sp*m x n_rxn): net flux of species i on metabolite j
    cols          reaction ids matching the columns of W
    """

    def __init__(self):
        self.exch_ids, self.met_e, self.especies = [], [], []
        self.interfaz, self.rxns_especie = {}, {}
        self.W, self.cols, self.m = None, [], 0

    def abrev(self, token):
        return NOMBRES.get(token, token)

    def token(self, abrev):
        cands = [k for k, v in NOMBRES.items() if v == abrev] or [abrev]
        for t in cands:
            for e in self.especies:
                if e.lower() == t.lower():
                    return e
        return cands[0]

    def flujos_netos(self, v):
        """v: flux vector ordered as self.cols -> array (n_sp, m)."""
        return (self.W @ v).reshape(len(self.especies), self.m)


def construir_mapa(model, tokens=None, objetivo="community_growth", interfaz="todas",
                   verbose=True):
    """Build E (one coordinate per community exchange reaction) and the net
    exchange of each species with every shared extracellular metabolite.

    interfaz = "todas": every reaction of the species that consumes or produces
                        a shared extracellular metabolite (net exchange with the
                        medium; recommended).
               "tex"  : only periplasm-extracellular transport reactions
                        (ids containing 'tex'/'texi'); reproduces the interface
                        sizes of the submitted manuscript (Table 5)."""
    filtro = re.compile(r"(?i)texi?_") if interfaz == "tex" else None
    tokens = list(tokens or NOMBRES.keys())
    pats = {t: _patron(t) for t in tokens}

    def especie_de(texto):
        hits = [t for t, p in pats.items() if p.search(texto)]
        return hits[0] if len(hits) == 1 else None

    mapa = Mapa()
    # --- community exchange reactions: boundary, one metabolite, untagged
    for r in model.reactions:
        if r.id == objetivo or not r.boundary or len(r.metabolites) != 1:
            continue
        if especie_de(r.id):
            continue
        met = list(r.metabolites)[0]
        mapa.exch_ids.append(r.id)
        mapa.met_e.append(met.id)
    orden = np.argsort(mapa.exch_ids)
    mapa.exch_ids = [mapa.exch_ids[k] for k in orden]
    mapa.met_e = [mapa.met_e[k] for k in orden]
    mapa.m = len(mapa.exch_ids)
    idx = {mid: j for j, mid in enumerate(mapa.met_e)}

    # --- species reactions touching a shared extracellular metabolite
    entradas = []   # (token, rxn_id, j, coef)
    for r in model.reactions:
        if r.id == objetivo or r.boundary:
            continue
        if filtro is not None and not filtro.search(r.id):
            continue
        compartidos = [(met, c) for met, c in r.metabolites.items() if met.id in idx]
        if not compartidos:
            continue
        tok = especie_de(r.id)
        if tok is None:
            otros = " ".join("%s %s" % (met.id, met.compartment)
                             for met in r.metabolites if met.id not in idx)
            tok = especie_de(otros)
        if tok is None:
            continue
        for met, c in compartidos:
            entradas.append((tok, r.id, idx[met.id], float(c)))

    mapa.especies = sorted({e[0] for e in entradas})
    sp_idx = {t: i for i, t in enumerate(mapa.especies)}
    mapa.cols = sorted({e[1] for e in entradas})
    col_idx = {rid: k for k, rid in enumerate(mapa.cols)}
    filas = [sp_idx[t] * mapa.m + j for t, _, j, _ in entradas]
    columnas = [col_idx[rid] for _, rid, _, _ in entradas]
    datos = [c for _, _, _, c in entradas]
    mapa.W = sparse.csr_matrix((datos, (filas, columnas)),
                               shape=(len(mapa.especies) * mapa.m, len(mapa.cols)))
    for t in mapa.especies:
        mapa.interfaz[t] = sorted({j for tt, _, j, _ in entradas if tt == t})
        mapa.rxns_especie[t] = sorted({rid for tt, rid, _, _ in entradas if tt == t})

    if verbose:
        _log("E: m = %d community exchange reactions; interface = %s" % (mapa.m, interfaz))
        _log("species found: %d / %d" % (len(mapa.especies),
                                           len({NOMBRES.get(t, t) for t in tokens})))
        presentes = {mapa.abrev(t) for t in mapa.especies}
        faltan = [t for t in tokens if NOMBRES.get(t, t) not in presentes]
        if faltan:
            warnings.warn("tokens not found in the model: %s -- check spelling "
                          "against reaction / compartment ids" % faltan)
    return mapa


def resumen_mapa(mapa):
    filas = []
    for t in mapa.especies:
        a = mapa.abrev(t)
        tier = next((k for k, v in TIERS.items() if a in v), "-")
        filas.append(dict(token=t, especie=a, tier=tier,
                          reacciones_interfaz=len(mapa.rxns_especie[t]),
                          coordenadas_E=len(mapa.interfaz[t]),
                          validada=a in VALIDADAS))
    return (pd.DataFrame(filas)
            .sort_values("reacciones_interfaz", ascending=False)
            .reset_index(drop=True))


# ----------------------------------------------------------------------
# 2. Reference flag
# ----------------------------------------------------------------------
def _tier_sets(mapa, excluir=None):
    sets = {}
    for k, abrevs in TIERS.items():
        s = set()
        for a in abrevs:
            t = mapa.token(a)
            if t in mapa.interfaz and t != excluir:
                s |= set(mapa.interfaz[t])
        sets[k] = s
    return sets


def orden_bandera(mapa, excluir=None, variante="original", rng=None):
    """Return rank[j] in 1..m (position of coordinate j along the flag).

    variante: 'original'  tiers T0<T1<T2, within block by decreasing frequency
              'permutada' same blocks, random order inside each block
              'invertida' tiers T2<T1<T0, within block by frequency
              'aleatoria' uniformly random order
    excluir: species token left out (leave-one-out reference).
    """
    m = mapa.m
    rng = rng if rng is not None else np.random.default_rng(0)
    if variante == "aleatoria":
        orden = rng.permutation(m)
    else:
        freq = np.zeros(m, int)
        for t in mapa.especies:
            if t != excluir:
                freq[mapa.interfaz[t]] += 1
        J = _tier_sets(mapa, excluir)
        seq = ["T2", "T1", "T0"] if variante == "invertida" else ["T0", "T1", "T2"]
        usados, bloques = set(), []
        for k in seq:
            b = sorted(J[k] - usados)
            usados |= set(b)
            bloques.append(b)
        bloques.append(sorted(set(range(m)) - usados))
        orden = []
        for b in bloques:
            if variante == "permutada":
                b = list(rng.permutation(b))
            else:
                b = sorted(b, key=lambda j: (-freq[j], mapa.exch_ids[j]))
            orden.extend(b)
        orden = np.asarray(orden, int)
    rank = np.empty(m, float)
    rank[orden] = np.arange(1, m + 1)
    return rank


def dimensiones_bandera(mapa, excluir=None):
    """Cumulative dimensions of the tier levels (manuscript: 17, 199, 254)."""
    J = _tier_sets(mapa, excluir)
    a = len(J["T0"])
    b = len(J["T0"] | J["T1"])
    c = len(J["T0"] | J["T1"] | J["T2"])
    return dict(T0=a, T0_T1=b, T0_T1_T2=c, m=mapa.m)


# ----------------------------------------------------------------------
# 3. Random-objective FBA sampling (cached)
# ----------------------------------------------------------------------
def recalcular_F(V, cols, mapa):
    """Net-exchange array for another map (e.g. interfaz='tex') from the raw
    interface fluxes V saved in the cache, without re-sampling."""
    pos = {c: k for k, c in enumerate(cols)}
    idx = [pos[c] for c in mapa.cols]
    return np.stack([mapa.flujos_netos(np.asarray(v)[idx]) for v in V])


def conjunto_tier(mapa, tier):
    return sorted(_tier_sets(mapa)[tier])


def _aplicar_restriccion(model, mapa, coords, factor=0.1, modo="ub"):
    for j in coords:
        r = model.reactions.get_by_id(mapa.exch_ids[j])
        if modo in ("ub", "ambos") and r.upper_bound > 0:
            r.upper_bound = factor * r.upper_bound
        if modo in ("lb", "ambos") and r.lower_bound < 0:
            r.lower_bound = factor * r.lower_bound


def muestrear(model, mapa, escenario, n=150, cache_dir="cache", restringir=None,
              factor=0.1, modo="ub", frac_opt=0.9, objetivo="community_growth",
              semilla=0, guardar_cada=5, etiqueta=None, max_nuevos=None):
    """Random-objective FBA: each state maximizes a random unit-norm linear
    max_nuevos : stop after this many new states (memory grows with each
                 objective change in some solver interfaces; run in batches)
    objective over the species interface reactions, with community growth
    >= frac_opt * optimum. Returns F with shape (n, n_species, m): net exchange
    of each species with each shared metabolite. Resumes from cache.

    escenario  : 'SS19' / 'SS37' (restricts the tier in ESCENARIOS) or 'base'
    restringir : explicit list of coordinates to restrict (overrides escenario)
    """
    os.makedirs(cache_dir, exist_ok=True)
    etiqueta = etiqueta or escenario
    ruta = os.path.join(cache_dir, "flujos_%s_n%d.npz" % (etiqueta, n))
    F, V = [], []
    if os.path.exists(ruta):
        _z = np.load(ruta)
        F = list(_z["F"])
        V = list(_z["V"]) if "V" in _z.files else []
        if len(F) >= n:
            _log("%s: %d states loaded from cache" % (etiqueta, len(F)))
            return np.asarray(F[:n])
        _log("%s: resuming from %d cached states" % (etiqueta, len(F)))

    if restringir is None and escenario in ESCENARIOS:
        restringir = conjunto_tier(mapa, ESCENARIOS[escenario])
    restringir = restringir or []
    rng = np.random.default_rng(semilla + 7919 * len(F))
    rxns = [model.reactions.get_by_id(rid) for rid in mapa.cols]

    with model:
        _aplicar_restriccion(model, mapa, restringir, factor, modo)
        model.objective = objetivo
        opt = model.slim_optimize()
        if not np.isfinite(opt) or opt <= 0:
            raise RuntimeError("%s: community growth infeasible or zero (%s)" % (etiqueta, opt))
        model.reactions.get_by_id(objetivo).lower_bound = frac_opt * opt
        _log("%s: optimum %.4g; %d restricted coordinates; sampling %d states"
             % (etiqueta, opt, len(restringir), n - len(F)))
        t0 = time.time()
        fallos = 0
        objetivo_n = n if max_nuevos is None else min(n, len(F) + max_nuevos)
        while len(F) < objetivo_n:
            c = rng.normal(size=len(rxns))
            c /= np.linalg.norm(c)
            model.objective = {r: float(ci) for r, ci in zip(rxns, c)}
            model.objective_direction = "max"
            sol = model.optimize()
            if sol.status != "optimal":
                fallos += 1
                if fallos > 5 * n:
                    raise RuntimeError("too many non-optimal solutions")
                continue
            v = sol.fluxes.reindex(mapa.cols).fillna(0.0).values
            F.append(mapa.flujos_netos(v))
            V.append(v.astype(np.float32))
            k = len(F)
            if k % guardar_cada == 0 or k == objetivo_n:
                np.savez_compressed(ruta, F=np.asarray(F), V=np.asarray(V),
                                    cols=np.asarray(mapa.cols))
                dt = (time.time() - t0) / max(1, k)
                _log("%s: %d/%d states (~%.1f s/state)" % (etiqueta, k, n, dt))
    return np.asarray(F)


# ----------------------------------------------------------------------
# 4. Schubert descriptors
# ----------------------------------------------------------------------
def descriptores(A, rank):
    """A: (N, m) boolean active matrix; rank: (m,) flag positions 1..m.
    Returns d, codim, nu, z (Proposition 1 / Corollary 1)."""
    m = A.shape[1]
    d = A.sum(axis=1).astype(float)
    s = A.astype(float) @ rank
    codim = d * (m - d) + d * (d + 1) / 2.0 - s
    rect = d * (m - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(rect > 0, codim / rect, np.nan)
        z = np.where(rect > 0, (codim - rect / 2) / np.sqrt(rect * (m + 1) / 12.0), np.nan)
    return d, codim, nu, z


def lambda_particion(activos_j, rank, m):
    """Full partition lambda for one state (list of active coordinates)."""
    p = np.sort(rank[np.asarray(activos_j, int)])
    d = p.size
    return [int((m - d) + l - pl) for l, pl in enumerate(p, start=1)]


def tabla_muestras(mapa, F_por_esc, eps=1e-6, variante="original", especies=None,
                   semilla=0, guardar=None):
    """One row per (scenario, state, species) with d, codim, nu, z.
    especies: list of abbreviations (default: all species in the model)."""
    rng = np.random.default_rng(semilla)
    especies = especies or [mapa.abrev(t) for t in mapa.especies]
    filas = []
    for a in especies:
        t = mapa.token(a)
        if t not in mapa.especies:
            continue
        i = mapa.especies.index(t)
        rank = orden_bandera(mapa, excluir=t, variante=variante, rng=rng)
        for esc, F in F_por_esc.items():
            A = np.abs(F[:, i, :]) > eps
            d, c, nu, z = descriptores(A, rank)
            for k in range(len(d)):
                filas.append(dict(especie=a, escenario=esc, muestra=k, d=int(d[k]),
                                  codim=float(c[k]), nu=nu[k], z=z[k]))
    df = pd.DataFrame(filas)
    df = df[(df["d"] > 0) & (df["d"] < mapa.m)].reset_index(drop=True)
    if guardar:
        df.to_csv(guardar, index=False)
        _log("written %s (%d rows)" % (guardar, len(df)))
    return df


# ----------------------------------------------------------------------
# 5. Flag sensitivity (Reviewer 1)
# ----------------------------------------------------------------------
def _medias_nu(mapa, A_por_sp, variante, rng):
    out = {}
    for a, (t, A) in A_por_sp.items():
        rank = orden_bandera(mapa, excluir=t, variante=variante, rng=rng)
        out[a] = np.nanmean(descriptores(A, rank)[2])
    return out


def sensibilidad_bandera(mapa, F_por_esc, tolerancia, B=1000, eps=1e-6,
                         especies=None, semilla=1, guardar=None):
    """Distribution of the H1a correlation (mean nu vs tolerance) and of the
    Kendall tau of the species ranking under permuted / reversed / random flags."""
    rng = np.random.default_rng(semilla)
    tol = tolerancia.dropna(subset=["tolerancia"]).set_index("especie")["tolerancia"]
    especies = [a for a in (especies or VALIDADAS) if a in tol.index
                and mapa.token(a) in mapa.especies]
    A_por_sp = {}
    for a in especies:
        t = mapa.token(a)
        i = mapa.especies.index(t)
        A = np.concatenate([np.abs(F[:, i, :]) > eps for F in F_por_esc.values()])
        A_por_sp[a] = (t, A)
    y = tol.loc[especies].values

    base = _medias_nu(mapa, A_por_sp, "original", rng)
    xb = np.array([base[a] for a in especies])
    rho0 = stats.spearmanr(xb, y)[0]
    filas = [dict(variante="original", rep=0, rho=rho0, tau=1.0)]
    for variante, reps in (("permutada", B), ("invertida", 1), ("aleatoria", B)):
        for r in range(reps):
            med = _medias_nu(mapa, A_por_sp, variante, rng)
            x = np.array([med[a] for a in especies])
            filas.append(dict(variante=variante, rep=r,
                              rho=stats.spearmanr(x, y)[0],
                              tau=stats.kendalltau(xb, x)[0]))
        _log("flag sensitivity: %s done" % variante)
    df = pd.DataFrame(filas)
    res = (df.groupby("variante")
             .agg(n=("rho", "size"), rho_mediana=("rho", "median"),
                  rho_p2_5=("rho", lambda s: s.quantile(0.025)),
                  rho_p97_5=("rho", lambda s: s.quantile(0.975)),
                  tau_mediana=("tau", "median"),
                  tau_p2_5=("tau", lambda s: s.quantile(0.025)))
             .reset_index())
    alea = df[df.variante == "aleatoria"]["rho"]
    res["p_vs_aleatoria"] = np.where(res.variante == "original",
                                     float((alea.abs() >= abs(rho0)).mean()), np.nan)
    if guardar:
        df.to_csv(guardar, index=False)
    return res, df


# ----------------------------------------------------------------------
# 6. Activity threshold sweep
# ----------------------------------------------------------------------
def barrido_epsilon(mapa, F_por_esc, eps_lista=(1e-6, 1e-4, 1e-2, 1e-1, 1, 10),
                    especies=None):
    especies = especies or VALIDADAS
    filas = []
    for a in especies:
        t = mapa.token(a)
        if t not in mapa.especies:
            continue
        i = mapa.especies.index(t)
        rank = orden_bandera(mapa, excluir=t)
        for esc, F in F_por_esc.items():
            for e in eps_lista:
                d, c, nu, z = descriptores(np.abs(F[:, i, :]) > e, rank)
                filas.append(dict(especie=a, escenario=esc, eps=e,
                                  d_media=np.mean(d), nu_media=np.nanmean(nu)))
    df = pd.DataFrame(filas)
    ref = df[df.eps == min(eps_lista)].set_index(["especie", "escenario"])["nu_media"]
    df["cambio_nu"] = df.apply(lambda r: r.nu_media - ref.loc[(r.especie, r.escenario)], axis=1)
    return df


# ----------------------------------------------------------------------
# 7. Scenario effect and random-perturbation control
# ----------------------------------------------------------------------
def _beta_ordinal(nu, x):
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ok = ~np.isnan(nu)
    nu, x = nu[ok], x[ok]
    try:
        y = pd.qcut(nu, 3, labels=False, duplicates="drop")
        if len(np.unique(y)) < 2:
            return np.nan, np.nan, np.nan, np.nan
        res = OrderedModel(y, x.reshape(-1, 1).astype(float), distr="logit") \
            .fit(method="bfgs", disp=False)
        b, se = res.params[0], res.bse[0]
        return b, b - 1.96 * se, b + 1.96 * se, res.pvalues[0]
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def regresion_ordinal(muestras, especies=None, a="SS19", b="SS37"):
    """Proportional-odds model of nu-tertile on scenario (b vs a), per species."""
    especies = especies or (TIERS["T1"] + TIERS["T2"])
    filas = []
    for sp in especies:
        s = muestras[(muestras.especie == sp) & muestras.escenario.isin([a, b])]
        if s.empty:
            continue
        beta, lo, hi, p = _beta_ordinal(s["nu"].values, (s["escenario"] == b).values)
        filas.append(dict(especie=sp, beta=beta, ic_inf=lo, ic_sup=hi, p=p))
    return pd.DataFrame(filas)


def control_escenarios(model, mapa, B=10, n=30, eps=1e-6, cache_dir="cache",
                       especies=None, semilla=11, inicio=0, **kw):
    """Random-perturbation control: in each replicate, restrict the same number
    of NON-tier exchange coordinates as SS19 (|J_T2|) and SS37 (|J_T1|), chosen
    at random, and estimate the same ordinal coefficient.
    Cost: 2*B*n linear programs."""
    rng = np.random.default_rng(semilla)
    J = _tier_sets(mapa)
    fuera = sorted(set(range(mapa.m)) - (J["T0"] | J["T1"] | J["T2"]))
    n19, n37 = len(J["T2"]), len(J["T1"])
    if len(fuera) < max(n19, n37):
        warnings.warn("not enough non-tier coordinates; sampling with replacement from all")
        fuera = list(range(mapa.m))
    especies = especies or (TIERS["T1"] + TIERS["T2"])
    filas = []
    for rep in range(inicio, inicio + B):
        rng = np.random.default_rng(semilla + 7919 * rep)
        R19 = list(rng.choice(fuera, n19, replace=False))
        R37 = list(rng.choice(fuera, n37, replace=False))
        F19 = muestrear(model, mapa, "ctrl", n, cache_dir, restringir=R19,
                        semilla=semilla + rep, etiqueta="ctrl%03dA" % rep, **kw)
        F37 = muestrear(model, mapa, "ctrl", n, cache_dir, restringir=R37,
                        semilla=semilla + 1000 + rep, etiqueta="ctrl%03dB" % rep, **kw)
        df = tabla_muestras(mapa, {"SS19": F19, "SS37": F37}, eps=eps, especies=especies)
        r = regresion_ordinal(df, especies)
        r["rep"] = rep
        filas.append(r)
        _log("control replicate %d done" % rep)
    return pd.concat(filas, ignore_index=True)


def comparar_con_control(obs, ctrl):
    out = []
    for _, r in obs.iterrows():
        c = ctrl[ctrl.especie == r.especie]["beta"].dropna()
        out.append(dict(especie=r.especie, beta=r.beta, ic_inf=r.ic_inf, ic_sup=r.ic_sup,
                        p=r.p, beta_ctrl=c.mean() if len(c) else np.nan,
                        p_ctrl=float((c.abs() >= abs(r.beta)).mean()) if len(c) else np.nan))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 8. FVA envelope (Reviewer 3)
# ----------------------------------------------------------------------
def envolvente_fva(model, mapa, escenario="SS19", especies=None, frac_opt=0.9,
                   eps=1e-6, factor=0.1, modo="ub", objetivo="community_growth",
                   processes=1, cache_dir="cache"):
    """Largest support compatible with >= frac_opt optimal growth. Coordinate j
    is in the envelope of species i if some interface reaction of i touching j
    can carry |flux| > eps. Cost: 2 LPs per interface reaction (use Gurobi)."""
    from cobra.flux_analysis import flux_variability_analysis
    os.makedirs(cache_dir, exist_ok=True)
    especies = especies or VALIDADAS
    ruta = os.path.join(cache_dir, "fva_%s.csv" % escenario)
    rxns = sorted({rid for a in especies if mapa.token(a) in mapa.especies
                   for rid in mapa.rxns_especie[mapa.token(a)]})
    if os.path.exists(ruta):
        fva = pd.read_csv(ruta, index_col=0)
        _log("FVA %s loaded from cache" % escenario)
    else:
        _log("FVA %s on %d reactions (2 LPs each) ..." % (escenario, len(rxns)))
        with model:
            restr = conjunto_tier(mapa, ESCENARIOS[escenario]) if escenario in ESCENARIOS else []
            _aplicar_restriccion(model, mapa, restr, factor, modo)
            model.objective = objetivo
            fva = flux_variability_analysis(model, reaction_list=rxns,
                                            fraction_of_optimum=frac_opt,
                                            processes=processes)
        fva.to_csv(ruta)
    amp = fva[["minimum", "maximum"]].abs().max(axis=1)
    activas = set(amp[amp > eps].index)
    col = {rid: k for k, rid in enumerate(mapa.cols)}
    Wc = mapa.W.tocsc()
    filas = []
    for a in especies:
        t = mapa.token(a)
        if t not in mapa.especies:
            continue
        i = mapa.especies.index(t)
        A = np.zeros((1, mapa.m), bool)
        for rid in mapa.rxns_especie[t]:
            if rid in activas:
                sub = Wc[:, col[rid]].tocoo()
                for row in sub.row:
                    if row // mapa.m == i:
                        A[0, row % mapa.m] = True
        d, c, nu, z = descriptores(A, orden_bandera(mapa, excluir=t))
        filas.append(dict(especie=a, escenario=escenario, d=int(d[0]),
                          codim=float(c[0]), nu=nu[0], z=z[0]))
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------
# 9. Persistent homology
# ----------------------------------------------------------------------
def homologia_persistente(mapa, F, especies=None, guardar_bottleneck=None):
    """H0/H1 Vietoris-Rips persistence of each species' sampled net-exchange
    vectors; pairwise bottleneck distances between H1 diagrams."""
    from ripser import ripser
    from persim import bottleneck
    especies = especies or (TIERS["T1"] + TIERS["T2"])
    dgms, filas = {}, []
    for a in especies:
        t = mapa.token(a)
        if t not in mapa.especies:
            continue
        X = F[:, mapa.especies.index(t), :]
        X = X[:, X.std(axis=0) > 0] if (X.std(axis=0) > 0).any() else X[:, :1]
        D = ripser(X, maxdim=1)["dgms"]
        h0 = D[0][np.isfinite(D[0][:, 1])]
        h1 = D[1]
        dgms[a] = h1 if len(h1) else np.array([[0.0, 0.0]])
        filas.append(dict(especie=a,
                          persistencia_H0=float((h0[:, 1] - h0[:, 0]).sum()),
                          persistencia_H1=float((h1[:, 1] - h1[:, 0]).sum()) if len(h1) else 0.0,
                          n_ciclos_H1=int(len(h1))))
    nombres = list(dgms)
    M = np.zeros((len(nombres), len(nombres)))
    for p in range(len(nombres)):
        for q in range(p + 1, len(nombres)):
            M[p, q] = M[q, p] = bottleneck(dgms[nombres[p]], dgms[nombres[q]])
    Dm = pd.DataFrame(M, index=nombres, columns=nombres)
    if guardar_bottleneck:
        Dm.to_csv(guardar_bottleneck)
        _log("written %s" % guardar_bottleneck)
    return pd.DataFrame(filas), Dm


# ----------------------------------------------------------------------
# 10. Tolerance table template
# ----------------------------------------------------------------------
def plantilla_tolerancia(ruta="tolerancia.csv", especies=None):
    """Write a template only if the file does not exist. Fill 'tolerancia'
    (NaCl tolerance breadth, same units for all species) and 'alta_confianza'
    (1 = direct experimental measurement) from the literature you compiled."""
    if os.path.exists(ruta):
        _log("%s already exists; not overwritten" % ruta)
        return pd.read_csv(ruta)
    especies = especies or list(NOMBRES.values())
    df = pd.DataFrame(dict(especie=especies, tolerancia=np.nan,
                           alta_confianza=np.nan, fuente=""))
    df.to_csv(ruta, index=False)
    _log("template written to %s: fill in the values before continuing" % ruta)
    return df


# ----------------------------------------------------------------------
# 0. SBML repair (community12.xml from nested community merges)
# ----------------------------------------------------------------------
def reparar_sbml(entrada, salida, compartimento="ext"):
    """Write a corrected copy of a merged community SBML:
      (a) declares every species referenced by a reaction but never declared
          (e.g. 'community_biomass_new_community');
      (b) sets boundaryCondition="false" on all species, so COBRApy does not
          add free exchange reactions on top of the model's own R_EX_*
          (and cannot create biomass from nothing).
    The original file is not modified."""
    txt = open(entrada, encoding="utf-8").read()
    declaradas = set(re.findall(r'<species [^>]*?\bid="([^"]+)"', txt))
    usadas = set(re.findall(r'<speciesReference [^>]*?species="([^"]+)"', txt))
    faltan = sorted(usadas - declaradas)
    nuevos = "".join(
        '\n      <species metaid="meta_%s" id="%s" name="%s" compartment="%s" '
        'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
        % (s, s, s, compartimento) for s in faltan)
    txt = txt.replace("<listOfSpecies>", "<listOfSpecies>" + nuevos, 1)
    n_bc = txt.count('boundaryCondition="true"')
    txt = txt.replace('boundaryCondition="true"', 'boundaryCondition="false"')
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write(txt)
    _log("repaired SBML -> %s: %d species declared %s; %d boundaryCondition flags cleared"
         % (salida, len(faltan), faltan, n_bc))
    return faltan, n_bc


# ----------------------------------------------------------------------
# Medium (the SBML has every exchange open at -1000)
# ----------------------------------------------------------------------
# M9 minimal medium as in CarveMe's media database, with glycerol as the
# only carbon source. Edit if the paper's runs used other bounds.
M9_GLICEROL = {c: -1000.0 for c in
               ["ca2", "cl", "co2", "cobalt2", "cu2", "fe2", "fe3", "h", "h2o", "k",
                "mg2", "mn2", "mobd", "na1", "nh4", "ni2", "o2", "pi", "so4", "zn2"]}
M9_GLICEROL["glyc"] = -10.0


def aplicar_medio(model, medio=None, prefijo="EX_", sufijo="_e", verbose=True):
    """Close every exchange uptake, then open those in `medio`
    ({compound: lower bound}). Secretion (upper bounds) is left unchanged."""
    medio = M9_GLICEROL if medio is None else medio
    for r in model.reactions:
        if r.boundary and r.id.startswith(prefijo):
            r.lower_bound = 0.0
    faltan = []
    for c, lb in medio.items():
        rid = prefijo + c + sufijo
        if rid in model.reactions:
            model.reactions.get_by_id(rid).lower_bound = lb
        else:
            faltan.append(rid)
    if verbose:
        _log("medium applied: %d compounds open%s"
             % (len(medio) - len(faltan), "; missing: %s" % faltan if faltan else ""))
    return faltan
