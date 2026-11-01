# -*- coding: utf-8 -*-
"""
curar_modelo.py -- thermodynamic and structural curation of the community GSMM
(community12.xml) used in the Schubert manuscript.

Steps (each one logged in a table of changes):
  1. Oxidases / oxygenases / peroxidases: reversible reactions that consume
     O2 or H2O2 on one side are made irreversible in the oxidant-consuming
     direction.
  2. Specific energy-generating cycles found by the Fritzemeier test
     (e.g. NAMNPP, ATP-coupled nicotinate phosphoribosyltransferase).
  3. Fritzemeier et al. (2017) test per species for energy metabolites in a
     closed system; returns any remaining cycles.

Use from Jupyter:
    import curar_modelo as cm
    model = cm.cargar("community12_fixed.xml")
    cambios = cm.curar(model)
    egc = cm.test_egc(model)          # must be empty
    cm.guardar(model, "community12_curated.xml")
"""
import re
import time
import cobra
import pandas as pd

from reglas import reversibles_con_oxidante, fijar_direccion

# reaction base id -> direction (+1: forward as written, -1: backward)
# found by test_egc(); the direction is the thermodynamically favourable one
FIJAS = {
    "NAMNPP": +1,   # ATP + nac + prpp -> ADP + Pi + nicrnt + ppi (ATP-driven)
    "GCDH": +1,     # glutaryl-CoA dehydrogenase (decarboxylating), irreversible
    "NAR_syn": +1,  # respiratory nitrate reductase, nitrate -> nitrite
}

ENERGIA = {
    "ATP":   ({"atp_c": -1, "h2o_c": -1}, {"adp_c": 1, "pi_c": 1, "h_c": 1}),
    "CTP":   ({"ctp_c": -1, "h2o_c": -1}, {"cdp_c": 1, "pi_c": 1, "h_c": 1}),
    "GTP":   ({"gtp_c": -1, "h2o_c": -1}, {"gdp_c": 1, "pi_c": 1, "h_c": 1}),
    "UTP":   ({"utp_c": -1, "h2o_c": -1}, {"udp_c": 1, "pi_c": 1, "h_c": 1}),
    "NADH":  ({"nadh_c": -1}, {"nad_c": 1, "h_c": 1}),
    "NADPH": ({"nadph_c": -1}, {"nadp_c": 1, "h_c": 1}),
    "FADH2": ({"fadh2_c": -1}, {"fad_c": 1, "h_c": 2}),
    "Q8H2":  ({"q8h2_c": -1}, {"q8_c": 1, "h_c": 2}),
    "MQL8":  ({"mql8_c": -1}, {"mqn8_c": 1, "h_c": 2}),
    "ACCOA": ({"accoa_c": -1, "h2o_c": -1}, {"ac_c": 1, "coa_c": 1, "h_c": 1}),
    "GLU":   ({"glu__L_c": -1, "h2o_c": -1}, {"akg_c": 1, "nh4_c": 1, "h_c": 2}),
    "PMF":   ({"h_p": -1}, {"h_c": 1}),
}


def _log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def cargar(ruta, solver="hybrid"):
    m = cobra.io.read_sbml_model(ruta)
    try:
        m.solver = solver
    except Exception:
        m.solver = "glpk"
    return m


def sufijos(model):
    return [r.id[len("Growth_"):] for r in model.reactions if r.id.startswith("Growth_")]


def especie_de(rid, sufs):
    for s in sufs:
        if rid.endswith("_" + s):
            return s
    return None


def _direccion(r):
    if r.lower_bound < 0 < r.upper_bound:
        return "rev"
    return "fwd" if r.lower_bound >= 0 else "bwd"


def armonizar_direccionalidad(model, referencia_sufijo="_community_new_community", minimo=2):
    """Reactions reversible in a species but irreversible (in a single,
    consistent direction) in >= `minimo` reference reconstructions, and
    reversible in none of them, take the reference direction. The reference
    species are the CarveMe reconstructions (directionality from the curated
    BiGG universal model)."""
    sufs = sufijos(model)
    ref = [s for s in sufs if s.endswith(referencia_sufijo)]
    dirs = {}
    for r in model.reactions:
        s = especie_de(r.id, sufs)
        if s in ref and not r.boundary:
            dirs.setdefault(r.id[: -len(s) - 1], []).append(_direccion(r))
    consenso = {}
    for base, ds in dirs.items():
        if "rev" in ds or len(ds) < minimo:
            continue
        if len(set(ds)) == 1:
            consenso[base] = +1 if ds[0] == "fwd" else -1
    cambios = []
    for r in model.reactions:
        s = especie_de(r.id, sufs)
        if s is None or s in ref or r.boundary:
            continue
        base = r.id[: -len(s) - 1]
        if base in consenso and _direccion(r) == "rev":
            antes = r.bounds
            fijar_direccion(r, consenso[base])
            cambios.append(dict(reaccion=r.id, base=base, especie=s.split("_")[0],
                                regla="directionality harmonized with CarveMe reconstructions",
                                antes=antes, despues=r.bounds))
    return cambios


def reversibles_desbalanceadas(model, tol=1e-6):
    """Reversible reactions that remain charge- or element-imbalanced after the
    charge repair (excluding pseudo-metabolites such as biomass). Run in reverse,
    they can act as electron sources; they are fixed to the written direction."""
    out = []
    for r in model.reactions:
        if r.boundary or not (r.lower_bound < 0 < r.upper_bound):
            continue
        if any("biomass" in m.id for m in r.metabolites):
            continue
        b = {k: v for k, v in r.check_mass_balance().items() if abs(v) > tol}
        if b:
            out.append((r, b))
    return out


def curar(model, fijas=None):
    """Apply steps 1-2 in place. Returns a DataFrame of changes."""
    fijas = dict(FIJAS, **(fijas or {}))
    sufs = sufijos(model)
    cambios = armonizar_direccionalidad(model)
    for r, signo in reversibles_con_oxidante(model):
        antes = r.bounds
        fijar_direccion(r, signo)
        cambios.append(dict(reaccion=r.id, base=r.id.split("_")[0],
                            especie=(especie_de(r.id, sufs) or "").split("_")[0],
                            regla="oxidase/oxygenase irreversible", antes=antes, despues=r.bounds))
    for r in model.reactions:
        b = r.id.split("_")[0]
        if b in fijas and r.lower_bound < 0 < r.upper_bound:
            antes = r.bounds
            fijar_direccion(r, fijas[b])
            cambios.append(dict(reaccion=r.id, base=b,
                                especie=(especie_de(r.id, sufs) or "").split("_")[0],
                                regla="energy-generating cycle (Fritzemeier test)",
                                antes=antes, despues=r.bounds))
    for r, bal in reversibles_desbalanceadas(model):
        antes = r.bounds
        fijar_direccion(r, +1)
        cambios.append(dict(reaccion=r.id, base=r.id.split("_")[0],
                            especie=(especie_de(r.id, sufs) or "").split("_")[0],
                            regla="reversible and imbalanced %s: written direction only"
                                  % sorted(bal), antes=antes, despues=r.bounds))
    _log("curation: %d reactions changed" % len(cambios))
    return pd.DataFrame(cambios)


def extraer_especie(model, suf):
    sp = cobra.Model("sp_" + suf)
    rx = [r for r in model.reactions if r.id.endswith("_" + suf)]
    ex = [r for r in model.reactions if r.boundary and r.id.startswith("EX_")]
    sp.add_reactions([r.copy() for r in rx + ex])
    for met in list(sp.metabolites):
        if "biomass" in met.id:
            sp.add_boundary(met, type="sink")
    sp.objective = "Growth_" + suf
    sp.solver = "glpk"
    return sp


def test_egc(model, especies=None, tol=1e-6, devolver_ciclos=True):
    """Fritzemeier et al. (2017): with every boundary closed, maximize a
    dissipation reaction for each energy metabolite in each species. Any
    positive value is an energy-generating cycle. Returns a DataFrame."""
    from cobra.flux_analysis import pfba
    sufs = especies or sufijos(model)
    filas = []
    for suf in sufs:
        sp = extraer_especie(model, suf)
        for r in sp.reactions:
            if r.boundary:
                r.bounds = (0, 0)
        for nombre, (izq, der) in ENERGIA.items():
            mets = {}
            ok = True
            for k, c in list(izq.items()) + list(der.items()):
                mid = "%s_%s" % (k, suf)
                if mid not in sp.metabolites:
                    ok = False
                    break
                mets[sp.metabolites.get_by_id(mid)] = c
            if not ok:
                continue
            d = cobra.Reaction("DISS_" + nombre, lower_bound=0, upper_bound=1000)
            d.add_metabolites(mets)
            with sp:
                sp.add_reactions([d])
                sp.objective = d
                v = sp.slim_optimize(error_value=0.0)
                ciclo = ""
                if v > tol and devolver_ciclos:
                    d.bounds = (1, 1)
                    try:
                        s = pfba(sp).fluxes
                        act = s[(s.abs() > 1e-6) & (s.index != d.id)]
                        ciclo = "; ".join("%s(%+.2f)" % (i.replace("_" + suf, ""), x)
                                          for i, x in act.items())
                    except Exception:
                        ciclo = "?"
            if v > tol:
                filas.append(dict(especie=suf.split("_")[0], metabolito=nombre,
                                  flujo=round(v, 3), ciclo=ciclo))
        _log("EGC test %s: %d cycles" % (suf.split("_")[0],
                                         sum(1 for f in filas if f["especie"] == suf.split("_")[0])))
    return pd.DataFrame(filas, columns=["especie", "metabolito", "flujo", "ciclo"])


def crecimiento_por_especie(model, medio, sin_carbono=("glyc",)):
    """Growth of each species alone on `medio` and on `medio` without carbon."""
    import pipeline_schubert as ps
    filas = []
    for suf in sufijos(model):
        sp = extraer_especie(model, suf)
        with sp:
            ps.aplicar_medio(sp, medio, verbose=False)
            g = sp.slim_optimize(error_value=0.0)
        with sp:
            m2 = {k: v for k, v in medio.items() if k not in sin_carbono}
            ps.aplicar_medio(sp, m2, verbose=False)
            a = sp.slim_optimize(error_value=0.0)
        filas.append(dict(especie=suf.split("_")[0], n_rxn=len(sp.reactions),
                          crec_medio=round(g, 4), crec_sin_carbono=round(a, 4)))
    return pd.DataFrame(filas)


def guardar(model, ruta):
    cobra.io.write_sbml_model(model, ruta)
    _log("written %s" % ruta)


def curar_completo(model, ruta_sbml_original):
    """Full curation in the order used in the manuscript:
       charges (inference) -> directionality harmonization -> oxidases ->
       specific EGC fixes -> reversible imbalanced reactions."""
    from cargas import marcar_desconocidas, inferir_cargas, resolver_conjunto
    n = marcar_desconocidas(model, ruta_sbml_original)
    inf1 = inferir_cargas(model, verbose=False)
    inf2 = resolver_conjunto(model, verbose=False)
    inf3 = inferir_cargas(model, verbose=False)
    _log("charges: %d metabolites without charge; %d base metabolites inferred"
         % (n, len(inf1) + len(inf2) + len(inf3)))
    return curar(model)


def acoplar(model, c=400.0, u=0.01, solo_interfaz=True, mapa=None):
    """Coupling constraints (Heinken et al. 2013): for every reaction r of
    species i,  -(c*mu_i + u) <= v_r <= c*mu_i + u , where mu_i is the flux of
    Growth_i. With solo_interfaz=True only the reactions that exchange shared
    extracellular metabolites are coupled (enough to prevent non-growing
    species from feeding the others; ~9,000 constraints instead of ~45,000).
    Constraints are added to the optlang problem and saved with the model
    only in memory; call it after loading."""
    sufs = sufijos(model)
    bio = {s: model.reactions.get_by_id("Growth_" + s) for s in sufs}
    if solo_interfaz:
        if mapa is None:
            import pipeline_schubert as ps
            mapa = ps.construir_mapa(model, verbose=False)
        rxns = [model.reactions.get_by_id(rid) for t in mapa.especies for rid in mapa.rxns_especie[t]]
    else:
        rxns = [r for r in model.reactions if especie_de(r.id, sufs) and not r.id.startswith("Growth_")]
    cons = []
    for r in rxns:
        s = especie_de(r.id, sufs)
        if s is None:
            continue
        mu = bio[s].flux_expression
        cons.append(model.problem.Constraint(r.flux_expression - c * mu, ub=u, name="cpl_up_" + r.id))
        cons.append(model.problem.Constraint(-r.flux_expression - c * mu, ub=u, name="cpl_lo_" + r.id))
    model.add_cons_vars(cons, sloppy=True)
    model.solver.update()
    _log("coupling: %d constraints (c=%g, u=%g) on %d reactions" % (len(cons), c, u, len(rxns)))
    return len(cons)
