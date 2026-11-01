# -*- coding: utf-8 -*-
"""
Replace taxa in the curated community model by individual reconstructions.

Each individual model (CarveMe SBML, compartments c/p/e, biomass 'Growth') is
inserted with the suffix _<Token>: its extracellular metabolites are mapped to
the shared pool (compartment 'ext'), other metabolites and compartments are
suffixed, its own EX_ reactions are dropped (the community exchanges replace
them; a community EX_ is created for any new extracellular metabolite, closed
for uptake), and Growth_<Token> produces community_biomass.
"""
import pickle
import time
import cobra

import os
from config_v2 import PKL, COMUNIDAD_PREVIA

# suffix in the old community -> (individual model, new token)
REEMPLAZOS = {
    "Arhodomonas": ("Arhodomonas", "Arhodomonas"),
    "Gracilimonas": ("Gracilimonas", "Gracilimonas"),
    "Oceanicola": ("Oceanicola", "Oceanicola"),
    "Puniceicoccus": ("Puniceicoccus", "Puniceicoccus"),
    "Roseavarius": ("Roseavarius", "Roseovarius"),
    "Spiribacter_new_community": ("Spiribacter", "Spiribacter"),
    "halomicrobium_new_community": ("Halomicrobium", "Halomicrobium"),
    "salinibacter_community_new_community": ("iSRU780", "Salinibacter"),
    "Haloquadratum_community_new_community": ("iHQW-TK", "Haloquadratum"),
}


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def quitar_especie(com, suf):
    rx = [r for r in com.reactions if r.id.endswith("_" + suf)]
    com.remove_reactions(rx, remove_orphans=False)
    mets = [m for m in com.metabolites if m.id.endswith("_" + suf) and not m.reactions]
    com.remove_metabolites(mets)
    for c in [c for c in list(com.compartments) if c.endswith("_" + suf)]:
        try:
            del com.compartments[c]
        except Exception:
            pass
    return len(rx), len(mets)


def insertar(com, ind, tok):
    shared = {m.id for m in com.metabolites if m.compartment == "ext"}
    mapa, nuevos_ext = {}, []
    comps = {}
    for x in ind.metabolites:
        if x.compartment in ("e", "C_e"):
            if x.id not in shared:
                y = cobra.Metabolite(x.id, formula=x.formula, name=x.name,
                                     charge=x.charge, compartment="ext")
                nuevos_ext.append(y)
                shared.add(x.id)
                mapa[x.id] = y
            else:
                mapa[x.id] = None          # resolved after adding
        else:
            cid = "C_%s_%s" % (x.compartment.replace("C_", ""), tok)
            comps[cid] = "%s (%s)" % (ind.compartments.get(x.compartment, x.compartment), tok)
            mapa[x.id] = cobra.Metabolite("%s_%s" % (x.id, tok), formula=x.formula,
                                          name=x.name, charge=x.charge, compartment=cid)
    if nuevos_ext:
        com.add_metabolites(nuevos_ext)
    com.compartments = dict(com.compartments, **comps)
    nuevos_int = [m for k, m in mapa.items() if m is not None and m.compartment != "ext"]
    com.add_metabolites(nuevos_int)
    def met(x):
        y = mapa[x.id]
        return com.metabolites.get_by_id(x.id) if y is None or y.compartment == "ext" else \
            com.metabolites.get_by_id(y.id)
    nuevas = []
    for r in ind.reactions:
        if r.boundary and r.id.startswith("EX_"):
            continue
        nr = cobra.Reaction("%s_%s" % (r.id, tok), name=r.name,
                            lower_bound=r.lower_bound, upper_bound=r.upper_bound)
        nr.add_metabolites({met(x): c for x, c in r.metabolites.items()})
        if r.gene_reaction_rule:
            gpr = r.gene_reaction_rule
            for g in sorted(r.genes, key=lambda g: -len(g.id)):
                gpr = gpr.replace(g.id, "%s__%s" % (g.id, tok))
            nr.gene_reaction_rule = gpr
        nuevas.append(nr)
    com.add_reactions(nuevas)
    g = com.reactions.get_by_id("Growth_%s" % tok)
    g.add_metabolites({com.metabolites.get_by_id("community_biomass"): 1.0})
    # community exchange for new shared metabolites (closed for uptake)
    ex = []
    for y in nuevos_ext:
        rid = "EX_" + y.id
        if rid not in com.reactions:
            e = cobra.Reaction(rid, name="exchange " + y.id, lower_bound=0.0, upper_bound=1000.0)
            e.add_metabolites({com.metabolites.get_by_id(y.id): -1.0})
            ex.append(e)
    com.add_reactions(ex)
    return len(nuevas), len(nuevos_ext)


if __name__ == "__main__":
    com = cobra.io.read_sbml_model(COMUNIDAD_PREVIA)
    log("community loaded: %d reactions" % len(com.reactions))
    for suf, (archivo, tok) in REEMPLAZOS.items():
        nr, nm = quitar_especie(com, suf)
        log("removed %-38s %5d reactions, %5d metabolites" % (suf, nr, nm))
    for suf, (archivo, tok) in REEMPLAZOS.items():
        ind = pickle.load(open("ind2_%s.pkl" % archivo, "rb"))   # written by curar_indiv2.py
        nr, ne = insertar(com, ind, tok)
        log("inserted %-14s as _%-13s %5d reactions, %3d new shared metabolites" % (archivo, tok, nr, ne))
    com.id = "community17_v2"
    com.objective = "community_growth"
    pickle.dump(com, open(PKL, "wb"))
    cobra.io.write_sbml_model(com, "community17_v2_curated.xml")
    log("community v2: %d reactions, %d metabolites" % (len(com.reactions), len(com.metabolites)))
