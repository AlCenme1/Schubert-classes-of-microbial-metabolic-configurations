"""Thermodynamic directionality rules for gap-filled CarveMe models."""
import re

OXIDANTES = re.compile(r"^(o2|h2o2)_[cp](_|$)")   # cytosolic / periplasmic O2 and H2O2

def _base(mid):
    return mid.split("_")[0]

def es_transporte(r):
    """Same metabolite in two compartments (full id minus compartment/species suffix)."""
    from cargas import base_met
    bases = [base_met(m.id) for m in r.metabolites]
    return len(bases) != len(set(bases))

def reversibles_con_oxidante(model):
    """Reversible non-transport reactions that consume O2 or H2O2 on one side.
    Returns [(reaction, sign)]: sign = direction in which the oxidant is CONSUMED."""
    out = []
    for r in model.reactions:
        if not (r.lower_bound < 0 < r.upper_bound) or r.boundary or es_transporte(r):
            continue
        ox = [(m, c) for m, c in r.metabolites.items() if OXIDANTES.match(m.id)]
        if not ox:
            continue
        signos = {1 if c < 0 else -1 for m, c in ox}
        if len(signos) == 1:          # O2/H2O2 only on one side
            out.append((r, signos.pop()))
    return out

def fijar_direccion(r, signo):
    """Make reaction irreversible in direction `signo` (+1 forward, -1 backward)."""
    if signo > 0:
        r.lower_bound = 0.0
    else:
        r.upper_bound = 0.0
