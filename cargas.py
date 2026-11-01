"""Infer missing metabolite charges from reaction charge balance."""
import collections, re

COMP = re.compile(r"^(.*?)_(c|p|e)(?:_.*)?$")

def base_met(mid):
    """'g3p_c_salinibacter_x' -> 'g3p' ; 'glc__D_e' -> 'glc__D'."""
    mm = COMP.match(mid)
    return mm.group(1) if mm else mid

def inferir_cargas(model, max_iter=50, verbose=True):
    """Fill metabolites with charge None, assuming one charge per base metabolite
    (same formula in every compartment and species). Iteratively uses reactions
    whose only unknown is a single base metabolite. Returns {base: charge}."""
    conocidas = {}
    for met in model.metabolites:
        if met.charge is not None:
            conocidas.setdefault(base_met(met.id), collections.Counter())[int(met.charge)] += 1
    conocidas = {b: c.most_common(1)[0][0] for b, c in conocidas.items()}
    desconocidas = {base_met(m.id) for m in model.metabolites if m.charge is None} - set(conocidas)
    inferidas = {}
    for it in range(max_iter):
        votos = collections.defaultdict(collections.Counter)
        for r in model.reactions:
            if r.boundary:
                continue
            coef_u = collections.Counter()
            suma = 0.0
            for met, c in r.metabolites.items():
                b = base_met(met.id)
                if b in conocidas:
                    suma += c * conocidas[b]
                elif b in inferidas:
                    suma += c * inferidas[b]
                else:
                    coef_u[b] += c
            coef_u = {b: c for b, c in coef_u.items() if abs(c) > 1e-9}
            if len(coef_u) == 1:
                (b, c), = coef_u.items()
                q = -suma / c
                if abs(q - round(q)) < 1e-6:
                    votos[b][int(round(q))] += 1
        nuevos = 0
        for b, cnt in votos.items():
            q, n = cnt.most_common(1)[0]
            if n >= max(1, 0.6 * sum(cnt.values())):   # clear majority among reactions
                inferidas[b] = q
                nuevos += 1
        if verbose:
            print("iteration %d: %d charges inferred (total %d / %d unknown)"
                  % (it + 1, nuevos, len(inferidas), len(desconocidas)))
        if nuevos == 0:
            break
    n = 0
    for met in model.metabolites:
        if met.charge is None:
            b = base_met(met.id)
            if b in inferidas:
                met.charge = inferidas[b]
                n += 1
    if verbose:
        sin = len({base_met(m.id) for m in model.metabolites if m.charge is None})
        print("metabolites filled: %d; base metabolites still without charge: %d" % (n, sin))
    return inferidas


def ids_sin_carga(ruta_sbml):
    """Species ids (cobra form, without the M_ prefix) that have no fbc:charge."""
    txt = open(ruta_sbml, encoding="utf-8").read()
    out = set()
    for tag in re.findall(r"<species [^>]*>", txt):
        if "fbc:charge=" not in tag:
            mid = re.search(r'\bid="([^"]+)"', tag).group(1)
            out.add(mid[2:] if mid.startswith("M_") else mid)
    return out


def marcar_desconocidas(model, ruta_sbml):
    ids = ids_sin_carga(ruta_sbml)
    n = 0
    for met in model.metabolites:
        if met.id in ids:
            met.charge = None
            n += 1
    return n


def resolver_conjunto(model, verbose=True):
    """Remaining unknown charges: integer least-squares over all reactions that
    involve them (one unknown per base metabolite). Accepts the solution only if
    it is integral and reduces the number of charge-imbalanced reactions."""
    import numpy as np
    unk = sorted({base_met(m.id) for m in model.metabolites if m.charge is None})
    if not unk:
        return {}
    idx = {b: i for i, b in enumerate(unk)}
    filas, rhs = [], []
    for r in model.reactions:
        if r.boundary:
            continue
        fila = np.zeros(len(unk)); suma = 0.0; tiene = False
        for met, c in r.metabolites.items():
            b = base_met(met.id)
            if met.charge is None:
                fila[idx[b]] += c; tiene = True
            else:
                suma += c * met.charge
        if tiene and np.abs(fila).sum() > 1e-9:
            filas.append(fila); rhs.append(-suma)
    A, y = np.array(filas), np.array(rhs)
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    xr = np.round(x)
    antes = int((np.abs(A @ np.zeros(len(unk)) - y) > 1e-6).sum())
    despues = int((np.abs(A @ xr - y) > 1e-6).sum())
    sol = {b: int(xr[i]) for b, i in idx.items() if abs(x[i] - xr[i]) < 0.25}
    if verbose:
        print("joint solution: %d/%d integral; imbalanced among these reactions %d -> %d"
              % (len(sol), len(unk), antes, despues))
    for met in model.metabolites:
        if met.charge is None and base_met(met.id) in sol:
            met.charge = sol[base_met(met.id)]
    return sol
