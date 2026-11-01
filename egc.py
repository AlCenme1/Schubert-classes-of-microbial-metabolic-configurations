import cobra, curar_modelo as cm
from cobra.flux_analysis import pfba

def egc_ciclos(m):
    out = {}
    with m:
        for r in m.reactions:
            if r.boundary: r.bounds = (0, 0)
        for k, (izq, der) in cm.ENERGIA.items():
            mets = {}
            if any(mid not in m.metabolites for mid in list(izq) + list(der)): continue
            for mid, c in list(izq.items()) + list(der.items()):
                mets[m.metabolites.get_by_id(mid)] = c
            d = cobra.Reaction("DISS_" + k, lower_bound=0, upper_bound=1000); d.add_metabolites(mets)
            with m:
                m.add_reactions([d]); m.objective = d
                if m.slim_optimize(error_value=0.0) > 1e-6:
                    d.bounds = (1, 1)
                    s = pfba(m).fluxes
                    out[k] = {i: round(x, 2) for i, x in s[(s.abs() > 1e-6) & (s.index != d.id)].items()}
    return out

