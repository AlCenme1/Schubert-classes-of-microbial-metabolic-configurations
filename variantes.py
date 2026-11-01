"""Mean nu per validated taxon under alternative references (eps = 0.02)."""
import pickle, numpy as np, pandas as pd, pipeline_schubert as ps
EPS = 0.02
mapa = pickle.load(open("mapa_todas.pkl", "rb"))
F = {e: np.load("cache/flujos_%s_n150.npz" % e)["F"] for e in ("SS19", "SS37")}
rng = np.random.default_rng(5)
rows = []
for a in ps.VALIDADAS:
    t = mapa.token(a); i = mapa.especies.index(t)
    A = np.concatenate([np.abs(F[e][:, i, :]) > EPS for e in F])
    A = A[A.sum(1) > 0]
    if len(A) == 0:
        continue
    r = {"especie": a, "estados_activos": len(A)}
    for v in ("original", "permutada", "invertida", "aleatoria"):
        k = 1 if v in ("original", "invertida") else 200
        r[v] = np.mean([np.nanmean(ps.descriptores(A, ps.orden_bandera(mapa, excluir=t, variante=v, rng=rng))[2]) for _ in range(k)])
    freq = np.zeros(mapa.m)
    for tt in mapa.especies:
        if tt != t: freq[mapa.interfaz[tt]] += 1
    orden = sorted(range(mapa.m), key=lambda j: (-freq[j], mapa.exch_ids[j]))
    rank = np.empty(mapa.m); rank[orden] = np.arange(1, mapa.m + 1)
    r["solo_frecuencia"] = np.nanmean(ps.descriptores(A, rank)[2])
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv("resultados/nu_por_variante.csv", index=False)
print(df.round(3).to_string()); print(df.mean(numeric_only=True).round(3).to_string())
