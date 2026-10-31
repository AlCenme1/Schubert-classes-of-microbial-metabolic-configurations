# -*- coding: utf-8 -*-
"""Synthetic validation: raw |lambda| vs normalized nu under three regimes.

positional : d_i independent of z_i; active positions concentrated early in
             the flag in proportion to z_i (real structural signal)
confounded : d_i increases with z_i; positions uniformly random (no signal)
null       : d_i independent of z_i; positions uniformly random
"""
import numpy as np
import pandas as pd
from scipy import stats


def estado(rng, m, d, z, beta):
    if beta == 0:
        return rng.choice(m, d, replace=False) + 1
    w = np.exp(beta * z * (1 - np.arange(m) / m))
    return rng.choice(m, d, replace=False, p=w / w.sum()) + 1


def descriptores(pos, m):
    d = len(pos)
    codim = d * (m - d) + d * (d + 1) / 2 - pos.sum()
    return codim, codim / (d * (m - d))


def comunidad(rng, n_s, m, regimen, gamma, N=20):
    z = rng.uniform(0, 1, n_s)
    if regimen == "confounded":
        d = np.clip(np.round(8 + 140 * z + rng.normal(0, 10, n_s)), 3, m - 3).astype(int)
        beta = 0.0
    else:
        d = rng.integers(8, 150, n_s)
        beta = 0.5 * gamma if regimen == "positional" else 0.0
    raw, nu = np.zeros(n_s), np.zeros(n_s)
    for i in range(n_s):
        vals = [descriptores(estado(rng, m, d[i], z[i], beta), m) for _ in range(N)]
        raw[i] = np.mean([v[0] for v in vals])
        nu[i] = np.mean([v[1] for v in vals])
    return z, raw, nu


def estudio(m=553, ns=(4, 6, 8, 11, 20, 50), reps=1500, gamma=0.7, alpha=0.05,
            semilla=2026, N=20):
    rng = np.random.default_rng(semilla)
    filas = []
    for regimen in ("positional", "confounded", "null"):
        for n_s in ns:
            sig_raw = sig_nu = 0
            rho_raw, rho_nu = [], []
            for _ in range(reps):
                z, raw, nu = comunidad(rng, n_s, m, regimen, gamma, N)
                r1, p1 = stats.spearmanr(raw, z)
                r2, p2 = stats.spearmanr(nu, z)
                sig_raw += p1 < alpha
                sig_nu += p2 < alpha
                rho_raw.append(r1)
                rho_nu.append(r2)
            filas.append(dict(regimen=regimen, n_s=n_s,
                              rechazo_raw=sig_raw / reps, rechazo_nu=sig_nu / reps,
                              rho_raw_mediana=np.nanmedian(rho_raw),
                              rho_nu_mediana=np.nanmedian(rho_nu)))
            print(filas[-1], flush=True)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    import sys
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    df = estudio(reps=reps)
    df.to_csv("resultados/sintetico.csv", index=False)
    print(df.round(3).to_string())
