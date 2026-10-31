#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 1 — ¿Sobrevive la asociación codimensión ~ tolerancia a NaCl
cuando se controla la dimensión del subespacio d?

Base matemática (subespacio y bandera coordenados)
--------------------------------------------------
Sean p_1 < ... < p_d las posiciones (1..m) en la bandera de las reacciones
activas de un estado. El Algoritmo 1 da

    |lambda| = d(m-d) + d(d+1)/2 - sum_l p_l

que es exactamente el estadístico U de Mann-Whitney: el número de pares
(activa, inactiva) en que la reacción activa aparece ANTES en la bandera.
Por lo tanto:

    AUC = |lambda| / [d(m-d)]  en [0,1]
        = P(una reacción activa precede a una inactiva en la bandera)

Bajo una bandera aleatoria (permutación uniforme de las m reacciones):

    E[|lambda|]   = d(m-d)/2
    Var[|lambda|] = d(m-d)(m+1)/12

lo que da el z-score  z = (|lambda| - d(m-d)/2) / sqrt(d(m-d)(m+1)/12).

AUC y z eliminan la dependencia mecánica de d; lo que queda es la
"posición" de las reacciones activas respecto de la referencia ambiental.

Entradas
--------
--resultados : CSV con una fila por estado muestreado. Columnas mínimas:
               especie, codim, d   (nombres configurables)
               Alternativa: columna 'lambda' con la partición como texto
               "[l1, l2, ...]" o "l1;l2;..." -> d = longitud, codim = suma.
               Columna opcional: escenario.
--tolerancia : CSV con columnas especie, tolerancia (amplitud de tolerancia
               a NaCl). Opcional: columna 'alta_confianza' (1/0).

Salidas (en --salida)
---------------------
  por_especie.csv       resumen por especie
  correlaciones.csv     Spearman crudo / AUC / z / d y AUC parcial | d
  figura_paso1.pdf/png  figura de 4 paneles

Uso
---
  python paso1_codim_normalizada.py --resultados muestras.csv \
         --tolerancia tolerancia.csv --m 552 --salida paso1/
  python paso1_codim_normalizada.py --autotest

Compatible con Python 3.9.
"""
import argparse
import itertools
import math
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------
# Núcleo matemático
# ----------------------------------------------------------------------
def algoritmo1(posiciones_activas, m):
    """Implementación literal del Algoritmo 1 del manuscrito, para un
    subespacio coordenado y una bandera coordenada. Las posiciones son 1..m."""
    activas = set(int(p) for p in posiciones_activas)
    d = len(activas)
    delta = [0] * (m + 1)
    for j in range(1, m + 1):
        delta[j] = delta[j - 1] + (1 if j in activas else 0)
    lam = []
    for l in range(1, d + 1):
        j_l = min(j for j in range(m + 1) if delta[j] == l)
        lam.append((m - d) + l - j_l)
    return lam, sum(lam)


def codim_cerrada(posiciones_activas, m):
    p = np.asarray(posiciones_activas, dtype=float)
    d = p.size
    return d * (m - d) + d * (d + 1) / 2.0 - p.sum()


def normalizar(codim, d, m):
    """Devuelve (AUC, z) para arreglos de codimensión y dimensión."""
    codim = np.asarray(codim, dtype=float)
    d = np.asarray(d, dtype=float)
    rect = d * (m - d)
    with np.errstate(divide="ignore", invalid="ignore"):
        auc = np.where(rect > 0, codim / rect, np.nan)
        sd = np.sqrt(rect * (m + 1) / 12.0)
        z = np.where(sd > 0, (codim - rect / 2.0) / sd, np.nan)
    return auc, z


# ----------------------------------------------------------------------
# Estadística con n pequeño
# ----------------------------------------------------------------------
def _rangos(x):
    return stats.rankdata(x)


def spearman_perm(x, y, n_mc=200000, semilla=0):
    """Spearman con p-valor por permutación (exacto si n <= 9)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    n = x.size
    if n < 4:
        return np.nan, np.nan, n
    rx = _rangos(x) - _rangos(x).mean()
    ry = _rangos(y) - _rangos(y).mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    rho = float((rx * ry).sum() / den)
    if n <= 9:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
        nulos = (ry[perms] * rx).sum(axis=1) / den
    else:
        rng = np.random.default_rng(semilla)
        idx = np.argsort(rng.random((n_mc, n)), axis=1)
        nulos = (ry[idx] * rx).sum(axis=1) / den
    p = float((np.abs(nulos) >= abs(rho) - 1e-12).mean())
    return rho, p, n


def spearman_parcial_perm(x, y, z, n_mc=200000, semilla=0):
    """Spearman parcial rho(x, y | z): se residualizan los rangos de x e y
    sobre los rangos de z y se correlacionan los residuos. P-valor por
    permutación de y (Freedman-Lane simplificado: permuta residuos de y)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    ok = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[ok], y[ok], z[ok]
    n = x.size
    if n < 5:
        return np.nan, np.nan, n
    rx, ry, rz = _rangos(x), _rangos(y), _rangos(z)
    Z = np.column_stack([np.ones(n), rz])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    ex, ey = resid(rx), resid(ry)
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    if den == 0:
        return np.nan, np.nan, n
    rho = float((ex * ey).sum() / den)
    if n <= 9:
        perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    else:
        rng = np.random.default_rng(semilla)
        perms = np.argsort(rng.random((n_mc, n)), axis=1)
    nulos = (ey[perms] * ex).sum(axis=1) / den
    p = float((np.abs(nulos) >= abs(rho) - 1e-12).mean())
    return rho, p, n


# ----------------------------------------------------------------------
# Lectura de datos
# ----------------------------------------------------------------------
def _parse_lambda(s):
    nums = re.findall(r"-?\d+", str(s))
    return [int(v) for v in nums]


def cargar_resultados(ruta, col_esp, col_codim, col_d, col_lambda, col_esc):
    df = pd.read_csv(ruta)
    if col_lambda and col_lambda in df.columns:
        lams = df[col_lambda].map(_parse_lambda)
        df["d"] = lams.map(len)
        df["codim"] = lams.map(sum)
    else:
        faltan = [c for c in (col_codim, col_d) if c not in df.columns]
        if faltan:
            sys.exit("Faltan columnas en --resultados: %s. Columnas presentes: %s"
                     % (faltan, list(df.columns)))
        df = df.rename(columns={col_codim: "codim", col_d: "d"})
    df = df.rename(columns={col_esp: "especie"})
    if col_esc and col_esc in df.columns:
        df = df.rename(columns={col_esc: "escenario"})
    return df


# ----------------------------------------------------------------------
# Análisis
# ----------------------------------------------------------------------
def analizar(df, tol, m, salida, escenario=None, n_mc=200000):
    os.makedirs(salida, exist_ok=True)
    if escenario is not None:
        if "escenario" not in df.columns:
            sys.exit("Se pidió --escenario pero no hay columna de escenario.")
        df = df[df["escenario"].astype(str) == str(escenario)]

    df = df.copy()
    df = df[(df["d"] > 0) & (df["d"] < m)]
    df["auc"], df["z"] = normalizar(df["codim"], df["d"], m)

    g = df.groupby("especie")
    esp = pd.DataFrame({
        "n_estados": g.size(),
        "d_media": g["d"].mean(),
        "codim_media": g["codim"].mean(),
        "auc_media": g["auc"].mean(),
        "z_medio": g["z"].mean(),
        "codim_max_posible": g["d"].mean() * (m - g["d"].mean()),
    }).reset_index()

    tol = tol.rename(columns={tol.columns[0]: "especie"}) if "especie" not in tol.columns else tol
    esp = esp.merge(tol, on="especie", how="left")
    esp.to_csv(os.path.join(salida, "por_especie.csv"), index=False)

    con_tol = esp.dropna(subset=["tolerancia"])
    filas = []

    def bloque(sub, etiqueta):
        y = sub["tolerancia"].values
        for nombre, col in [("codim cruda", "codim_media"),
                            ("d (dimensión)", "d_media"),
                            ("AUC = codim/[d(m-d)]", "auc_media"),
                            ("z vs bandera aleatoria", "z_medio")]:
            r, p, n = spearman_perm(sub[col].values, y, n_mc=n_mc)
            filas.append(dict(subconjunto=etiqueta, prueba="tolerancia ~ " + nombre,
                              rho=r, p_perm=p, n=n))
        # Nota: NO se usa "codim cruda | d": codim y d son casi colineales
        # (rho ~ 0.99) y el residuo de rangos es ruido de empates, lo que da
        # falsos positivos (verificado en --autotest). La prueba correcta es
        # si la posición normalizada (AUC) aporta algo más allá de d.
        r, p, n = spearman_parcial_perm(sub["auc_media"].values, y,
                                        sub["d_media"].values, n_mc=n_mc)
        filas.append(dict(subconjunto=etiqueta,
                          prueba="tolerancia ~ AUC | d (parcial)",
                          rho=r, p_perm=p, n=n))
        r, p, n = spearman_perm(sub["codim_media"].values, sub["d_media"].values,
                                n_mc=n_mc)
        filas.append(dict(subconjunto=etiqueta, prueba="codim cruda ~ d",
                          rho=r, p_perm=p, n=n))

    bloque(con_tol, "todas con tolerancia")
    if "alta_confianza" in con_tol.columns:
        alta = con_tol[con_tol["alta_confianza"].astype(int) == 1]
        if len(alta) >= 4:
            bloque(alta, "alta confianza")

    corr = pd.DataFrame(filas)
    corr.to_csv(os.path.join(salida, "correlaciones.csv"), index=False)

    figura(df, con_tol, m, salida)
    return esp, corr


def figura(df, esp, m, salida):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.4))
    ax = axes.ravel()

    # (A) codim vs d, con la banda nula
    ax[0].scatter(df["d"], df["codim"], s=6, alpha=0.25, color="0.45",
                  label="estados muestreados")
    dd = np.linspace(1, max(df["d"].max(), 2), 200)
    mu = dd * (m - dd) / 2.0
    sd = np.sqrt(dd * (m - dd) * (m + 1) / 12.0)
    ax[0].plot(dd, mu, color="C3", lw=1.5, label="bandera aleatoria (media)")
    ax[0].fill_between(dd, mu - 2 * sd, mu + 2 * sd, color="C3", alpha=0.15,
                       label="± 2 DE")
    ax[0].plot(dd, dd * (m - dd), color="k", lw=0.8, ls="--", label="máximo d(m−d)")
    ax[0].set_xlabel("Dimensión del subespacio, $d$")
    ax[0].set_ylabel("Codimensión de Schubert $|\\lambda|$")
    ax[0].legend(fontsize=7, frameon=False)

    paneles = [("codim_media", "Codimensión media (cruda)"),
               ("auc_media", "AUC $=|\\lambda|/[d(m-d)]$"),
               ("z_medio", "$z$ respecto de bandera aleatoria")]
    for k, (col, xl) in enumerate(paneles, start=1):
        a = ax[k]
        a.scatter(esp[col], esp["tolerancia"], s=36, color="C0", zorder=3)
        for _, row in esp.iterrows():
            a.annotate(str(row["especie"]), (row[col], row["tolerancia"]),
                       fontsize=6.5, xytext=(3, 3), textcoords="offset points",
                       style="italic")
        r, p, n = spearman_perm(esp[col].values, esp["tolerancia"].values)
        a.set_title("Spearman $\\rho$ = %.2f, $p_{perm}$ = %.3f (n = %d)" % (r, p, n),
                    fontsize=8.5)
        a.set_xlabel(xl)
        a.set_ylabel("Amplitud de tolerancia a NaCl")
    if "auc_media" in esp:
        ax[2].axvline(0.5, color="C3", lw=0.8, ls=":")
    ax[3].axvline(0.0, color="C3", lw=0.8, ls=":")

    for k, a in enumerate(ax):
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        # Letra de panel fuera del eje, esquina superior izquierda
        a.text(-0.16, 1.06, "(%s)" % "ABCD"[k], transform=a.transAxes,
               fontsize=12, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(salida, "figura_paso1." + ext), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Autotest
# ----------------------------------------------------------------------
def autotest(salida="autotest_paso1"):
    rng = np.random.default_rng(42)
    m = 552
    print("1) Fórmula cerrada = Algoritmo 1 ...", end=" ")
    for _ in range(300):
        d = int(rng.integers(1, 200))
        pos = np.sort(rng.choice(np.arange(1, m + 1), d, replace=False))
        _, c1 = algoritmo1(pos, m)
        c2 = codim_cerrada(pos, m)
        assert c1 == c2, (c1, c2)
    print("OK (300 casos)")

    print("2) Nulo analítico vs Monte Carlo ...", end=" ")
    for d in (7, 20, 100, 165):
        sims = np.array([codim_cerrada(rng.choice(np.arange(1, m + 1), d, replace=False), m)
                         for _ in range(20000)])
        mu, var = d * (m - d) / 2.0, d * (m - d) * (m + 1) / 12.0
        assert abs(sims.mean() - mu) / math.sqrt(var) < 0.05
        assert abs(sims.var() / var - 1) < 0.05
    print("OK")

    print("3) Escenario confundido: codim refleja solo d, sin señal real")
    especies = ["sp%02d" % i for i in range(11)]
    d_esp = np.array([7, 7, 9, 9, 14, 21, 94, 105, 132, 165, 60])
    tol = d_esp + rng.normal(0, 25, d_esp.size)       # tolerancia ~ d
    filas = []
    for s, d in zip(especies, d_esp):
        for t in range(150):
            dd = max(1, d + int(rng.integers(-2, 3)))
            pos = rng.choice(np.arange(1, m + 1), dd, replace=False)  # bandera sin info
            filas.append(dict(especie=s, codim=codim_cerrada(pos, m), d=dd))
    df = pd.DataFrame(filas)
    tdf = pd.DataFrame(dict(especie=especies, tolerancia=tol))
    _, corr = analizar(df, tdf, m, os.path.join(salida, "confundido"), n_mc=50000)
    print(corr.to_string(index=False))
    r_crudo = corr.loc[corr.prueba == "tolerancia ~ codim cruda", "rho"].iloc[0]
    r_auc = corr.loc[corr.prueba.str.contains("AUC"), "rho"].iloc[0]
    p_parc = corr.loc[corr.prueba.str.contains("parcial"), "p_perm"].iloc[0]
    assert r_crudo > 0.6 and abs(r_auc) < 0.6 and p_parc > 0.05
    print("   -> la codim cruda 'encuentra' asociación; AUC y parcial no. OK\n")

    print("4) Escenario con señal real: posición en la bandera depende de la tolerancia")
    filas = []
    tol2 = rng.uniform(0, 1, d_esp.size)
    for s, d, tv in zip(especies, d_esp, tol2):
        w = np.exp(-(1.5 * tv) * np.arange(1, m + 1) / m * 3)   # más tolerancia -> antes
        w = w / w.sum()
        for t in range(150):
            pos = rng.choice(np.arange(1, m + 1), d, replace=False, p=w)
            filas.append(dict(especie=s, codim=codim_cerrada(pos, m), d=d))
    df = pd.DataFrame(filas)
    tdf = pd.DataFrame(dict(especie=especies, tolerancia=tol2))
    _, corr = analizar(df, tdf, m, os.path.join(salida, "senal"), n_mc=50000)
    print(corr.to_string(index=False))
    r_auc = corr.loc[corr.prueba.str.contains("AUC"), "rho"].iloc[0]
    p_parc = corr.loc[corr.prueba.str.contains("parcial"), "p_perm"].iloc[0]
    assert r_auc > 0.6 and p_parc < 0.05
    print("   -> AUC y parcial recuperan la señal real. OK")
    print("\nAutotest completo. Figuras en", salida)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados")
    ap.add_argument("--tolerancia")
    ap.add_argument("--m", type=int, default=552)
    ap.add_argument("--salida", default="paso1")
    ap.add_argument("--escenario", default=None,
                    help="filtrar un escenario (p. ej. SS19); por defecto usa todos")
    ap.add_argument("--col-especie", default="especie")
    ap.add_argument("--col-codim", default="codim")
    ap.add_argument("--col-d", default="d")
    ap.add_argument("--col-lambda", default=None)
    ap.add_argument("--col-escenario", default="escenario")
    ap.add_argument("--n-mc", type=int, default=200000)
    ap.add_argument("--autotest", action="store_true")
    a = ap.parse_args()

    if a.autotest:
        autotest()
        return
    if not (a.resultados and a.tolerancia):
        ap.error("se requieren --resultados y --tolerancia (o --autotest)")

    df = cargar_resultados(a.resultados, a.col_especie, a.col_codim, a.col_d,
                           a.col_lambda, a.col_escenario)
    tol = pd.read_csv(a.tolerancia)
    esp, corr = analizar(df, tol, a.m, a.salida, a.escenario, a.n_mc)

    pd.set_option("display.width", 140)
    print("\nResumen por especie:\n", esp.round(4).to_string(index=False))
    print("\nCorrelaciones:\n", corr.round(4).to_string(index=False))
    print("\nLectura rápida:")
    print("  * Si 'AUC', 'z' y 'AUC | d' son ~0 y no significativos, H1a no sobrevive.")
    print("  * Si se mantienen con p < 0.05, la señal no es un artefacto de d.")
    print("Archivos en", os.path.abspath(a.salida))


if __name__ == "__main__":
    main()
