# -*- coding: utf-8 -*-
"""Paths and model loading shared by every script of the v2 pipeline.

Put in the working folder either
  community17_v2_curated.xml   (curated SBML, coupling is applied on loading), or
  community_v2.pkl             (written by ensamblar.py, or by cargar_comunidad()
                                the first time it reads the SBML; loads in seconds)
"""
import os
import pickle

DIR = os.path.dirname(os.path.abspath(__file__))
SBML = os.path.join(DIR, "community17_v2_curated.xml")
PKL = os.path.join(DIR, "community_v2.pkl")
MAPA = os.path.join(DIR, "mapa_todas.pkl")
INDIV = os.path.join(DIR, "indiv")          # individual SBML models (CarveMe, iSRU780, iHQW-TK)
COMUNIDAD_PREVIA = os.path.join(DIR, "community12_curated.xml")   # earlier curated community


def cargar_comunidad(solver="hybrid", medio=True, acoplar=True, lp_method=None):
    """Curated 17-taxon community with M9 + glycerol (+ traces) and coupling."""
    import cobra
    import pipeline_schubert as ps
    import curar_modelo as cm
    if os.path.exists(PKL):
        m = pickle.load(open(PKL, "rb"))
    else:
        m = cobra.io.read_sbml_model(SBML)
        pickle.dump(m, open(PKL, "wb"))
    m.solver = solver
    if lp_method:
        m.solver.configuration.lp_method = lp_method
    mapa = cargar_mapa(m)
    if medio:
        ps.aplicar_medio(m, verbose=False)
    if acoplar:
        cm.acoplar(m, mapa=mapa)
    return m, mapa


def cargar_mapa(model=None):
    import pipeline_schubert as ps
    if os.path.exists(MAPA):
        return pickle.load(open(MAPA, "rb"))
    mapa = ps.construir_mapa(model, interfaz="todas")
    pickle.dump(mapa, open(MAPA, "wb"))
    return mapa
