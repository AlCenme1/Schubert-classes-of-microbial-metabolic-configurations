"""Interface map restricted to tex reactions (sensitivity analysis)."""
import pickle, pipeline_schubert as ps
from config_v2 import cargar_comunidad
m, _ = cargar_comunidad(medio=False, acoplar=False)
mt = ps.construir_mapa(m, interfaz="tex")
pickle.dump(mt, open("mapa_tex.pkl", "wb"))
