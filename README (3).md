# Schubert classes of microbial metabolic configurations

Code, sampled states and results for the manuscript

> Centeno-Mejía, A. A. *Schubert classes of microbial metabolic configurations: an exact rank-statistic representation, its null model, and what it reveals in a hypersaline community model.* Submitted to the *Journal of Theoretical Biology*.

Each taxon in a genome-scale community model is encoded as a Schubert class. The class is built from the taxon's active exchange coordinates relative to a reference flag on the exchange space. Its codimension |λ| equals a Mann–Whitney U statistic, so the representation comes with an exact null model. The repository applies this to a curated 17-taxon hypersaline community grown on M9 + glycerol. It tests whether the normalized codimension ν tracks NaCl tolerance (it does not), and reads the cross-feeding the model predicts.

## Main results

| Analysis | Result |
|---|---|
| ν vs NaCl tolerance (n = 9 taxa) | Spearman ρ = 0.29, p = 0.44 |
| Raw codimension vs active dimension *d* | ρ = 0.98 (driven by *d*, not by structure) |
| Departure of ν from random flags | Reproduced in full (103 %) by a flag built from usage frequency alone |
| Scenario effect (SS19 vs SS37) | No larger than in 10 random-restriction control replicates |
| Cross-feeding | *Sal. ruber* grows only with community-derived L-lysine (main donor *Chromohalobacter*). *Sal. ruber* is the sole dihydroxyacetone (DHA) producer, and DHA uptake raises *Hqr. walsbyi* growth. |

## Repository layout

```
├── ejecutar_pipeline_v2.ipynb   # runs the whole analysis, cell by cell (start here)
├── config_v2.py                 # paths and model loading (cargar_comunidad, cargar_mapa)
├── pipeline_schubert.py         # core: exchange map, flags, Schubert classes, ν, z, null model,
│                                #   random-objective FBA sampling, ordinal regression, persistent homology
├── curar_modelo.py, reglas.py, egc.py, inspeccion.py, cargas.py
│                                # curation: reversibility rules, energy-generating cycles, charge/mass balance
├── ensamblar.py, curar_indiv2.py
│                                # (optional) rebuild the 17-taxon model from individual reconstructions
├── lote.py                      # sampling in batches of 40 states per fresh process
├── mapa_tex.py                  # high-confidence ("tex") exchange map for the sensitivity analysis
├── analisis.py                  # H1a, flag sensitivity, ε sweep, ordinal regression, persistent homology
├── variantes.py                 # ν under alternative flags
├── control_run.py               # one random-perturbation control replicate per process
├── intercambio.py               # cross-feeding network across sampled states
├── ecologia.py, ecologia_dep.py # growth alone vs in community, pFBA network, blocking tests
├── sintetico.py                 # synthetic validation of |λ| vs ν
├── figuras_revision.py, figuras_run.py, fig_ecologia.py
│                                # manuscript figures
├── suplementario_v2.py, a_apendice.py, rellenar_v4.py
│                                # tables, appendices, and the LaTeX manuscript
├── tolerancia.csv               # NaCl tolerance range used for each taxon
├── curacion_previa/             # curation report for the earlier 12-taxon community
├── curacion_v2/                 # diagnostics for the inserted models (growth, EGCs)
├── cache/                       # sampled net-exchange states (.npz): 150 per scenario + controls
├── resultados/                  # all result tables (.csv) and meta.json
└── figs/                        # figures (PDF + PNG)
```

## Model

The curated community model, `community17_v2_curated.xml` (SBML, ~83 MB), will be deposited on Zenodo (DOI pending). The earlier version is at <https://doi.org/10.5281/zenodo.21731909>. It is too large to keep uncompressed in the repository. Download it, or unzip `community17_v2_curated.xml.zip`, and place it in the repository root.

The model contains 17 taxa:
- 7 taxa reconstructed with CarveMe;
- *Salinibacter ruber* (iSRU780);
- *Haloquadratum walsbyi* (iHQW-TK; Centeno-Mejía et al., *ISME Communications*, in press);
- 8 taxa retained from the earlier community.

The medium is M9 + glycerol, plus traces of L-asparagine and siroheme (0.1 mmol gDW⁻¹ h⁻¹). Species are coupled to community biomass following Heinken et al. (2013), with c = 400 and u = 0.01.

On first load, `config_v2.cargar_comunidad()` reads the SBML and writes `community_v2.pkl`. Later loads take seconds.

## Installation

Python ≥ 3.10.

```bash
pip install -r requirements.txt
```

The LP solver is HiGHS, called through optlang's `hybrid` interface. Blocking tests that add net-uptake constraints need `lp_method="interior point"`, because simplex can stall on them. `ecologia_dep.py` uses interior point by default.

## Running

Open `ejecutar_pipeline_v2.ipynb` in Jupyter from the repository root and run the cells in order:

| Section | What it does | Approx. time |
|---|---|---|
| 0 | (optional) rebuild the model from the individual reconstructions | 20 min |
| 1 | load the model and check the community optimum (0.780 h⁻¹) | 1 min |
| 2 | random-objective FBA sampling, 150 states per scenario | ~1.5 h |
| 3 | exchange maps and the main analysis | 5 min |
| 4 | random-perturbation controls (10 replicates) | ~3 h |
| 5 | cross-feeding and dependency tests | ~45 min |
| 6 | figures, appendices and manuscript | 2 min |

Every expensive step writes to `cache/` or `resultados/` and resumes from there, so an interrupted run loses at most one batch. The sampled states are already included in `cache/`. You can therefore skip sections 2 and 4 and reproduce every table and figure in a few minutes.

Single scripts can also be run from a cell, for example `%run analisis.py`.

## Key parameters

| Parameter | Value | Where |
|---|---|---|
| Activity threshold ε | 0.02 mmol gDW⁻¹ h⁻¹ (sweep 0.005–0.5) | `analisis.py` |
| States per scenario | 150 (SS19, SS37) | `lote.py` |
| Control replicates | 10 × 30 states per scenario | `control_run.py` |
| Fraction of optimum (pFBA, blocking) | 0.9 | `ecologia*.py` |
| Exchange-space dimension *m* | 584 | `mapa_todas.pkl` |

## Citation

If you use this code, please cite the manuscript above (details will be updated on publication) and the archived model (Zenodo DOI pending).

## License

To be defined.

## Contact

Alex A. Centeno-Mejía, Doctorado en Modelamiento Matemático Aplicado, Universidad Católica del Maule, Talca, Chile.
GitHub: [@AlCenme1](https://github.com/AlCenme1)

Funding: ANID BECAS/DOCTORADO NACIONAL No. 21240906.
