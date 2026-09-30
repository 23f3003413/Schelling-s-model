# Schelling-s-model
Schelling's Segregation Model extended with Dissimilarity Index, heterogeneous thresholds, and boundary analysis — ICA 2026

# Schelling Segregation Model — Implementation & Extended Analysis

> *"Even mild individual preference is sufficient to produce macro-level segregation."*
> — Thomas Schelling, 1971

A Python implementation of Schelling's Segregation Model, extended with
quantitative measurement, heterogeneous agent behaviour, and boundary condition
analysis. Built during a research internship under **Prof. Sukanta Das** at
**IIEST Shibpur**, as part of **ICA 2026 (Indian Cellular Automata Summer School)**.

---

## What this is

Schelling's 1971 model shows that residential segregation can emerge purely from
mild individual tolerance preferences — without any explicit discrimination.
For 50 years, most implementations remained qualitative: you could *see* clusters
form but couldn't *measure* them.

This project extends the standard model with:

- **Dissimilarity Index (DI)** — the standard census metric, used globally by
  the US Census Bureau and UN Habitat, added as a quantitative output variable
- **Heterogeneous thresholds** — different agents get different personal tolerance
  levels, testing Zhang (2011) and Granovetter (1978)
- **Toroidal boundary** — wraps grid edges to eliminate boundary bias
- **Multi-grid experiments** — 20×20, 50×50, 100×100 with fixed patch size

---

## Key findings

### 1 — Clusters form and are now measurable

Even at T = 0.3 (agents need only 30% same-type neighbours), strong clusters
emerge within 25 steps. DI rises from **~0.16 → ~0.44**.

![Four-panel snapshots](results/02_four_panel_snapshots.png)

---

### 2 — Unhappy agents live at boundaries

The happiness map reveals that orange (unhappy) agents sit almost entirely at
the *boundary* between clusters. Interior agents never move — clusters
reinforce themselves automatically.

![Happiness map](results/03_happiness_map.png)

---

### 3 — A city can be 100% happy AND segregated

Tracking DI and happiness together exposes a key tension: happiness hits 100%
by step 10, but DI keeps rising until step 40. The city reaches equilibrium
in a *segregated* state where everyone is satisfied.

![DI and happiness dual chart](results/04_di_happiness_dual.png)

---

### 4 — MAUP independently discovered and fixed

When patch size was scaled proportionally to grid size, DI *decreased* with
larger grids — the wrong direction. This is the **Modifiable Areal Unit
Problem** (Openshaw, 1984), a documented issue in urban measurement.

**Fix:** patch size fixed at 4×4 across all grid sizes. DI values now
directly comparable.

| Grid | Normal DI | Toroidal DI |
|------|-----------|-------------|
| 20×20 | 0.499 | 0.433 |
| 50×50 | 0.464 | 0.478 |
| 100×100 | 0.465 | 0.457 |

![Grid size comparison](results/06_grid_size_comparison.png)

The toroidal DI is consistently lower at small grids (boundary effect is real)
but converges at 100×100, confirming the effect diminishes at scale.

---

### 5 — Intolerant minority drives city-wide segregation

Splitting agents into tolerant (T = 0.2) and intolerant (T = 0.7), and varying
the intolerant fraction from 0% to 100% reveals a monotonic X → DI
relationship. Zhang (2011) proved this effect exists — this maps it
quantitatively.

![X* curve](results/10_x_star_curve.png)

```
0%  intolerant → DI = 0.31
20% intolerant → DI = 0.46  (+48%)
50% intolerant → DI = 0.66
100% intolerant → DI = 0.89
```

No sharp tipping point observed on finite grids — consistent with finite-size
rounding predicted by Gauvin et al. (2009).

---

### 6 — Granovetter (1978) confirmed in Schelling ABM

Two populations with **identical mean tolerance (0.4)** but different
distribution shapes:

| Population | Setup | DI at convergence |
|---|---|---|
| Uniform | everyone T = 0.4 | 0.5749 ± 0.032 |
| Bimodal | 50% T=0.1 + 50% T=0.7 | 0.6075 ± 0.021 |

Bimodal produces higher segregation despite the same average tolerance —
confirming Granovetter's 1978 prediction inside the Schelling ABM using DI
as the measurement variable.

---

## Installation

```bash
git clone https://github.com/23f3003413/schelling-segregation.git
cd schelling-segregation
pip install numpy matplotlib
```

No other dependencies. Python 3.8+ required.

---

## How to run

```bash
python schelling.py
```

All parameters are in the **PARAMETERS block at the top of the file** —
change anything there without touching the logic:

```python
# ===== PARAMETERS — change these freely =====
GRID_SIZE    = 20     # grid dimensions
EMPTY_FRAC   = 0.2   # fraction of empty cells
A_FRAC       = 0.4   # fraction of Group A
B_FRAC       = 0.4   # fraction of Group B
THRESHOLD    = 0.3   # happiness threshold (Schelling's T)
NUM_STEPS    = 100   # simulation steps
```

The script runs all tasks and experiments sequentially and opens output
windows in this order:

```
Window 1  → Initial grid
Window 2  → Happiness map
Window 3  → Before / after (one step)
Window 4  → 4-panel snapshots with DI
Window 5  → DI + Happiness dual chart
Window 6  → Patch heatmap (step 0 and step 100)
Window 7  → Experiment A: grid sizes + toroidal comparison
Window 8  → Experiment B: threshold grid visualisation
Window 9  → Experiment B: before/after heterogeneous simulation
Window 10 → Experiment B: X* curve (0% to 100% intolerant)
Window 11 → Experiment C: Uniform vs Bimodal bar chart
```

---

## Code structure

```
schelling.py
│
├── PARAMETERS block          ← all tunable values
│
├── Functions
│   ├── get_neighbors()       ← Moore neighbourhood (8-cell)
│   ├── is_happy()            ← happiness check per agent
│   ├── run_one_step()        ← move all unhappy agents (uniform T)
│   ├── get_happiness_percent()
│   ├── get_dissimilarity_index()   ← DI formula, patch-based
│   ├── get_neighbors_toroidal()    ← wrap-around boundary
│   ├── is_happy_toroidal()
│   ├── run_one_step_toroidal()
│   ├── get_happiness_percent_toroidal()
│   ├── create_threshold_grid()     ← heterogeneous thresholds
│   ├── is_happy_hetero()           ← per-agent threshold check
│   ├── run_one_step_hetero()       ← threshold travels with agent
│   └── create_bimodal_threshold_grid()
│
├── Tasks 1–7                 ← basic model + DI + patch heatmap
├── Experiment A              ← grid sizes + toroidal + MAUP fix
├── Experiment B              ← heterogeneous thresholds + X* curve
└── Experiment C              ← Granovetter bimodal vs uniform
```

---

## References

| Paper | What we used it for |
|---|---|
| Schelling, T.C. (1971). *Journal of Mathematical Sociology* 1(2). | Original model — grid setup, Moore neighbourhood, threshold rule |
| Ubarevičienė et al. (2024). *Cities*, 145, 104711. | Identified measurement gap — motivated DI implementation |
| Gauvin, Vannimenus & Nadal (2009). *European Physical Journal B* 70(2). | Toroidal boundary, finite-size rounding of phase transition |
| Zhang, J. (2011). *J. Economic Behavior & Organization* 80(1). | Intolerant minority causes segregation — Experiment B |
| Granovetter, M. (1978). *American Journal of Sociology* 83(6). | Distribution shape matters more than mean — Experiment C |
| Fossett, M. (2006). *J. Mathematical Sociology* 30(3–4). | Heterogeneous models match real cities better |

---

## Author

**Ritik Joshi**
IIT Madras (BS DS & ML) + SBCET (B.Tech CSAI)
Research internship under Prof. Sukanta Das, IIEST Shibpur
ICA 2026 — Indian Cellular Automata Summer School

---

*Research internship: completed. Research journey: still ongoing.*
