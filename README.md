# VitaNova

Code and analysis for a study of **collective-level Darwinian populations** in a
minimal agent-based model.

![A collective forms, closes into a ring, and divides into two daughter
collectives (seed 7; no fitness function anywhere in the
model)](media/vitanova_division.gif)

*One full life cycle of a collective: dispersal, aggregation, ring formation,
and division into two daughters (shaded). Left: the whole arena, with the
camera window marked. Right: the tracked collective. Rendered by
`analysis/anim_division.py`.*

VitaNova is a boids-derived simulation in which agents carry evolvable neural
controllers, gain energy only when they have near neighbours, and reproduce
asexually when they have accumulated enough of it. There is **no fitness
function anywhere in the model**, no term that refers to a group, and no rule
that evaluates one. Collectives nevertheless form, acquire an internal ring-like
structure, grow, and divide, and the divisions compound into lineages.

The system was introduced in Le et al. (2025), arXiv:2409.13254, which described
that behaviour qualitatively. This repository is the updated continuation of the
original 2024 implementation that accompanied that paper,
[namlehai90/Evolving-Neuronal-Flocks](https://github.com/namlehai90/Evolving-Neuronal-Flocks):
the model is unchanged, ported to a vectorised core and instrumented, and this
repository adds the measurements that the earlier paper did not make.

---

## Main findings

All numbers below are produced by scripts in this repository from stored runs.

**A Darwinian population exists at the collective level.** Across sixteen
independent runs of 9,000 steps: 504 collectives, 198 of which divided, 403 with
a recorded parent, and a deepest lineage of **19 collective generations**
(verified chain of 20 collectives, steps 1,400 to 8,725).

**Collective size is transmitted.** Parent–offspring regression on size at
division, over 185 pairs: raw slope +0.72 (*r* = +0.73); +0.64 after centring
within replicate and within temporal phase; permutation null −0.00 ± 0.09,
*p* < 0.0005.

**Reproductive success varies sharply.** 306 of 504 collectives leave no
offspring; the rest leave two or three. Opportunity for selection *I* = 1.57.

**Division is reproduction, not fragmentation.** Median daughter/parent size
ratio, each measured at its own division: 0.96 (IQR 0.71–1.14).

**Two conditions are necessary.** (i) The energetic subsidy for grouping: with it
off, median peak annularity falls from 0.92 to 0.00 and no collective exceeds 21
members. This is an arithmetic consequence of the energy constants — without the
subsidy an agent accumulates at most 150 energy in a 1,000-step lifetime against
a reproduction threshold of 200, so nothing can reproduce at all.
(ii) Unbounded growth: adding a crowding cost that stabilises the ring structure
collapses the deepest lineage from **19 generations to 1**, and three of ten
populations then go extinct.

**Evolution shapes the collectives but does not make them reproduce.** Genotype
explains 82.9% of the variance in collective structure against 1.7% for initial
conditions (12 genotypes × 8 initial conditions, *F* = 37.1, *p* < 0.0001);
selection raises median peak annularity from 0.59 (clonal) to 0.92 (evolving,
*p* = 0.0002); mutation adds nothing (0.84, *p* = 0.62 against the full model);
and the **rate of collective division is unchanged** (28/40 clonal vs 30/40
evolving, Fisher *p* = 0.80).

**Which variants produce the ring.** Same eight seeds, same 9,000 steps, same
statistic, scored on the largest collective with a geometric guard
(`code/variant_rings.py`):

| variant | median peak *A* | established | ring in |
| --- | --- | --- | --- |
| NoCaste (FFN, castes deleted) | 0.96 | 7/8 | 7/8 |
| Crowded (size ceiling) | 0.95 | 7/8 | 7/8 |
| Fast (FFN, the baseline) | 0.93 | 7/8 | 7/8 |
| Locked (role fixed at birth) | 0.79 | 5/8 | 5/8 |
| Linear (no hidden layer) | 0.00 | 3/8 | 3/8 |
| No ecology (subsidy off) | 0.08 | 1/8 | 2/8 |

"Established" means a collective of at least 20 members formed at some point.
Note the linear controller: conditional on establishing it reaches 0.70, but it
establishes in only three of eight runs, so unconditionally it does **not**
reproduce the phenomenon. An earlier note in this project claimed it did; that
figure was computed on established runs only and is corrected here.

---

## Layout

```
code/
  fast/vitafast.py        vectorised port of the original model (the baseline)
  reproduce/shape.py      annularity and radial-profile statistics
  reproduce/              headless drivers, plotting, the standalone HTML viewer
  eusocial/nocaste.py     castes deleted (4 outputs, 150 weights)
  eusocial/crowded.py     density-dependent maintenance cost (the size ceiling)
  eusocial/locked.py      agent role fixed at birth
  eusocial/*.py           recognition, gestalt-badge and adhesion variants
  linear/linear_vita.py   controller reduced to tanh(Wx), 66 coefficients
  deb/                    dynamic-energy-budget variant of the metabolism
  Neat_vitanova/          NEAT replacing the fixed-topology network
  evo_cooperative_flock.py   the original pygame implementation

analysis/                 figure and statistics scripts for the manuscript
results/                  stored summary statistics as JSON (small)
paper/                    manuscript source, bibliography and figures
```

Stored `.npz` simulation output is **not** in the repository; it is
150–200 MB and is regenerated by the run scripts.

---

## Installation

Python 3.9 or later. Tested with Python 3.9.18, NumPy 2.0.2, SciPy 1.13.1,
Matplotlib 3.9.4.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

`pygame` is needed only for `code/evo_cooperative_flock.py`, the original
interactive implementation. Every result in the manuscript comes from the
headless port, which does not need it.

Set `OMP_NUM_THREADS=1` before running anything that uses multiple processes.
The run scripts already do this internally, but BLAS threading inside worker
processes otherwise oversubscribes the machine badly.

---

## Reproducing the results

Run from the repository root with `code/fast`, `code/reproduce` and the relevant
variant directory on `PYTHONPATH`. On Windows:

```powershell
$env:PYTHONPATH="code\fast;code\reproduce;code\eusocial;code\linear"
$env:OMP_NUM_THREADS="1"
```

On macOS or Linux:

```bash
export PYTHONPATH="code/fast:code/reproduce:code/eusocial:code/linear"
export OMP_NUM_THREADS=1
```

| Result | Script | Approximate cost |
| --- | --- | --- |
| Baseline batch, 16 runs × 9,000 steps | `code/reproduce/` drivers | ~15 min on 12 cores |
| Collective lineages and depth | `analysis/fig2_lineage.py` | seconds, from stored `collectives` tables |
| Parent–offspring heredity | `analysis/fig3_heredity.py` | seconds |
| Evolution ablation, 3 arms × 40 seeds | `code/linear/evolution_ffn.py` | ~20 min |
| Genotype × initial condition crossing | `code/linear/genotype_variance_ffn.py` | ~30 min |
| Ecology ablation | `code/eusocial/no_ecology.py` | ~10 min |
| Size-ceiling effect on lineage depth | `code/eusocial/long_multi.py` | ~30 min |
| Which variants produce rings | `code/variant_rings.py` | ~10 min |

Figure scripts contain absolute paths from the machine they were written on and
need those edited before use. This is the main rough edge in the repository.

---

## Two cautions for anyone building on this

**Measure lineage depth, not division count.** A hundred divisions that never
compound describe a completely different system from a hundred that reach
nineteen generations of descent, and only the second is a population that
selection could act on.

**A structure statistic needs a size-matched null.** An earlier composite score
used here assigned *exactly* 0.00 to the real core-plus-annulus configuration,
because a ring with an occupied centre is not hollow. Two other statistics
failed in this project: Couzin order parameters computed about the centroid of a
fragmented population reported a milling signature that images showed to be four
independent clumps, and a fusion-to-fission ratio fell towards 1.0 under every
intervention simply because destabilising a collective raises both counts.

---

## Status

Work in progress. The manuscript in `paper/` is a draft and its claims have
changed substantially as measurements accumulated; the numbers in this README
are current and the draft may lag them.

Results depend on platform floating-point behaviour. Seeds reproduce exactly on
the same machine and library versions, and approximately across platforms.

Nam H. Le — `namlehai90@gmail.com`
