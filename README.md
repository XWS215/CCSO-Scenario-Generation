# CCSO Open-Source Package

Chinese version: [README.md](README.md)

CCSO is a multimodal optimization algorithm inspired by the roles of Chinese chess pieces and the phases of a game. This directory contains the plotting data, algorithm implementations, and test functions used in the paper.

## Directory Structure

```
开源文件CCSO/
├── 01_Original data and graphs/   Plotting data and code for four experiment groups
│   ├── A_data/
│   ├── B_data/
│   ├── C_data/
│   └── D_data/
├── 02_Algorithm code/             CCSO, ablation variant, and baseline algorithms
└── 03_Test function/              8 standard benchmark functions
```

## 01 Plotting Data and Code

Each figure has its own folder, for example `A_data/A_1_figure/`, containing:

- `data.csv`: the plotting data used by that figure;
- `plot.py`: reads only `data.csv` from the same folder and writes `figure.png` and `figure.tif`;
- `figure.png` / `figure.tif`: the rendered figure at 600 dpi.

Run:

```bash
cd "01_Original data and graphs/A_data/A_1_figure"
python plot.py
```

Figures are 9 × 2.7 cm (10:3); A.13 is 18 × 5.4 cm. There are 33 figures in total; A.11 is not included.

## 02 Algorithm Code


| File                         | Content                                      |
| ------------------------------ | ---------------------------------------------- |
| `ccso.py`                    | Core CCSO                                    |
| `ccso_ablation.py`           | Ablation CCSO (`ABLATIONS`, `build_variant`) |
| `pso.py` / `sa.py` / `de.py` | PSO / SA / DE baselines                      |
| `demo_对比实验.py`           | Comparison of CCSO against PSO/SA/DE         |

Usage example:

```python
import numpy as np
from ccso import CCSO

opt = CCSO(10, (-5.12, 5.12), pop_size=50, seed=0, n_solutions=2)
res = opt.optimize(lambda x: float(np.sum(np.asarray(x) ** 2)), max_iter=100)
print(res["best_fitness"])
```

Run the comparison:

```bash
cd "02_Algorithm code"
python demo_对比实验.py
```

Current result: CCSO average rank 1.22, PSO 2.35, SA 3.03, DE 3.40 — CCSO ranks first. See `demo_运行结果.txt` for the full output.

## 03 Test Functions

`rastrigin.py`, `schwefel220.py`, `schaffer_nd.py`, `griewank.py`, `salomon.py`, `michalewicz.py`, `katsuura.py`, `cigar.py`

Each file provides `f(x)`, `LB`, `UB`, `D_MIN`, and `f_opt(dim)`.

```python
import sys
sys.path.insert(0, "03_Test function")
from rastrigin import f, LB, UB, f_opt
print(f([0.0, 0.0]), LB, UB, f_opt(2))
```

`Fig_Funs.png` gives an overview of the 8 benchmark landscapes in 2-D.

## Environment

Python 3.8+; the algorithms require only NumPy; the comparison demo also requires SciPy; the plotting scripts also require pandas and Matplotlib.
