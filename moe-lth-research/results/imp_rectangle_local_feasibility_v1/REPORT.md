# Exact local rectangle feasibility at dose 0.075

GPU training remained paused. The frozen one-million-proposal sampler was replayed for all 12 degree controls. Every matrix with no generated rectangle was then exhaustively searched under the first-rectangle local rules, with a declared bound of 100 million ordered high-corner pairs per matrix.

## Result

| Seed | Order | Control | Uncovered | Witness missed | Provably none | Unresolved |
|---:|:---:|:---|---:|---:|---:|---:|
| 7 | A | degree1 | 106 | 2 | 104 | 0 |
| 7 | A | degree2 | 104 | 1 | 103 | 0 |
| 7 | B | degree1 | 110 | 2 | 108 | 0 |
| 7 | B | degree2 | 104 | 3 | 101 | 0 |
| 17 | A | degree1 | 104 | 4 | 100 | 0 |
| 17 | A | degree2 | 106 | 0 | 106 | 0 |
| 17 | B | degree1 | 101 | 1 | 100 | 0 |
| 17 | B | degree2 | 110 | 2 | 108 | 0 |
| 29 | A | degree1 | 104 | 0 | 104 | 0 |
| 29 | A | degree2 | 109 | 0 | 109 | 0 |
| 29 | B | degree1 | 102 | 3 | 99 | 0 |
| 29 | B | degree2 | 112 | 1 | 111 | 0 |
| **Total** |  |  | **1272** | **19** | **1253** | **0** |

A witness proves only that the matrix could accept a first rectangle; it does not prove 95% coverage, sufficient dose, or any global frozen gate. Exhaustive failure proves absence only under the current local eligibility, stratum, activity, magnitude, energy, and per-axis energy rules.

## Missed witnesses

| Seed | Order | Control | Tensor | a | b | cross-r | cross-s | Pairs tested |
|---:|:---:|:---|:---|---:|---:|---:|---:|---:|
| 7 | A | degree1 | `blocks.0.moe.experts.19.fc2.weight` | 137094 | 97851 | 136763 | 98182 | 36759784 |
| 7 | A | degree1 | `blocks.1.moe.experts.10.fc2.weight` | 127318 | 129137 | 127089 | 129366 | 10629468 |
| 7 | A | degree2 | `blocks.1.moe.experts.28.fc2.weight` | 121180 | 196543 | 121791 | 195932 | 20251056 |
| 7 | B | degree1 | `blocks.0.moe.experts.3.fc2.weight` | 122683 | 257855 | 122687 | 257851 | 8365265 |
| 7 | B | degree1 | `blocks.0.moe.experts.5.fc2.weight` | 82567 | 46043 | 82907 | 45703 | 23025453 |
| 7 | B | degree2 | `blocks.0.moe.experts.10.fc2.weight` | 125243 | 231077 | 125605 | 230715 | 16750615 |
| 7 | B | degree2 | `blocks.1.moe.experts.15.fc2.weight` | 96291 | 154314 | 96970 | 153635 | 567079 |
| 7 | B | degree2 | `blocks.2.moe.experts.8.fc2.weight` | 119038 | 139559 | 119079 | 139518 | 5903617 |
| 17 | A | degree1 | `blocks.0.moe.experts.27.fc2.weight` | 132063 | 260743 | 131719 | 261087 | 16983254 |
| 17 | A | degree1 | `blocks.1.moe.experts.14.fc2.weight` | 138436 | 192967 | 138695 | 192708 | 32538828 |
| 17 | A | degree1 | `blocks.1.moe.experts.20.fc2.weight` | 147071 | 246818 | 146466 | 247423 | 11418666 |
| 17 | A | degree1 | `blocks.1.moe.experts.5.fc2.weight` | 142381 | 208050 | 142514 | 207917 | 11729753 |
| 17 | B | degree1 | `blocks.2.moe.experts.23.fc2.weight` | 197491 | 249344 | 197120 | 249715 | 6383694 |
| 17 | B | degree2 | `blocks.0.moe.experts.23.fc2.weight` | 153632 | 185181 | 154461 | 184352 | 16050032 |
| 17 | B | degree2 | `blocks.1.moe.experts.27.fc2.weight` | 14671 | 5718 | 14934 | 5455 | 19947801 |
| 29 | B | degree1 | `blocks.0.moe.experts.12.fc2.weight` | 189109 | 150183 | 189095 | 150197 | 7005086 |
| 29 | B | degree1 | `blocks.0.moe.experts.17.fc2.weight` | 18563 | 147076 | 19076 | 146563 | 35900395 |
| 29 | B | degree1 | `blocks.1.moe.experts.28.fc2.weight` | 90222 | 208421 | 90661 | 207982 | 1560136 |
| 29 | B | degree2 | `blocks.0.moe.experts.30.fc2.weight` | 252021 | 185282 | 252866 | 184437 | 38316380 |

## Decision

Do not launch GPU training. Adding the 19 missed first rectangles could raise coverage by at most 19 matrix hits spread across the controls, while each control currently has 101–112 uncovered matrices and requires at least 244/256 covered matrices for the frozen 95% gate. The present intervention remains infeasible. A different search algorithm would not repair the dominant local infeasibility under these rules; a separate design is required.
