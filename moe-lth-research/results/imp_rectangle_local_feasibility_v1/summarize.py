import csv,json
from collections import Counter,defaultdict
from pathlib import Path

here=Path(__file__).resolve().parent
rows=json.loads((here/"classification.json").read_text())
fields=["seed","order","label","tensor","status","shape","high_pool","eligible_survivors",
        "ordered_pairs_tested","stratum_eligible_seen","seconds","witness"]
with (here/"classification.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        x={k:r.get(k) for k in fields}; x["shape"]=json.dumps(x["shape"]); x["witness"]=json.dumps(x["witness"]) if x["witness"] else ""; w.writerow(x)
groups=defaultdict(Counter)
for r in rows: groups[(r["seed"],r["order"],r["label"])][r["status"]]+=1
lines=["# Exact local rectangle feasibility at dose 0.075","",
       "GPU training remained paused. The frozen one-million-proposal sampler was replayed for all 12 degree controls. Every matrix with no generated rectangle was then exhaustively searched under the first-rectangle local rules, with a declared bound of 100 million ordered high-corner pairs per matrix.","",
       "## Result","", "| Seed | Order | Control | Uncovered | Witness missed | Provably none | Unresolved |","|---:|:---:|:---|---:|---:|---:|---:|"]
for key in sorted(groups):
    c=groups[key]; lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {sum(c.values())} | {c['witness_found']} | {c['provably_none']} | {c['unresolved']} |")
total=Counter(r["status"] for r in rows)
lines += [f"| **Total** |  |  | **{len(rows)}** | **{total['witness_found']}** | **{total['provably_none']}** | **{total['unresolved']}** |","",
          "A witness proves only that the matrix could accept a first rectangle; it does not prove 95% coverage, sufficient dose, or any global frozen gate. Exhaustive failure proves absence only under the current local eligibility, stratum, activity, magnitude, energy, and per-axis energy rules.","",
          "## Missed witnesses","", "| Seed | Order | Control | Tensor | a | b | cross-r | cross-s | Pairs tested |","|---:|:---:|:---|:---|---:|---:|---:|---:|---:|"]
for r in rows:
    if r["status"]!="witness_found": continue
    w=r["witness"]
    lines.append(f"| {r['seed']} | {r['order']} | {r['label']} | `{r['tensor']}` | {w['a']} | {w['b']} | {w['cross_r']} | {w['cross_s']} | {r['ordered_pairs_tested']} |")
lines += ["", "## Decision","", "Do not launch GPU training. Adding the 19 missed first rectangles could raise coverage by at most 19 matrix hits spread across the controls, while each control currently has 101–112 uncovered matrices and requires at least 244/256 covered matrices for the frozen 95% gate. The present intervention remains infeasible. A different search algorithm would not repair the dominant local infeasibility under these rules; a separate design is required."]
(here/"REPORT.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
