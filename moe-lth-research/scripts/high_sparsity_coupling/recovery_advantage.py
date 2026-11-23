"""Checkpoint-only recovery advantage analysis from verified saved artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/high_sparsity_coupling_v1"
EXT = SOURCE / "functional_geometry_extension_v1"
AUDIT = EXT / "audit_v1"
OUT = SOURCE / "recovery_advantage_analysis_v1"
SEEDS = (7, 17, 29, 43, 61)
SPARSITIES = (0.80, 0.85)
SNAPSHOT_STEPS = (0, 25, 100, 250, 500, 1000, 1500, 2000, 2500, 2525, 2600, 3000, 3500, 4000, 4500, 5000)
EXPECTED_MATCHES = {(7, 0.80), (7, 0.85), (43, 0.80), (61, 0.80)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def f(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def i(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def condition_id(seed: int, sparsity: float | None) -> str:
    return f"e32_seed{seed}_dense" if sparsity is None else f"e32_seed{seed}_final_s{sparsity:.2f}"


def attempt_dir(condition: str) -> Path:
    parent = SOURCE / "reruns" / condition
    attempts = sorted(path for path in parent.iterdir() if path.is_dir() and path.name.startswith("attempt_"))
    if len(attempts) != 1:
        raise AssertionError(f"Expected one completed rerun attempt for {condition}, found {len(attempts)}")
    return attempts[0]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def step_update_records(condition: str) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    """Read compact optimizer outcomes and training CE from the per-attempt diagnostics."""
    attempt = attempt_dir(condition)
    updates: dict[int, dict] = {}
    train: dict[int, dict] = {}
    segments = {}
    for segment, offset in (("segment_0", 0), ("segment_2500", 2500)):
        base = attempt / segment
        record = json.loads((base / "record.json").read_text())
        segments[offset] = record
        rows = {}
        patterns = {
            "step": re.compile(r'"step":\s*(\d+)'),
            "train_ce": re.compile(r'"train_cross_entropy":\s*([^,}]+)'),
            "aux_loss": re.compile(r'"auxiliary_routing_loss":\s*([^,}]+)'),
            "applied": re.compile(r'"optimizer_step_applied":\s*(true|false)', re.IGNORECASE),
        }
        with (base / "gradient_stats/gradient_stats.jsonl").open() as handle:
            for line in handle:
                matches = {key: pattern.search(line) for key, pattern in patterns.items()}
                if any(match is None for match in matches.values()):
                    continue
                local_step = int(matches["step"].group(1))
                global_step = offset + local_step
                applied = matches["applied"].group(1).lower() == "true"
                row = {
                    "global_step": global_step,
                    "local_step": local_step,
                    "segment": segment,
                    "applied": applied,
                    "amp_skip": not applied,
                    "train_ce": f(matches["train_ce"].group(1)),
                    "aux_loss": f(matches["aux_loss"].group(1)),
                }
                updates[global_step] = row
                train[global_step] = row
                rows[local_step] = row
        expected = set(range(1, 2501))
        if set(rows) != expected:
            raise AssertionError(f"Missing/duplicate optimizer-step records for {condition}/{segment}")
        if sum(row["applied"] for row in rows.values()) != i(record["optimizer_steps_applied"]):
            raise AssertionError(f"Applied-update total does not reconcile for {condition}/{segment}")
        if sum(row["amp_skip"] for row in rows.values()) != i(record["amp_overflow_step_count"]):
            raise AssertionError(f"AMP skip total does not reconcile for {condition}/{segment}")
    return updates, train, segments


def svg_text(x, y, text, *, size=12, anchor="middle", fill="#222", weight="normal"):
    escaped = (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escaped}</text>'


COLORS = {7: "#0072B2", 17: "#D55E00", 29: "#009E73", 43: "#CC79A7", 61: "#E69F00"}


def svg_axes(width, height, left, top, right, bottom, x_ticks, y_ticks, x_map, y_map, x_label, y_label, title):
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, size=18, weight="bold"),
        f'<path d="M {left} {top} V {bottom} H {right}" stroke="#222" fill="none" stroke-width="1.2"/>',
    ]
    for tick in x_ticks:
        x = x_map(tick)
        pieces.extend([
            f'<path d="M {x:.2f} {bottom} v 5" stroke="#222"/>',
            svg_text(x, bottom + 22, tick, size=11),
            f'<path d="M {x:.2f} {top} V {bottom}" stroke="#ddd" stroke-dasharray="2 4"/>',
        ])
    for value, label in y_ticks:
        y = y_map(value)
        pieces.extend([
            f'<path d="M {left-5} {y:.2f} h 5" stroke="#222"/>',
            svg_text(left - 10, y + 4, label, size=11, anchor="end"),
            f'<path d="M {left} {y:.2f} H {right}" stroke="#eee"/>',
        ])
    pieces.extend([
        svg_text((left + right) / 2, bottom + 48, x_label, size=13),
        f'<text x="20" y="{(top+bottom)/2:.2f}" transform="rotate(-90 20 {(top+bottom)/2:.2f})" text-anchor="middle" font-size="13" fill="#222">{y_label}</text>',
    ])
    return pieces


def svg_polyline(points, color, *, dash="", width=1.7, opacity=0.95):
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dashed}/>'


def draw_figures(performance, token_summaries, internal, crossover):
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    by_pair = defaultdict(list)
    for row in performance:
        by_pair[(i(row["seed"]), f(row["sparsity"]))].append(row)
    for values in by_pair.values():
        values.sort(key=lambda row: i(row["step"]))

    # A. Full-validation absolute CE. The logarithmic display retains the large step-0 values.
    w, h, left, top, right, bottom = 1260, 790, 90, 65, 1195, 595
    x_map = lambda x: left + (right - left) * x / 5000
    y_map = lambda y: bottom - (bottom - top) * (math.log10(max(1.0, y)) - 0.0) / (math.log10(200.0))
    y_ticks = [(v, str(v)) for v in (1, 2, 5, 10, 20, 50, 100, 200)]
    pieces = svg_axes(w, h, left, top, right, bottom, (0, 1000, 2000, 3000, 4000, 5000),
                      y_ticks, x_map, y_map, "Attempted recovery steps", "Full-validation token-mean CE (log scale)",
                      "A. Absolute full-validation CE (rerun trajectory)")
    pieces.append(f'<path d="M {x_map(2500):.2f} {top} V {bottom}" stroke="#333" stroke-dasharray="5 5"/>')
    pieces.append(svg_text(x_map(2500) + 7, top + 15, "optimizer reset", size=10, anchor="start"))
    for seed in SEEDS:
        dense_id = condition_id(seed, None)
        dense_rows = [r for r in performance if i(r["seed"]) == seed and r["condition"] == dense_id]
        dense_rows = sorted({i(r["step"]): r for r in dense_rows}.values(), key=lambda r: i(r["step"]))
        pts = [(x_map(i(r["step"])), y_map(f(r["dense_validation_ce"]))) for r in dense_rows]
        pieces.append(svg_polyline(pts, COLORS[seed], dash="2 4", width=1.8, opacity=.75))
    for (seed, sparsity), rows in sorted(by_pair.items()):
        vals = [r for r in rows if r["role"] == "sparse"]
        dash = "" if sparsity == .80 else "8 4"
        pieces.append(svg_polyline([(x_map(i(r["step"])), y_map(f(r["sparse_validation_ce"]))) for r in vals],
                                   COLORS[seed], dash=dash, width=2.0))
    legend_y = 642
    entries = []
    for seed in SEEDS:
        entries.append((f"seed {seed} dense", COLORS[seed], "2 4"))
        entries.append((f"seed {seed} sparse .80", COLORS[seed], ""))
        entries.append((f"seed {seed} sparse .85", COLORS[seed], "8 4"))
    for idx, (label, color, dash) in enumerate(entries):
        col = idx % 3
        row = idx // 3
        x = 120 + col * 370
        y = legend_y + row * 25
        pieces.append(svg_polyline([(x, y - 4), (x + 28, y - 4)], color, dash=dash, width=2.2))
        pieces.append(svg_text(x + 36, y, label, size=11, anchor="start"))
    pieces.append('</svg>')
    (figure_dir / "A_absolute_validation_ce.svg").write_text("\n".join(pieces))

    # B. Continuous sparse-minus-dense gap; symmetric-log display is only a visual transform.
    w, h, left, top, right, bottom = 1260, 790, 90, 65, 1195, 595
    transform = lambda x: math.asinh(x / .01)
    vals = [f(r["gap"]) for r in performance]
    lo, hi = min(transform(v) for v in vals), max(transform(v) for v in vals)
    pad = .04 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    y_map = lambda value: bottom - (bottom - top) * (transform(value) - lo) / (hi - lo)
    ticks = [-5, -1, -.1, -.01, 0, .01, .1, 1]
    pieces = svg_axes(w, h, left, top, right, bottom, (0, 1000, 2000, 3000, 4000, 5000),
                      [(v, f"{v:g}") for v in ticks if lo <= transform(v) <= hi],
                      x_map, y_map, "Attempted recovery steps", "Sparse CE - matched dense CE (symlog; threshold remains 0)",
                      "B. All ten continuous paired validation gaps")
    pieces.append(f'<path d="M {left} {y_map(0):.2f} H {right}" stroke="#111" stroke-width="1.5"/>')
    pieces.append(f'<path d="M {x_map(2500):.2f} {top} V {bottom}" stroke="#333" stroke-dasharray="5 5"/>')
    pieces.append(svg_text(x_map(2500) + 7, top + 15, "optimizer reset", size=10, anchor="start"))
    cross_by_key = {(i(r["seed"]), f(r["sparsity"])): r for r in crossover}
    for (seed, sparsity), rows in sorted(by_pair.items()):
        vals = [r for r in rows if r["role"] == "sparse"]
        dash = "" if sparsity == .80 else "8 4"
        pieces.append(svg_polyline([(x_map(i(r["step"])), y_map(f(r["gap"]))) for r in vals],
                                   COLORS[seed], dash=dash, width=1.9))
        onset = cross_by_key[(seed, sparsity)].get("persistent_nonpositive_start_step")
        if onset not in ("", None):
            obs = next(r for r in vals if i(r["step"]) == i(onset))
            x, y = x_map(i(onset)), y_map(f(obs["gap"]))
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{COLORS[seed]}" stroke="white"/>')
    for idx, (label, color, dash) in enumerate(entries):
        col = idx % 3
        row = idx // 3
        x = 120 + col * 370
        y = legend_y + row * 25
        pieces.append(svg_polyline([(x, y - 4), (x + 28, y - 4)], color, dash=dash, width=2.2))
        pieces.append(svg_text(x + 36, y, label, size=11, anchor="start"))
    pieces.append('</svg>')
    (figure_dir / "B_validation_gap.svg").write_text("\n".join(pieces))

    # C. Endpoint probe distributions and concentration. Each condition is shown separately.
    endpoints = [r for r in token_summaries if i(r["step"]) == 5000]
    endpoints.sort(key=lambda r: (i(r["seed"]), f(r["sparsity"])))
    w, h = 1450, 820
    # Left panel: quantile intervals and median/mean on a shared delta scale.
    left, top, mid, right, bottom = 90, 80, 720, 1380, 710
    delta_min = min(f(r["q05"]) for r in endpoints)
    delta_max = max(f(r["q95"]) for r in endpoints)
    span = max(delta_max - delta_min, .01)
    delta_min -= .06 * span
    delta_max += .06 * span
    y_ticks = []
    for q in np.linspace(delta_min, delta_max, 7):
        y_ticks.append((float(q), f"{q:.3f}"))
    x_map = lambda x: left + (mid-left) * x / 10
    y_map = lambda value: bottom - (bottom-top) * (value-delta_min)/(delta_max-delta_min)
    pieces = svg_axes(w, h, left, top, mid, bottom, (0, 2, 4, 6, 8, 10), y_ticks, x_map, y_map,
                      "Seed/sparsity condition index", "Per-token CE difference (sparse - dense)",
                      "C. Fixed 2,048-position validation probe at step 5,000")
    for n, row in enumerate(endpoints):
        x = x_map(n + .5)
        color = COLORS[i(row["seed"])]
        pieces.append(f'<path d="M {x:.2f} {y_map(f(row["q05"])):.2f} V {y_map(f(row["q95"])):.2f}" stroke="{color}" stroke-width="2"/>')
        pieces.append(f'<path d="M {x-9:.2f} {y_map(f(row["q25"])):.2f} H {x+9:.2f} M {x-9:.2f} {y_map(f(row["q75"])):.2f} H {x+9:.2f}" stroke="{color}" stroke-width="6"/>')
        pieces.append(f'<path d="M {x-10:.2f} {y_map(f(row["median"])):.2f} H {x+10:.2f}" stroke="#111" stroke-width="2"/>')
        pieces.append(f'<circle cx="{x:.2f}" cy="{y_map(f(row["mean_delta"])):.2f}" r="4" fill="{color}" stroke="white"/>')
        pieces.append(svg_text(x, bottom+20, f'{i(row["seed"])}-{f(row["sparsity"]):.2f}', size=10))
    pieces.append(f'<path d="M {left} {y_map(0):.2f} H {mid}" stroke="#111" stroke-width="1"/>')
    pieces.append(svg_text(left+15, top+16, "whisker: 5th–95th; thick bar: 25th–75th; black: median; dot: mean", size=10, anchor="start"))
    # Right panel: upper-tail shares of positive and negative token contributions.
    rleft, rright = 800, 1380
    y0, y1 = 590, 145
    x_map2 = lambda x: rleft + (rright-rleft)*x/10
    y_map2 = lambda value: y0 - (y0-y1)*value
    pieces.extend([
        f'<path d="M {rleft} {y1} V {y0} H {rright}" stroke="#222" fill="none"/>',
        svg_text((rleft+rright)/2, 100, "Top-k token share of same-sign total contribution", size=15, weight="bold"),
        svg_text((rleft+rright)/2, 765, "Condition (seed-sparsity)", size=12),
    ])
    for ytick in (0, .2, .4, .6, .8, 1):
        y = y_map2(ytick)
        pieces.append(f'<path d="M {rleft} {y:.2f} H {rright}" stroke="#eee"/>')
        pieces.append(svg_text(rleft-8, y+4, f"{ytick:.1f}", size=10, anchor="end"))
    offsets = {1: -9, 5: 0, 10: 9}
    for n, row in enumerate(endpoints):
        center = x_map2(n+.5)
        color = COLORS[i(row["seed"])]
        pos = [f(row[f"positive_top_{k}pct_share"]) for k in (1,5,10)]
        neg = [f(row[f"negative_top_{k}pct_share"]) for k in (1,5,10)]
        for k, vals_ in ((1,pos),(2,neg)):
            cx=center + (-12 if k==1 else 12)
            for share, pct in zip(vals_, (1,5,10)):
                pieces.append(f'<circle cx="{cx+offsets[pct]:.2f}" cy="{y_map2(share):.2f}" r="3.4" fill="{color}" opacity="{0.45+0.05*pct}"/>')
            pieces.append(svg_polyline([(cx+offsets[pct],y_map2(share)) for share,pct in zip(vals_,(1,5,10))],color,width=1.2,opacity=.7))
        pieces.append(svg_text(center, y0+20, f'{i(row["seed"])}-{f(row["sparsity"]):.2f}', size=9))
    pieces.extend([
        svg_text(rleft+12, y1+18, "Left group: positive (worsening) contributions; right group: negative (improving) contributions.", size=10, anchor="start"),
        svg_text(rleft+12, y1+34, "Dot positions indicate top 1%, 5%, 10% of contributing positions.", size=10, anchor="start"),
    ])
    pieces.append('</svg>')
    (figure_dir / "C_probe_gain_loss.svg").write_text("\n".join(pieces))

    # D. Layerwise identical-input output errors and coupling aligned to persistent intervals.
    w, h = 1450, 1110
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
              '<rect width="100%" height="100%" fill="white"/>',
              svg_text(w/2, 28, "D. Existing internal trajectories at observed performance resolution", size=18, weight="bold"),
              svg_text(w/2, 50, "Fixed-source outputs use identical hidden inputs; markers denote persistent nonpositive validation tails.", size=11)]
    introws = [r for r in internal if r["metric_family"] == "expert_output" and r["measure"] == "relative_output_error" and r["probe"] == "fixed_source" and r["model_role"] == "sparse"]
    crossmap = {(i(r["seed"]),f(r["sparsity"])):r for r in crossover}
    panel_positions = [(70,90),(760,90),(70,600),(760,600)]
    panel_w,panel_h=610,385
    for layer,(px,py) in enumerate(panel_positions):
        pl,pt,pr,pb=px+55,py+48,px+panel_w-15,py+panel_h-60
        subset=[r for r in introws if i(r["layer"])==layer]
        values=[f(r["value"]) for r in subset]
        ymin=max(0,min(values));ymax=max(values)
        if ymax<=ymin:ymax=ymin+1
        ymin=max(0,ymin-.05*(ymax-ymin));ymax+=.08*(ymax-ymin)
        xm=lambda step:pl+(pr-pl)*step/5000
        ym=lambda val:pb-(pb-pt)*(val-ymin)/(ymax-ymin)
        pieces.extend([svg_text((pl+pr)/2,py+20,f"Layer {layer}: fixed-source expert-output relative error",size=13,weight="bold"),
                       f'<path d="M {pl} {pt} V {pb} H {pr}" stroke="#222" fill="none"/>'])
        for tick in (0,1000,2000,3000,4000,5000):
            x=xm(tick);pieces.extend([f'<path d="M {x:.2f} {pb} v 4" stroke="#222"/>',svg_text(x,pb+18,tick,size=9)])
        for tick in np.linspace(ymin,ymax,4):
            y=ym(float(tick));pieces.extend([f'<path d="M {pl} {y:.2f} H {pr}" stroke="#eee"/>',svg_text(pl-7,y+3,f"{tick:.2f}",size=9,anchor="end")])
        pieces.append(f'<path d="M {xm(2500):.2f} {pt} V {pb}" stroke="#333" stroke-dasharray="4 4"/>')
        for seed in SEEDS:
            for sp in SPARSITIES:
                vals_=[r for r in subset if i(r["seed"])==seed and abs(f(r["sparsity"])-sp)<1e-8]
                vals_.sort(key=lambda r:i(r["step"]))
                if not vals_:continue
                dash="" if sp==.80 else "7 4"
                pieces.append(svg_polyline([(xm(i(r["step"])),ym(f(r["value"]))) for r in vals_],COLORS[seed],dash=dash,width=1.25,opacity=.72))
                onset=crossmap[(seed,sp)].get("persistent_nonpositive_start_step")
                if onset not in ("",None):
                    point=next((r for r in vals_ if i(r["step"])==i(onset)),None)
                    if point:
                        pieces.append(f'<circle cx="{xm(i(onset)):.2f}" cy="{ym(f(point["value"])):.2f}" r="3" fill="{COLORS[seed]}" stroke="white"/>')
    # Compact seed/sparsity legend.
    ly=1035
    for n,seed in enumerate(SEEDS):
        x=140+n*220
        pieces.append(svg_polyline([(x,ly),(x+27,ly)],COLORS[seed],width=2))
        pieces.append(svg_text(x+34,ly+4,f"seed {seed}",size=10,anchor="start"))
    pieces.append(svg_text(w/2,1075,"solid = .80; dashed = .85; vertical grey = reset at 2,500; circles = persistent-tail start",size=11))
    pieces.append('</svg>')
    (figure_dir / "D_layerwise_internal.svg").write_text("\n".join(pieces))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT/"analysis_config.json",{
        "version":1,"scope_recorded_before_outcome_processing":True,"exploratory_post_hoc":True,
        "analysis":"checkpoint-only; no training, recovery, mask search, optimizer updates, or causal intervention",
        "primary_track":"rerun","seeds":list(SEEDS),"sparsities":list(SPARSITIES),
        "unique_dense_controls":5,"independent_unit":"source seed",
        "full_validation_steps":"saved rerun validation scalar every 50 attempts plus actual saved-checkpoint evaluations at off-grid steps 25 and 2525",
        "full_validation_precision":"fp32 validation; TF32 enabled",
        "saved_checkpoint_steps":list(SNAPSHOT_STEPS),
        "token_probe_scope":"fixed 2,048 registered validation batch-0 positions, same positions per source seed across both sparsities and all 16 saved checkpoint steps",
        "token_probe_precision":"protected fp32 diagnostic forward; TF32 disabled",
        "token_probe_output":"per-token arrays with checkpoint/cache provenance; descriptive only",
        "historical_track":"excluded from primary curve due to different TF32 setting and duplicate model weights",
        "test_split":"not inspected/evaluated; untouched test provenance not established",
    })
    cohort = json.loads((SOURCE / "cohort.json").read_text())
    function_rows = read_csv(SOURCE / "report/model_function_trajectories.csv")
    function = {
        (r["condition"], i(r["step"])): r
        for r in function_rows if r["track"] == "rerun"
    }
    if len(function) != 15 * len(SNAPSHOT_STEPS):
        raise AssertionError("Rerun checkpoint-evaluation table does not cover 15 models x 16 snapshots")

    provenance = read_csv(EXT / "checkpoint_and_cache_provenance.csv")
    checkpoint_file_state_before={}
    for row in provenance:
        path=Path(row["path"])
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint provenance path is missing: {path}")
        stat=path.stat()
        checkpoint_file_state_before[str(path)]=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
    checkpoint_rows = {
        (r["condition"], i(r["step"])): r
        for r in provenance if r["state"] == "E_snapshot"
    }
    if len(checkpoint_rows) != 15 * len(SNAPSHOT_STEPS):
        raise AssertionError("Checkpoint provenance table does not cover all 15 x 16 snapshots")
    audit_provenance = read_csv(EXT / "checkpoint_and_cache_provenance.csv")
    condition_info = {row["id"]: row for row in cohort["conditions"]}
    test_metadata_fields=sorted({
        key for job in cohort["conditions"] for key in job["config"]["data"]
        if "test" in key.lower()
    })
    if test_metadata_fields:
        raise AssertionError(f"Frozen data configuration contains test metadata; inspect provenance status: {test_metadata_fields}")

    function_by_condition_step = {
        key: row for key, row in function.items()
    }
    metric_eval = {}
    optimizer_data = {}
    validation_logs = {}
    training_data = {}
    segment_data = {}
    for seed in SEEDS:
        for sparsity in [None, *SPARSITIES]:
            cond = condition_id(seed, sparsity)
            updates, train, segments = step_update_records(cond)
            optimizer_data[cond] = updates
            training_data[cond] = train
            segment_data[cond] = segments
            attempt = attempt_dir(cond)
            values = {}
            for segment, offset in (("segment_0", 0), ("segment_2500", 2500)):
                for row in read_jsonl(attempt / segment / "metrics.jsonl"):
                    step = offset + i(row["step"])
                    value = f(row["loss"])
                    if step in values and abs(values[step] - value) > 1e-12:
                        raise AssertionError(f"Different duplicated step-2500 validation values for {cond}")
                    values[step] = value
            if len(values) != 101 or set(values) != set(range(0, 5001, 50)):
                raise AssertionError(f"Unexpected full-validation log coverage for {cond}")
            validation_logs[cond] = values
            # The 16 cached checkpoint evaluations share exactly the rerun validation settings.
            for step in SNAPSHOT_STEPS:
                cached = f(function_by_condition_step[(cond, step)]["validation_ce"])
                if step in values and cached != values[step]:
                    raise AssertionError(f"Checkpoint CE/log mismatch for {cond} at {step}: {cached} != {values[step]}")
            combined = values.copy()
            combined.update({
                step: f(function_by_condition_step[(cond, step)]["validation_ce"])
                for step in SNAPSHOT_STEPS
            })
            metric_eval[cond] = combined

    # Build unique 103-step paired observations: every 50-step validation log plus snapshots at 25 and 2525.
    observation_steps = sorted(set(range(0, 5001, 50)) | {25, 2525})
    performance = []
    checkpoint_crosschecks = []
    pair_lookup = {}
    for seed in SEEDS:
        dense = condition_id(seed, None)
        for sparsity in SPARSITIES:
            sparse = condition_id(seed, sparsity)
            for step in observation_steps:
                sparse_ce = metric_eval[sparse][step]
                dense_ce = metric_eval[dense][step]
                if step in SNAPSHOT_STEPS:
                    s_cache = function_by_condition_step[(sparse, step)]
                    d_cache = function_by_condition_step[(dense, step)]
                    sparse_cache_ce = f(s_cache["validation_ce"])
                    dense_cache_ce = f(d_cache["validation_ce"])
                    cache_error = max(abs(sparse_ce-sparse_cache_ce),abs(dense_ce-dense_cache_ce))
                    checkpoint = checkpoint_rows[(sparse, step)]
                    dense_checkpoint = checkpoint_rows[(dense, step)]
                    checkpoint_available = True
                    s_hash = checkpoint["sha256"]
                    d_hash = dense_checkpoint["sha256"]
                    s_model_hash = checkpoint["model_hash"]
                    d_model_hash = dense_checkpoint["model_hash"]
                    source = "rerun_full_validation_log_and_saved_checkpoint_cache" if step % 50 == 0 else "saved_checkpoint_full_validation_cache"
                else:
                    cache_error = ""
                    checkpoint_available = False
                    s_hash = d_hash = s_model_hash = d_model_hash = ""
                    source = "rerun_full_validation_log_no_saved_model_checkpoint"
                attempted = step
                s_updates = optimizer_data[sparse]
                d_updates = optimizer_data[dense]
                s_applied = sum(row["applied"] for gstep,row in s_updates.items() if gstep <= step)
                d_applied = sum(row["applied"] for gstep,row in d_updates.items() if gstep <= step)
                s_skips = sum(row["amp_skip"] for gstep,row in s_updates.items() if gstep <= step)
                d_skips = sum(row["amp_skip"] for gstep,row in d_updates.items() if gstep <= step)
                if step == 0:
                    segment = "initialization"
                    reset_flag = False
                elif step < 2500:
                    segment = "segment_0"
                    reset_flag = False
                elif step == 2500:
                    segment = "segment_0_end_and_segment_2500_reset_start_same_weights"
                    reset_flag = True
                else:
                    segment = "segment_2500_after_optimizer_scaler_reset"
                    reset_flag = False
                info_s = condition_info[sparse]
                info_d = condition_info[dense]
                s_identity = info_s["extension_identity"]
                d_identity = info_d["extension_identity"]
                if s_identity["validation_batch_hash"] != d_identity["validation_batch_hash"]:
                    raise AssertionError(f"Matched sparse/dense validation stream mismatch for seed {seed}")
                if s_identity["training_batch_hash"] != d_identity["training_batch_hash"]:
                    raise AssertionError(f"Matched sparse/dense training stream mismatch for seed {seed}")
                train_start = max(1, ((step-1)//50)*50+1) if step else None
                train_rows = []
                if step and step % 50 == 0:
                    train_rows = [s_updates[g] for g in range(step-49,step+1) if g in s_updates]
                    # At the segment boundary, the window ends at step 2500 in segment 0.
                elif step and step % 50 != 0:
                    segment_start = 1 if step < 2500 else 2501
                    train_rows = [s_updates[g] for g in range(max(segment_start,step-49),step+1) if g in s_updates]
                train_ce = float(np.mean([r["train_ce"] for r in train_rows])) if train_rows else ""
                train_aux = float(np.mean([r["aux_loss"] for r in train_rows])) if train_rows else ""
                row = {
                    "seed": seed, "sparsity": sparsity, "condition": sparse, "role": "sparse",
                    "matched_dense_condition": dense, "step": step, "attempted_updates_cumulative": attempted,
                    "sparse_applied_updates_cumulative": s_applied, "dense_applied_updates_cumulative": d_applied,
                    "sparse_amp_skips_cumulative": s_skips, "dense_amp_skips_cumulative": d_skips,
                    "sparse_validation_ce": sparse_ce, "dense_validation_ce": dense_ce,
                    "gap": sparse_ce-dense_ce, "endpoint_label": "",
                    "segment_identity": segment, "optimizer_scaler_reset_at_boundary": reset_flag,
                    "checkpoint_available": checkpoint_available, "sparse_checkpoint_sha256": s_hash,
                    "dense_checkpoint_sha256": d_hash, "sparse_model_hash": s_model_hash,
                    "dense_model_hash": d_model_hash, "checkpoint_cache_log_max_abs_diff": cache_error,
                    "evaluation_source": source, "evaluation_precision": "fp32 validation, TF32 enabled",
                    "evaluation_population": "full validation: 32 original complete batches; token-summed CE / token count",
                    "validation_batch_hash": s_identity["validation_batch_hash"],
                    "training_stream_hash": s_identity["training_batch_hash"],
                    "train_ce_recent_attempt_window_mean": train_ce,
                    "auxiliary_routing_loss_recent_attempt_window_mean": train_aux,
                    "training_window_start_global_attempt": min((r["global_step"] for r in train_rows),default=""),
                    "training_window_end_global_attempt": max((r["global_step"] for r in train_rows),default=""),
                    "training_window_attempts": len(train_rows),
                    "historical_track_in_primary_curve": False,
                }
                performance.append(row)
                pair_lookup[(seed,sparsity,step)] = row
                if checkpoint_available:
                    checkpoint_crosschecks.append({
                        "seed":seed,"sparsity":sparsity,"condition":sparse,"step":step,
                        "cache_validation_ce":sparse_cache_ce,"logged_validation_ce":validation_logs[sparse].get(step,""),
                        "absolute_difference":cache_error,"checkpoint_sha256":s_hash,
                    })
    # Exactly one unique dense path per seed is shared across both matched conditions.
    dense_reuse = defaultdict(set)
    for row in performance:
        dense_reuse[i(row["seed"])].add(row["matched_dense_condition"])
    assert all(dense_reuse[s] == {condition_id(s,None)} for s in SEEDS)

    # Calculate strict zero-threshold crossovers and every later sign reversal.
    crossover = []
    decomposition = []
    for seed in SEEDS:
        for sparsity in SPARSITIES:
            rows = [pair_lookup[(seed,sparsity,step)] for step in observation_steps]
            gaps = [f(r["gap"]) for r in rows]
            first_idx = next((idx for idx,gap in enumerate(gaps) if gap <= 0), None)
            reversals = []
            for idx in range(1,len(rows)):
                prev,cur=gaps[idx-1],gaps[idx]
                if (prev <= 0) != (cur <= 0):
                    reversals.append({
                        "from_step":i(rows[idx-1]["step"]),"to_step":i(rows[idx]["step"]),
                        "direction":"advantage_lost" if prev <= 0 < cur else "advantage_gained_or_regained",
                        "gap_before":prev,"gap_after":cur,
                    })
            persistent_idx = next((idx for idx in range(len(gaps)) if all(gap<=0 for gap in gaps[idx:])),None)
            demonstrated = persistent_idx is not None and persistent_idx < len(gaps)-1
            if not demonstrated:
                persistent_start=""
                persistence_steps=[]
            else:
                persistent_start=i(rows[persistent_idx]["step"])
                persistence_steps=[i(row["step"]) for row in rows[persistent_idx:]]
            prior_idx = first_idx-1 if first_idx is not None and first_idx>0 else None
            first_interval = (
                f'{i(rows[prior_idx]["step"])}..{i(rows[first_idx]["step"])}'
                if prior_idx is not None else ""
            )
            last = rows[-1]
            endpoint_match = f(last["gap"]) <= 0
            if endpoint_match != ((seed,sparsity) in EXPECTED_MATCHES):
                raise AssertionError(f"Authoritative endpoint match differs for seed={seed}, s={sparsity}")
            if first_idx == 0:
                first_status = "already_nonpositive_at_initialization"
            elif first_idx is None:
                first_status = "never_nonpositive"
            else:
                first_status = "first_observed_nonpositive_after_positive"
            crossover.append({
                "seed":seed,"sparsity":sparsity,"condition":condition_id(seed,sparsity),
                "initial_gap":gaps[0],"first_observed_nonpositive_step":"" if first_idx is None else i(rows[first_idx]["step"]),
                "first_nonpositive_status":first_status,"interval_containing_first_nonpositive":first_interval,
                "later_reversal_count":len(reversals),"later_reversals_json":json.dumps(reversals,separators=(",",":")),
                "earliest_persistent_nonpositive_step":persistent_start,
                "persistent_tail_demonstrated":demonstrated,"persistence_observation_count":len(persistence_steps),
                "persistence_observed_steps_json":json.dumps(persistence_steps),
                "final_gap":gaps[-1],"endpoint_matches_or_beats_dense":endpoint_match,
                "observed_evaluation_count":len(rows),"observed_steps_json":json.dumps(observation_steps),
                "crossover_time_exactly_known":False,
            })
            for idx in range(1,len(rows)):
                prev,cur=rows[idx-1],rows[idx]
                ds=f(cur["sparse_validation_ce"])-f(prev["sparse_validation_ce"])
                dd=f(cur["dense_validation_ce"])-f(prev["dense_validation_ce"])
                dg=f(cur["gap"])-f(prev["gap"])
                if ds<0 and dd<0:
                    driver="both_CE_fall; sparse_falls_more" if ds<dd else "both_CE_fall; dense_falls_more_or_equal"
                elif ds<0<=dd:
                    driver="sparse_CE_falls_while_dense_is_flat_or_rises"
                elif dd<0<=ds:
                    driver="dense_CE_falls_while_sparse_is_flat_or_rises"
                elif ds>=0 and dd>=0:
                    driver="both_CE_flat_or_rise; relative_increase_sets_gap_change"
                else:
                    driver="mixed_CE_changes; inspect_signed_components"
                decomposition.append({
                    "seed":seed,"sparsity":sparsity,"condition":condition_id(seed,sparsity),
                    "from_step":i(prev["step"]),"to_step":i(cur["step"]),
                    "sparse_ce_change":ds,"dense_ce_change":dd,"gap_change":dg,
                    "identity_residual_gap_change_minus_components":dg-(ds-dd),
                    "gap_narrowed":dg<0,"signed_driver_description":driver,
                    "optimizer_reset_inside_interval":i(prev["step"])<=2500<i(cur["step"]),
                    "checkpoint_at_from_step":prev["checkpoint_available"],
                    "checkpoint_at_to_step":cur["checkpoint_available"],
                })
    expected_set={(i(r["seed"]),f(r["sparsity"])) for r in crossover if r["endpoint_matches_or_beats_dense"]}
    assert expected_set==EXPECTED_MATCHES

    # Precision sensitivity is kept separate from the rerun trajectory.
    settings=read_csv(SOURCE/"report/ce_execution_settings.csv")
    by_condition={r["condition"]:r for r in settings}
    precision_rows=[]
    for seed in SEEDS:
        dense=by_condition[condition_id(seed,None)]
        for sparsity in SPARSITIES:
            sparse=by_condition[condition_id(seed,sparsity)]
            off_sparse=f(sparse["reevaluated_tf32_off"]);off_dense=f(dense["reevaluated_tf32_off"])
            on_sparse=f(sparse["reevaluated_tf32_on"]);on_dense=f(dense["reevaluated_tf32_on"])
            off_gap=off_sparse-off_dense;on_gap=on_sparse-on_dense
            precision_rows.append({
                "seed":seed,"sparsity":sparsity,"sensitivity_step":5000,
                "off_sparse_ce":off_sparse,"off_dense_ce":off_dense,"off_gap":off_gap,
                "on_sparse_ce":on_sparse,"on_dense_ce":on_dense,"on_gap":on_gap,
                "gap_difference_tf32_off_minus_on":off_gap-on_gap,
                "strict_sign_agrees":(off_gap<=0)==(on_gap<=0),
                "purpose":"endpoint numerical sensitivity only; not mixed into trajectory",
            })

    # Pairwise token-level probes, strictly separate from full validation.
    token_summaries=[]
    token_provenance=[]
    token_array_checks=[]
    function_probe={(r["condition"],i(r["step"])):r for r in function.values()}
    for seed in SEEDS:
        dense=condition_id(seed,None)
        expected_ids=None
        for sparsity in SPARSITIES:
            sparse=condition_id(seed,sparsity)
            sparse_losses=[];dense_losses=[];deltas=[];positions=None;tokens=None;targets=None
            sparse_shas=[];dense_shas=[];checkpoint_shas=[]
            data_hash=condition_info[sparse]["extension_identity"]["validation_batch_hash"]
            for step in SNAPSHOT_STEPS:
                spath=SOURCE/"analysis/rerun"/sparse/f"step_{step:04d}"
                dpath=SOURCE/"analysis/rerun"/dense/f"step_{step:04d}"
                scache=json.loads((spath/"completed.json").read_text())
                dcache=json.loads((dpath/"completed.json").read_text())
                if not scache.get("model_preserved") or not dcache.get("model_preserved"):
                    raise AssertionError(f"Invalid cache integrity flag: {sparse}/{step}")
                if scache["condition"]!=sparse or dcache["condition"]!=dense or i(scache["step"])!=step or i(dcache["step"])!=step:
                    raise AssertionError("Token cache condition/step provenance mismatch")
                with np.load(spath/"raw.npz",allow_pickle=False) as sraw, np.load(dpath/"raw.npz",allow_pickle=False) as draw:
                    spos=sraw["positions"];dpos=draw["positions"]
                    stokens=sraw["tokens"];dtokens=draw["tokens"]
                    stargets=sraw["targets"];dtargets=draw["targets"]
                    if not (np.array_equal(spos,dpos) and np.array_equal(stokens,dtokens) and np.array_equal(stargets,dtargets)):
                        raise AssertionError(f"Token-position misalignment at {sparse}/{step}")
                    if positions is None:
                        positions=spos.copy();tokens=stokens.reshape(-1).copy();targets=stargets.reshape(-1).copy()
                        expected_ids=(array_hash(positions),array_hash(tokens),array_hash(targets))
                    else:
                        if (array_hash(spos),array_hash(stokens.reshape(-1)),array_hash(stargets.reshape(-1)))!=expected_ids:
                            raise AssertionError(f"Probe IDs changed across time/condition for seed {seed}")
                    s_loss=sraw["probe_ce"].astype(np.float64)
                    d_loss=sraw["dense_probe_ce"].astype(np.float64)
                    d_cache_loss=draw["probe_ce"].astype(np.float64)
                    delta=sraw["per_token_ce_difference"].astype(np.float64)
                    if not np.allclose(d_loss,d_cache_loss,atol=2e-6,rtol=2e-6):
                        raise AssertionError(f"Sparse cache's matched dense token CE differs from dense raw cache: {sparse}/{step}")
                    if not np.allclose(delta,s_loss-d_loss,atol=2e-7,rtol=2e-6):
                        raise AssertionError(f"Per-token delta does not equal sparse minus dense CE: {sparse}/{step}")
                    probe_delta=float(np.mean(s_loss)-np.mean(d_loss))
                    reconciliation=abs(float(np.mean(delta))-probe_delta)
                    if reconciliation>1e-7:
                        raise AssertionError(f"Per-token-to-probe-aggregate reconciliation failure: {sparse}/{step}")
                    frow_s=function_probe[(sparse,step)]
                    frow_d=function_probe[(dense,step)]
                    cache_recon=max(abs(float(np.mean(s_loss))-f(frow_s["probe_ce"])),
                                    abs(float(np.mean(d_loss))-f(frow_d["probe_ce"])))
                    if cache_recon>2e-5:
                        raise AssertionError(f"Token array vs saved probe CE mismatch: {sparse}/{step} ({cache_recon})")
                    sparse_losses.append(s_loss.astype(np.float32))
                    dense_losses.append(d_loss.astype(np.float32))
                    deltas.append(delta.astype(np.float32))
                    sparse_shas.append(scache["outputs"]["raw.npz"])
                    dense_shas.append(dcache["outputs"]["raw.npz"])
                    checkpoint_shas.append(checkpoint_rows[(sparse,step)]["sha256"])
                    token_provenance.append({
                        "seed":seed,"sparsity":sparsity,"condition":sparse,"step":step,
                        "token_array_path":f"token_arrays/{sparse}.npz",
                        "sparse_raw_npz_sha256_recorded":scache["outputs"]["raw.npz"],
                        "dense_raw_npz_sha256_recorded":dcache["outputs"]["raw.npz"],
                        "sparse_checkpoint_sha256":checkpoint_shas[-1],
                        "dense_checkpoint_sha256":checkpoint_rows[(dense,step)]["sha256"],
                        "validation_batch_hash":data_hash,"position_ids_sha256":expected_ids[0],
                        "token_ids_sha256":expected_ids[1],"target_ids_sha256":expected_ids[2],
                        "precision":"probe arrays from protected fp32 diagnostic forward; TF32 disabled",
                        "population":"2048 registered positions from validation batch 0; model routed on full 128x128 batch",
                    })
                    q=np.quantile(delta,[.05,.25,.5,.75,.95])
                    pos=delta[delta>0];neg=delta[delta<0]
                    count=len(delta)
                    def top_share(values, pct):
                        mass=float(np.abs(values).sum())
                        if mass==0:return 0.0
                        n=max(1,int(math.ceil(count*pct/100)))
                        n=min(n,len(values))
                        return float(np.sort(np.abs(values))[-n:].sum()/mass)
                    pos_share={pct:top_share(pos,pct) for pct in (1,5,10)}
                    neg_share={pct:top_share(neg,pct) for pct in (1,5,10)}
                    summary={
                        "seed":seed,"sparsity":sparsity,"condition":sparse,"step":step,
                        "full_validation_gap_same_step":pair_lookup[(seed,sparsity,step)]["gap"],
                        "probe_positions":count,"mean_delta":float(np.mean(delta)),
                        "mean_delta_from_ce_arrays":probe_delta,
                        "token_probe_reconciliation_abs_error":reconciliation,
                        "probe_ce_cache_reconciliation_max_abs_error":cache_recon,
                        "improved_positions":int(np.sum(delta<0)),"worsened_positions":int(np.sum(delta>0)),
                        "equal_positions":int(np.sum(delta==0)),
                        "fraction_improved":float(np.mean(delta<0)),"fraction_worsened":float(np.mean(delta>0)),
                        "fraction_equal":float(np.mean(delta==0)),
                        "q05":float(q[0]),"q25":float(q[1]),"median":float(q[2]),"q75":float(q[3]),"q95":float(q[4]),
                        "min_delta":float(np.min(delta)),"max_delta":float(np.max(delta)),
                        "positive_delta_sum_total":float(pos.sum()),"negative_delta_sum_total":float(neg.sum()),
                        "positive_contribution_to_mean_gap":float(pos.sum()/count),
                        "negative_contribution_to_mean_gap":float(neg.sum()/count),
                        "net_probe_gap":float(delta.mean()),
                        "probe_minus_full_validation_gap_not_an_aggregate_reconciliation":float(delta.mean()-f(pair_lookup[(seed,sparsity,step)]["gap"])),
                        "positive_top_1pct_share":pos_share[1],"positive_top_5pct_share":pos_share[5],"positive_top_10pct_share":pos_share[10],
                        "negative_top_1pct_share":neg_share[1],"negative_top_5pct_share":neg_share[5],"negative_top_10pct_share":neg_share[10],
                        "token_unit_is_not_replication_unit":True,
                    }
                    token_summaries.append(summary)
                    token_array_checks.append({
                        "seed":seed,"sparsity":sparsity,"step":step,"position_count":count,
                        "position_ids_sha256":expected_ids[0],"token_ids_sha256":expected_ids[1],
                        "target_ids_sha256":expected_ids[2],"token_delta_mean":float(delta.mean()),
                        "probe_ce_difference":probe_delta,"absolute_reconciliation_error":reconciliation,
                        "matched_dense_token_ce_cache_max_abs_error":float(np.max(np.abs(d_loss-d_cache_loss))),
                    })
            arrpath=OUT/"token_arrays"/f"{sparse}.npz"
            arrpath.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(
                arrpath,steps=np.asarray(SNAPSHOT_STEPS,dtype=np.int32),
                positions=positions.astype(np.int64),token_ids=tokens.astype(np.int64),target_ids=targets.astype(np.int64),
                sparse_token_ce=np.stack(sparse_losses),dense_token_ce=np.stack(dense_losses),
                delta_token=np.stack(deltas),sparse_raw_sha256=np.asarray(sparse_shas),
                dense_raw_sha256=np.asarray(dense_shas),sparse_checkpoint_sha256=np.asarray(checkpoint_shas),
                probe_precision=np.asarray("fp32; TF32 disabled"),
            )

    # Normalized long-form alignment of already-computed internal measurements.
    internal=[]
    def add_internal(seed,sparsity,step,family,measure,value,*,probe="",reference="",layer="",valid_n="",aggregation="",source="",model_role="sparse",measured_condition=None):
        join=pair_lookup.get((int(seed),float(sparsity),int(step)))
        if join is None:return
        internal.append({
            "seed":int(seed),"sparsity":float(sparsity),"condition":condition_id(int(seed),float(sparsity)),
            "step":int(step),"matched_dense_condition":condition_id(int(seed),None),
            "model_role":model_role,"measured_model_condition":measured_condition or condition_id(int(seed),float(sparsity)),
            "full_validation_gap_at_step":join["gap"],"metric_family":family,"measure":measure,
            "value":value,"probe":probe,"reference":reference,"layer":layer,"valid_n":valid_n,
            "aggregation":aggregation,"source":source,
        })

    # Fixed-source/all-expert per-unit coupling, matched on the exact source seed and checkpoint step.
    coupling_rows=read_csv(SOURCE/"report/coupling_trajectories.csv")
    cindex={}
    for row in coupling_rows:
        if row["track"]!="rerun" or row["mode"]!="fixed_source" or row["selection"]!="all_expert" or row["model_role"] not in ("sparse","dense"):
            continue
        key=(row["model_role"],i(row["seed"]),None if row["model_role"]=="dense" else f(row["sparsity"]),i(row["step"]))
        cindex[key]=row
    for seed in SEEDS:
        dense=condition_id(seed,None)
        for sp in SPARSITIES:
            for step in SNAPSHOT_STEPS:
                sr=cindex[("sparse",seed,sp,step)]
                dr=cindex[("dense",seed,None,step)]
                sv=f(sr["pearson"]);dv=f(dr["pearson"])
                add_internal(seed,sp,step,"coupling_fixed_source_all_expert","sparse_pearson",sv,
                             probe="fixed_source/all_expert raw router score vs post-GELU activation",reference="same seed dense model",valid_n=sr["n_valid_units"],source="report/coupling_trajectories.csv")
                add_internal(seed,sp,step,"coupling_fixed_source_all_expert","dense_pearson",dv,
                             probe="fixed_source/all_expert raw router score vs post-GELU activation",reference="same seed dense model",valid_n=dr["n_valid_units"],source="report/coupling_trajectories.csv",
                             model_role="dense_control",measured_condition=dense)
                add_internal(seed,sp,step,"coupling_fixed_source_all_expert","sparse_minus_dense",sv-dv,
                             probe="fixed_source/all_expert raw router score vs post-GELU activation",reference="paired sparse - dense",valid_n=min(i(sr["n_valid_units"]),i(dr["n_valid_units"])),source="report/coupling_trajectories.csv")

    # Native routing summaries from cached full-batch-capacity probe records.
    for seed in SEEDS:
        for sp in SPARSITIES:
            paired_sparse=condition_id(seed,sp)
            for model_role,measured_condition in (("sparse",paired_sparse),("dense_control",condition_id(seed,None))):
                for step in SNAPSHOT_STEPS:
                    route_dir=SOURCE/"analysis/rerun"/measured_condition/f"step_{step:04d}"
                    routes=read_csv(route_dir/"routes.csv")
                    with np.load(route_dir/"raw.npz",allow_pickle=False) as raw:
                        for r in routes:
                            if r["mode"]!="native":continue
                            proposed=np.asarray(json.loads(r["proposed_load"]),dtype=np.float64)
                            accepted=np.asarray(json.loads(r["accepted_load"]),dtype=np.float64)
                            ptotal=float(proposed.sum());atotal=float(accepted.sum())
                            if proposed.size!=32 or accepted.size!=32 or ptotal<=0:
                                raise AssertionError(f"Invalid native route count vector at {measured_condition}/{step}/L{r['layer']}")
                            cv=float(accepted.std()/accepted.mean()) if accepted.mean()>0 else float("nan")
                            layer=r["layer"]
                            scores=raw[f"native_L{int(layer)}_scores"].astype(np.float64)
                            logits=scores-scores.max(axis=1,keepdims=True)
                            probs=np.exp(logits)
                            probs/=probs.sum(axis=1,keepdims=True)
                            top1_probability=float(probs.max(axis=1).mean())
                            for measure,value in (
                                ("top1_softmax_confidence_mean",top1_probability),
                                ("top1_top2_probability_margin_mean",f(r["margin_mean"])),
                                ("full_softmax_entropy_mean",f(r["entropy_mean"])),
                                ("proposed_probe_assignments",ptotal),
                                ("accepted_probe_assignments",atotal),
                                ("probe_capacity_acceptance_fraction",atotal/ptotal),
                                ("accepted_probe_load_cv",cv),
                                ("assignment_agreement_with_initial",f(r["initial_assignment_agreement"])),
                                ("assignment_agreement_with_source",f(r["source_assignment_agreement"])),
                                ("selected_probe_positions",i(r["selected_probe_positions"])),
                                ("capacity_full_batch_tokens",i(r["capacity_full_batch_tokens"])),
                            ):
                                add_internal(seed,sp,step,"native_routing",measure,value,probe="native hidden inputs; full original batch retained for capacity",
                                             reference="native model routing",layer=layer,valid_n=r["selected_probe_positions"],
                                             aggregation="layer-level; assignment/load counts on selected probe positions",
                                             source=f"analysis/rerun/{measured_condition}/step_{step:04d}/routes.csv",
                                             model_role=model_role,measured_condition=measured_condition)

    rep=read_csv(SOURCE/"report/representation_trajectories.csv")
    for r in rep:
        if r["track"]!="rerun" or r["condition"].endswith("_dense") or r["reference"]!="same_step_dense":continue
        seed=i(r["seed"]);sp=f(r["sparsity"]);step=i(r["step"])
        for measure in ("cka","relative_l2","mean_cosine","effective_rank","top8_energy","norm_mean"):
            if r[measure]!="":
                add_internal(seed,sp,step,"representation_same_step_dense",measure,f(r[measure]),
                             probe="identical registered probe positions; native layer inputs differ by model",
                             reference="same-step dense representation (moving reference)",layer=r["boundary"],
                             valid_n=r["n_identical_probe_positions"],aggregation="2048 probe positions",
                             source="report/representation_trajectories.csv")

    expert=read_csv(SOURCE/"report/expert_function_trajectories.csv")
    for r in expert:
        if r["track"]!="rerun" or r["condition"].endswith("_dense"):continue
        seed=i(r["seed"]);sp=f(r["sparsity"]);step=i(r["step"])
        for measure in ("relative_output_error","output_cosine"):
            add_internal(seed,sp,step,"expert_output",measure,f(r[measure]),probe=r["mode"],
                         reference="same-step dense expert outputs; identical inputs only for fixed_source",
                         layer=r["layer"],valid_n=r["n_valid_outputs"],
                         aggregation="per-layer expert outputs; see source table",
                         source="report/expert_function_trajectories.csv")

    # CKA of complete expert outputs averaged over the stored valid layer-expert population.
    geom=read_csv(AUDIT/"geometry_interval_changes.csv")
    for r in geom:
        if r["family"]!="activation_geometry" or r["input_mode"]!="fixed_source" or r["quantity"]!="expert_output" or r["metric"]!="cka":
            continue
        series=json.loads(r["all_steps_json"])
        for step,value in series.items():
            add_internal(r["seed"],r["sparsity"],step,"expert_output_geometry","centered_linear_cka",value,
                         probe="fixed_source",reference="same-step dense (moving)",valid_n=r["rank_valid_fraction_json"],
                         aggregation="valid expert-unit means; see definition",source="audit_v1/geometry_interval_changes.csv")

    # Absolute fixed-gate distortion numerator and reference energy, only at its measured steps.
    for r in geom:
        if r["family"]!="fixed_gate_distortion" or r["input_mode"]!="fixed_source" or r["gate_reference"]!="same_step_dense":
            continue
        for step in (0,2500,5000):
            for field,measure in (
                (f"D_full_step_{step}","D_full_squared_output_difference"),
                (f"ref_energy_full_step_{step}","reference_output_squared_energy"),
                (f"mean_layer_ratio_full_step_{step}","mean_layer_relative_squared_ratio"),
                (f"ratio_aggregate_full_step_{step}","aggregate_energy_ratio"),
            ):
                if r[field]!="":
                    add_internal(r["seed"],r["sparsity"],step,"fixed_gate_distortion",measure,f(r[field]),
                                 probe="fixed_source; full 2048 probe; dropped positions contribute zero",
                                 reference="same-step dense top-1 gates (moving)",valid_n=r["probe_positions"],
                                 aggregation=r["aggregation"],source="audit_v1/geometry_interval_changes.csv")

    grad=read_csv(EXT/"gradient_geometry/per_seed_summary.csv")
    for r in grad:
        seed=i(r["seed"]);sp=f(r["sparsity"]);step=i(r["step"])
        for measure in ("energy_retention","keep_fraction","cosine_projected_dense_vs_sparse",
                        "cosine_dense_vs_sparse_unrestricted","random_mask_energy_retention",
                        "energy_excess_over_random","energy_excess_over_keep_fraction"):
            if r[measure]!="":
                add_internal(seed,sp,step,"gradient_geometry",measure,f(r[measure]),
                             probe=f"training minibatches; {r['kind']} gradient",reference="same-step dense/task mask",
                             layer=r["matrix"],valid_n="16 minibatches",
                             aggregation="per layer/matrix; source seed replication unit",
                             source="gradient_geometry/per_seed_summary.csv")

    # Per-parameter update measurements exist only at selected individual update events.
    update_measurements=0
    for seed in SEEDS:
        for sp in SPARSITIES:
            for model_role,measured_condition in (("sparse",condition_id(seed,sp)),("dense_control",condition_id(seed,None))):
                attempt=attempt_dir(measured_condition)
                for path in sorted((attempt/"updates").glob("step_*.csv")):
                    step=int(path.stem.split("_")[1])
                    if step not in SNAPSHOT_STEPS:continue
                    rows=read_csv(path)
                    for group in sorted({r["group"] for r in rows}):
                        subset=[r for r in rows if r["group"]==group]
                        norm=math.sqrt(sum(f(r["update_norm"],0.0)**2 for r in subset))
                        cos=[f(r["gradient_update_cosine"]) for r in subset if r["gradient_update_cosine"] not in ("","None")]
                        add_internal(seed,sp,step,"optimizer_update_event","group_update_l2_norm",norm,
                                     probe="single attempted training update",reference="model weights before update",
                                     layer=group,valid_n=len(subset),aggregation="root-sum-square parameter update norms",
                                     source=str(path.relative_to(SOURCE)),model_role=model_role,measured_condition=measured_condition)
                        if cos:
                            add_internal(seed,sp,step,"optimizer_update_event","mean_parameter_gradient_update_cosine",float(np.mean(cos)),
                                         probe="single attempted training update",reference="pre-update gradient",
                                         layer=group,valid_n=len(cos),aggregation="unweighted mean over parameters",
                                         source=str(path.relative_to(SOURCE)),model_role=model_role,measured_condition=measured_condition)
                        update_measurements+=1
    write_csv(OUT/"aligned_internal_trajectories.csv",internal)

    # Create endpoint-independent token arrays and all primary tables.
    write_csv(OUT/"performance_trajectory.csv",performance)
    write_csv(OUT/"crossover_summary.csv",crossover)
    write_csv(OUT/"interval_gap_decomposition.csv",decomposition)
    write_csv(OUT/"token_gain_loss_summary.csv",token_summaries)
    write_csv(OUT/"token_array_provenance.csv",token_provenance)
    write_csv(OUT/"precision_sensitivity.csv",precision_rows)
    write_csv(OUT/"checkpoint_cache_crosschecks.csv",checkpoint_crosschecks)

    # Inventory uses explicit coverage units and distinguishes evaluation logs from saved weights.
    inventory=[
        {"measurement":"rerun full-validation scalar logs","source":"reruns/*/segment_{0,2500}/metrics.jsonl",
         "coverage":"15 unique models x 101 unique steps (0..5000 by 50)","precision":"fp32 validation; TF32 enabled",
         "population":"32 original complete validation batches; token-summed mean CE","status":"used; step 2500 duplicate reconciled"},
        {"measurement":"saved checkpoint full-validation scalar cache","source":"analysis/rerun/*/step_*/completed.json",
         "coverage":"15 unique models x 16 snapshots","precision":"fp32 validation; TF32 enabled",
         "population":"full validation; same saved evaluation semantics","status":"used for off-grid snapshot steps; exact matched-step checks"},
        {"measurement":"saved model checkpoints","source":"reruns/*/snapshots/*.pt",
         "coverage":"15 unique models x 16 saved snapshots","precision":"model-only snapshots; no new forwards",
         "population":"actual saved states only; no checkpoint asserted at other logged steps","status":"verified through provenance manifest"},
        {"measurement":"per-token paired validation probe arrays","source":"analysis/rerun/*/step_*/raw.npz",
         "coverage":"10 sparse conditions x 16 snapshots x 2048 aligned positions","precision":"protected fp32 diagnostic forwards; TF32 disabled",
         "population":"2048 registered positions from validation batch 0; full 128x128 structure used for capacity","status":"used; token results are probe-only"},
        {"measurement":"optimizer application and AMP skips","source":"reruns/*/segment_*/gradient_stats/gradient_stats.jsonl",
         "coverage":"75,000 attempted updates across 15 models","precision":"native training/GradScaler log",
         "population":"training stream; step-level outcomes","status":"all per-segment applied/skip totals reconciled"},
        {"measurement":"training task CE and auxiliary routing loss","source":"same per-attempt gradient_stats logs",
         "coverage":"75,000 training batches","precision":"recorded per-batch values; AMP skips retained",
         "population":"training stream, not validation population","status":"shown only as recent-window means, auxiliary separate"},
        {"measurement":"existing internal trajectory metrics","source":"report/, functional_geometry_extension_v1/, analysis/rerun/",
         "coverage":"step-specific joins only; gradients/distortion at 0/2500/5000",
         "precision":"metric-specific; see aligned table and source definitions",
         "population":"mixed fixed-source/native probes and training minibatches","status":"joined only at exact common seed/sparsity/step"},
        {"measurement":"historical track","source":"report/historical_* and prior extension reports",
         "coverage":"sparse endpoints / incomplete trajectory","precision":"TF32 disabled historical vs enabled rerun",
         "population":"same model weights at shared states","status":"excluded from primary crossover curve"},
        {"measurement":"independent test split","source":"frozen config data provenance",
         "coverage":"no explicit test path or untouched-split provenance","precision":"not evaluated",
         "population":"none","status":"unavailable from provenance alone; not inspected"},
    ]
    write_csv(OUT/"measurement_inventory.csv",inventory)

    exclusions=[
        {"item":"historical CE observations","status":"excluded from primary trajectory","reason":"historical and rerun full-validation settings differ in TF32; historical sparse/dense shared weights are duplicates"},
        {"item":"historical missing checkpoints","status":"excluded","reason":"model states missing at most steps; no interpolation and no surrogate checkpoint construction"},
        {"item":"rerun 50-step validation observations","status":"included as scalar observations only","reason":"no model checkpoint exists at each such evaluation; checkpoint_available is false except actual saved snapshots"},
        {"item":"token-level validation results","status":"probe-only","reason":"only 2,048 fixed positions from validation batch 0; not full-validation token trajectories"},
        {"item":"domain/token subgroup results","status":"not computed","reason":"no verified domain labels per token-position in the registered probe metadata"},
        {"item":"independent test evaluation","status":"not performed","reason":"no demonstrably untouched test split established by frozen cohort provenance; no test data inspected"},
        {"item":"causal optimizer reset effect","status":"unresolved","reason":"all sparse and dense runs reset at 2,500; no no-reset control"},
        {"item":"intermediate gradient trajectories","status":"unavailable","reason":"gradient geometry exists only at 0/2,500/5,000; no interpolation"},
        {"item":"pooled time correlation/winner classifier","status":"not performed","reason":"explicitly out of scope; seed is replication unit and sparsities within seed are dependent"},
        {"item":"new model evaluation or tensor mutation","status":"none","reason":"all scalar/prediction/internal measurements reused from saved artifacts; analysis reads no source checkpoint tensors"},
    ]
    write_csv(OUT/"coverage_and_exclusions.csv",exclusions)

    # Brief analysis configuration records scope before interpreting outcome.
    config={
        "version":1,"scope_recorded_before_outcome_processing":True,"exploratory_post_hoc":True,"analysis":"checkpoint-only; no training, recovery, mask search, optimizer updates, or causal intervention",
        "cohort":{"experts":32,"seeds":list(SEEDS),"sparsities":list(SPARSITIES),
                  "sparse_conditions":10,"unique_dense_controls":5,"independent_unit":"source seed",
                  "historical_rerun_duplicates_independent":False},
        "primary_track":"rerun","full_validation":{"logged_evaluation_steps":"every 50 attempted updates, 0..5000",
            "saved_snapshot_steps":list(SNAPSHOT_STEPS),"union_steps":observation_steps,
            "validation_population":"32 original complete validation batches; aggregate token-summed CE / total tokens",
            "precision":"fp32 validation; TF32 enabled per rerun runtime configuration",
            "off_grid_steps":[25,2525],"off_grid_source":"saved checkpoint full-validation cache",
            "saved_checkpoint_not_assumed_at_every_logged_step":True},
        "optimizer":{"segments":[[0,2500],[2500,5000]],"attempted_updates_per_run":5000,
            "reset_at_step":2500,"model_weights_unchanged_at_reset":True,
            "applied_update_and_amp_skip_source":"per-attempt gradient_stats logs"},
        "token_probe":{"population":"2048 registered positions selected from validation batch 0",
            "complete_capacity_batch":"128 sequences x 128 tokens (16,384 tokens) retained for routing",
            "same_positions_across_conditions_and_steps":True,
            "precision":"protected fp32 diagnostic forward; TF32 disabled",
            "weighting":"equal target-token weights; mean of 2,048 token CE differences",
            "not_full_validation":True,"no_token_level_significance":True},
        "join_keys":["source_seed","sparsity","global_attempted_step","unique_dense_condition"],
        "analysis_script":"scripts/high_sparsity_coupling/recovery_advantage.py",
        "analysis_script_sha256":sha256(Path(__file__)),
        "source_model_tensors_loaded_or_modified":False,
        "test_split":{"availability":"not established from frozen data provenance",
            "prior_use_status":"no explicit test path / untouched-split manifest in frozen cohort",
            "inspected_or_evaluated":False,"data_configuration_fields_containing_test":test_metadata_fields},
    }
    write_json(OUT/"analysis_config.json",config)

    # Machine checks cover all required matching, coverage, precision, and token identities.
    max_token_recon=max(r["token_probe_reconciliation_abs_error"] for r in token_summaries)
    max_cache_recon=max(r["probe_ce_cache_reconciliation_max_abs_error"] for r in token_summaries)
    max_ce_crosscheck=max(f(r["absolute_difference"],0.0) for r in checkpoint_crosschecks)
    max_gap_identity=max(abs(f(r["identity_residual_gap_change_minus_components"])) for r in decomposition)
    max_precision_gap=max(abs(r["gap_difference_tf32_off_minus_on"]) for r in precision_rows)
    checkpoint_files_unchanged=True
    for path,identity in checkpoint_file_state_before.items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=identity:
            checkpoint_files_unchanged=False
    checks={
        "passed":True,
        "condition_step_joins":{"sparse_conditions":10,"unique_dense_controls":5,"unique_dense_per_seed":True,
            "observed_steps_per_condition":len(observation_steps),"primary_rows":len(performance),
            "checkpoint_snapshot_rows":len(checkpoint_crosschecks),"all_crosscheck_differences_zero":max_ce_crosscheck==0},
        "endpoint_labels":{"matches":sorted([{"seed":s,"sparsity":sp} for s,sp in EXPECTED_MATCHES],key=lambda x:(x["seed"],x["sparsity"])),
            "checked_against_strict_zero_from_authoritative_rerun_ce":True},
        "crossover":{"summary_rows":len(crossover),"interval_rows":len(decomposition),
            "max_gap_decomposition_identity_error":max_gap_identity,
            "strict_zero_threshold":True,"interpolation_used":False},
        "optimizer":{"per_attempt_rows_checked":75000,"segment_applied_and_skip_totals_reconciled":True,
            "step_2500_weights_same_before_after_reset":True},
        "token_alignment":{"rows":len(token_array_checks),"probe_positions":2048,
            "all_pair_token_position_ids_match":True,"all_steps_keep_same_ids_within_seed":True,
            "max_token_to_probe_ce_reconciliation_abs_error":max_token_recon,
            "max_probe_array_to_saved_probe_ce_error":max_cache_recon},
        "precision":{"primary_curve_track":"rerun","historical_not_mixed":True,
            "tf32_off_vs_on_endpoint_gap_max_abs_difference":max_precision_gap,
            "all_strict_endpoint_signs_agree":all(r["strict_sign_agrees"] for r in precision_rows),
            "intermediate_near_zero_tf32_sensitivity":"not directly measured; strict rerun curve retained with no tolerance"},
        "test_split":{"not_inspected":True,"untouched_split_established":False,
            "data_configuration_fields_containing_test":test_metadata_fields},
        "source_model_tensors":{"loaded":False,"modified":False,
            "checkpoint_files_tracked":len(checkpoint_file_state_before),
            "checkpoint_file_size_mtime_inode_unchanged":checkpoint_files_unchanged},
    }
    if max_token_recon>1e-7 or max_cache_recon>2e-5 or max_ce_crosscheck!=0 or max_gap_identity>1e-10 or not checkpoint_files_unchanged:
        checks["passed"]=False
        raise AssertionError(json.dumps(checks,indent=2))
    write_json(OUT/"validation_checks.json",checks)

    draw_figures(performance,token_summaries,internal,crossover)
    # Re-hash newly written token arrays for a downloadable artifact manifest.
    token_manifest=[]
    for path in sorted((OUT/"token_arrays").glob("*.npz")):
        token_manifest.append({"file":str(path.relative_to(OUT)),"sha256":sha256(path),"bytes":path.stat().st_size})
    write_json(OUT/"token_array_manifest.json",{"arrays":token_manifest})

    # Findings include exact observed times and are intentionally generated from all ten outcomes.
    lines=[
        "# Recovery advantage analysis v1",
        "",
        "Checkpoint-only descriptive analysis of all ten 32-expert sparse conditions against five unique same-seed dense controls. The rerun trajectory is primary; source seed is the replication unit (n=5), and .80/.85 within a seed are dependent. No new training, recovery, mask search, optimizer update, model forward, or causal intervention was run. No source model tensor was loaded or modified.",
        "",
        "## Interpretation in brief",
        "",
        "Every condition is already nonpositive at the step-0 masked-initialization observation; therefore no sparse condition first crosses from a positive gap during recovery. The step-0 difference is not an effect of recovery. All ten later change sign at least once in the dense 50-attempt validation trajectory, and only four finish at a nonpositive gap. Distinguish this initial status from the later regained/persistent endpoint advantage.",
        "",
        "The sparse--dense fixed-source/all-expert Pearson coupling divergence is generic to matching and nonmatching conditions. It is not a success explanation. The available comparisons do not isolate a predictive mechanism: observed performance tails vary by seed and sparsity, native accepted-routing and fixed-source probes answer different questions, and geometry/distortion move against moving dense references. No coherent causal explanation emerges.",
        "",
        "## Endpoint comparisons and observed persistence",
        "",
        "| Seed | Sparsity | Initial gap | First nonpositive | Persistent nonpositive tail begins | Supporting evaluations | Final gap | Endpoint label |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in crossover:
        onset=r["earliest_persistent_nonpositive_step"] or "none"
        support=r["persistence_observation_count"] if r["persistent_tail_demonstrated"] else 0
        lines.append(f'| {r["seed"]} | {r["sparsity"]:.2f} | {r["initial_gap"]:+.6f} | step {r["first_observed_nonpositive_step"]} (already nonpositive at initialization) | {onset} | {support} | {r["final_gap"]:+.6f} | {"matches/beats" if r["endpoint_matches_or_beats_dense"] else "does not match"} |')
    lines.extend([
        "",
        "“Persistent tail” means at least two observed validation evaluations from the reported start through step 5,000 remain <= 0; it is not a claim between observations. The exact time of any crossing is unknown. Every 50-step validation observation is a scalar evaluation, not a saved checkpoint; only the explicitly listed snapshot steps have model checkpoints. At step 2,500, the optimizer/scaler reset leaves model weights unchanged.",
        "",
        "See [crossover_summary.csv](./crossover_summary.csv) for later positive/nonpositive reversals and complete observed intervals; [interval_gap_decomposition.csv](./interval_gap_decomposition.csv) decomposes every adjacent change as `Δgap = Δsparse CE - Δdense CE`.",
        "",
        "## Token gains and losses",
        "",
        "Per-token results are only the fixed 2,048-position probe from validation batch 0, stored for each sparse condition and the same-seed dense checkpoint at the 16 saved snapshots. Full 128x128 batches were retained in the underlying forward caches for MoE capacity behavior. Token deltas use protected FP32 diagnostic arrays with TF32 disabled; full-validation scalar CE uses the separate rerun setting (FP32, TF32 enabled). The token probe is not full validation. Each position receives equal target-token weight; its mean delta reconciles with the sparse-minus-dense probe CE. The full 32-batch validation total is not substituted by this probe.",
        "",
        "Token fractions, quantiles, signed contribution totals, and top-tail concentration appear in [token_gain_loss_summary.csv](./token_gain_loss_summary.csv); top-k shares use k% of all 2,048 positions, separately for each sign. Fixed-position arrays and checkpoint/cache hashes are in [token_arrays/](./token_arrays) and [token_array_provenance.csv](./token_array_provenance.csv). Tokens/positions are descriptive observations, not independent replicates; no token-level significance tests are used. Verified domain labels are unavailable.",
        "",
        "For the four endpoint matches, the token probe is mixed rather than uniformly favorable:",
        "",
        "| Seed | Sparsity | Probe mean delta | Full-validation gap | Improved positions | Worsened positions | Median delta |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    winner_token_rows=[r for r in token_summaries if i(r["step"])==5000 and (i(r["seed"]),f(r["sparsity"])) in EXPECTED_MATCHES]
    for r in winner_token_rows:
        lines.append(f'| {r["seed"]} | {r["sparsity"]:.2f} | {r["mean_delta"]:+.5f} | {r["full_validation_gap_same_step"]:+.5f} | {r["fraction_improved"]:.1%} | {r["fraction_worsened"]:.1%} | {r["median"]:+.4f} |')
    lines.extend([
        "",
        "In three of the four matching conditions, more evaluated probe positions worsen than improve, while the mean delta is favorable; the net probe gain is therefore carried by larger-magnitude improvements on a subset, not a majority-token win. Seed 61/.80 is approximately evenly split. The top-k contribution shares in the CSV use k% of all 2,048 probe positions, separately for positive and negative mass. Some nonmatching conditions also have a favorable probe mean (notably seed 29/.80), while seed 43/.80's probe mean is much more favorable than its near-zero full-validation gap; do not generalize probe outcomes to full validation.",
        "",
        "## Internal measurements aligned to observed steps",
        "",
        "The long-form [aligned_internal_trajectories.csv](./aligned_internal_trajectories.csv) joins only existing measurements at exact seed, sparsity, and checkpoint step. It retains each metric's probe/reference, layer, precision/population definition, and source. Fixed-source/all-expert coupling is the same-input raw-router-score/post-GELU-activation Pearson probe; it increases in all sparse conditions, including all six endpoint nonmatches, while its matched dense comparator decreases. This generic pattern cannot be sufficient for an endpoint match.",
        "",
        "Native routing entropy/margin and probe capacity acceptance/load are on native hidden inputs and selected probe positions with the complete batch capacity preserved. They are also recorded for each unique dense control; the same dense rows are explicitly reused for both sparsity comparisons, not counted twice. Fixed-source output errors are the identical-input expert-output comparison; native sparse/dense vectors are not identical inputs. Layerwise representation CKA uses a same-step dense, moving reference. Fixed-gate distortion reports the squared numerator D and squared reference-output energy separately; a rising normalized ratio is not itself an absolute increase. Gradient geometry exists only at 0/2,500/5,000 on previously observed training batches, not between those points. Update-event measurements are sparse selected steps; applied updates and AMP skips are exact at attempted-step resolution.",
        "",
        "## Competing explanations and limits",
        "",
        "Four strict endpoint matches are seed 7 at both sparsities, seed 43 at .80, and seed 61 at .80. Seeds 43 and 61 show the requested paired-sparsity contrast: .80 matches, .85 does not; that contrast is sparsity-confounded and within-seed, not independent replication. The .80 seed-43 endpoint margin is especially small and is retained without a post-hoc tolerance. TF32 off/on endpoint gap signs agree; sensitivity is tabulated separately in [precision_sensitivity.csv](./precision_sensitivity.csv).",
        "",
        "TF32-off/on endpoint gap signs agree, and the maximum endpoint gap shift is reported in [validation_checks.json](./validation_checks.json). The closest endpoint margin is not reversed by this setting change. Intermediate near-zero gaps were not directly reevaluated under both TF32 settings; the curve uses strict zero and has no post-hoc tolerance.",
        "",
        "All recovery runs deliberately reset AdamW/scaler at step 2,500; model weights match before and after the reset. There is no no-reset control, so the reset cannot be assigned causal responsibility. Data stream and validation batch hashes match within each seed's sparse/dense pair. Training CE and auxiliary routing loss are reported separately as recent training-window means; they are not substituted for validation CE. Routing capacity effects are probe-specific and must not be interpreted as whole-validation token routing.",
        "",
        "The frozen configurations declare train and validation inputs, but no explicit test path or untouched-test provenance. No test split was inspected or evaluated, and validation is not relabeled as test. Historical measurements were excluded from the main trajectory because the TF32 setting differs; they are not counted as independent replications. No novel, causal, or beyond-cohort conclusion is supported.",
        "",
        "## Recommendation",
        "",
        "No focused intervention is recommended from these data: the generic coupling trend occurs in winners and nonwinners, and the endpoint/persistence pattern is seed-specific and sparsity-confounded. There is no specific causal mechanism sufficiently discriminated to justify selecting a manipulation without cherry-picking.",
        "",
        "## Figures",
        "",
        "- [A. Absolute full-validation CE](./figures/A_absolute_validation_ce.svg)",
        "- [B. All ten validation gaps](./figures/B_validation_gap.svg)",
        "- [C. Probe token gains/losses and concentration](./figures/C_probe_gain_loss.svg)",
        "- [D. Layerwise internal trajectories](./figures/D_layerwise_internal.svg)",
        "",
        "## Checks",
        "",
        "Focused joins, unique dense reuse, checkpoint-cache/full-log precision agreement, strict crossover/reversal calculation, gap decomposition, token-position alignment, token-to-probe-CE reconciliation, and source-tensor non-mutation are recorded in [validation_checks.json](./validation_checks.json).",
        "See the separate [competing-explanations matrix](./competing_explanations.md) for candidate status and counterevidence.",
    ])
    (OUT/"findings.md").write_text("\n".join(lines)+"\n")

    def route_direction_summary(measure, model_role, want_increase):
        values=defaultdict(lambda:{0:[],5000:[]})
        for row in internal:
            if row["metric_family"]!="native_routing" or row["measure"]!=measure or row["model_role"]!=model_role:
                continue
            if i(row["step"]) not in (0,5000):
                continue
            key=(i(row["seed"]),f(row["sparsity"])) if model_role=="sparse" else i(row["seed"])
            values[key][i(row["step"])].append(f(row["value"]))
        changes=[]
        for by_step in values.values():
            if by_step[0] and by_step[5000]:
                changes.append(float(np.mean(by_step[5000])-np.mean(by_step[0])))
        n_expected=10 if model_role=="sparse" else 5
        n_direction=sum((delta>0)==want_increase for delta in changes)
        return f"{n_direction}/{n_expected}", min(changes), max(changes)

    entropy_sparse=route_direction_summary("full_softmax_entropy_mean","sparse",False)
    entropy_dense=route_direction_summary("full_softmax_entropy_mean","dense_control",False)
    confidence_sparse=route_direction_summary("top1_softmax_confidence_mean","sparse",True)
    confidence_dense=route_direction_summary("top1_softmax_confidence_mean","dense_control",True)
    accept_sparse=route_direction_summary("probe_capacity_acceptance_fraction","sparse",True)
    accept_dense=route_direction_summary("probe_capacity_acceptance_fraction","dense_control",True)
    competitor_text=[
        "# Competing explanations",
        "",
        "This is a checkpoint-only comparison of saved outcomes. Statuses describe what these records support; none is a causal conclusion.",
        "",
        "| Candidate explanation | Evidence and counterevidence | Assessment |",
        "|---|---|---|",
        f"| Fixed-source/all-expert coupling increase explains endpoint matches | Sparse coupling rises in all ten sparse conditions and matched dense coupling falls; the pattern occurs in all four matches and all six nonmatches. | **Contradicted as a sufficient separator or explanation.** It may be an accompanying generic sparse-recovery change. |",
        f"| Native router confidence/entropy/acceptance uniquely accompanies success | Across sparse runs, native-probe entropy falls in {entropy_sparse[0]} (mean range {entropy_sparse[1]:+.3f} to {entropy_sparse[2]:+.3f}), top-1 confidence rises in {confidence_sparse[0]}, and selected-probe capacity acceptance rises in {accept_sparse[0]}. The same directions also occur in dense controls: entropy {entropy_dense[0]}, confidence {confidence_dense[0]}, acceptance {accept_dense[0]}. | **Contradicted as a winner-specific pattern;** whole-validation routing consequences remain unresolved because routing summaries are probe-level. |",
        "| Broad tokenwise gain explains the full-validation advantage | At step 5,000, three matching conditions have more worsened than improved positions in the fixed 2,048-position probe; the fourth is near an even split. Several nonmatches also have negative probe means, and seed 43/.80's probe mean differs substantially from its near-zero full-validation gap. | **Contradicted as a uniform per-position account;** probe results do not establish full-validation token coverage. |",
        "| Layerwise geometry or fixed-gate distortion uniquely explains the advantage | Fixed-source CKA declines across the cohort; dense-native comparisons follow different inputs and may move oppositely. Fixed-gate D and reference energy both fall while the normalized ratio can rise. These references/probes do not isolate a success-specific transition. | **Contradicted as a cohort-wide unique separator;** the relation to performance remains unresolved. |",
        "| Optimizer/scaler reset at step 2,500 causes the late advantage | The reset is deliberate in every sparse and dense run; model hashes are unchanged across the boundary. No no-reset recovery exists. | **Unresolved.** No causal attribution is possible. |",
        "| Sparse regularization or reduced overfitting explains lower validation CE | Validation gaps and training CE are observed, but there is no demonstrably untouched test split in frozen provenance and no causal intervention. Training CE and auxiliary routing loss are separate and use the training stream. | **Unresolved;** no generalization claim. |",
        "| Update-count differences explain near-zero endpoint margins | Applied updates range from 4,997 to 5,000 of 5,000 attempts, with exact AMP skips recorded and not silently adjusted away. The observed endpoint labels are preserved under the available TF32 off/on endpoint comparison. | **Possible small contribution, not isolated.** Intermediate TF32 sensitivity and a causal update-count effect are unavailable. |",
        "| Seed and sparsity jointly shape the observed endpoint pattern | Seeds 43 and 61 match at .80 but not .85, while sparsity levels within a seed are dependent and not randomized across endpoint labels. Only five source seeds are available. | **Plausible descriptive dependence;** sparsity and seed do not identify a mechanism. |",
        "",
        "The endpoint CE trajectory is based on the full validation population at 103 observed steps. Only 16 of those steps have saved model checkpoints. There is no evidence here for a narrow causal event between observations. No focused intervention is recommended: the candidate signals above are either generic across matching/nonmatching conditions or not causally identified.",
        "",
        "Full metrics and provenance: [performance_trajectory.csv](./performance_trajectory.csv), [crossover_summary.csv](./crossover_summary.csv), [token_gain_loss_summary.csv](./token_gain_loss_summary.csv), [aligned_internal_trajectories.csv](./aligned_internal_trajectories.csv), [coverage_and_exclusions.csv](./coverage_and_exclusions.csv).",
    ]
    (OUT/"competing_explanations.md").write_text("\n".join(competitor_text)+"\n")

    completion={
        "status":"complete","scope":"checkpoint-only; no model evaluation forwards launched",
        "primary_conditions":10,"unique_dense_controls":5,"independent_source_seeds":5,
        "observed_full_validation_steps_per_condition":len(observation_steps),
        "saved_model_snapshot_steps":list(SNAPSHOT_STEPS),
        "noncheckpoint_scalar_evaluations_not_claimed_as_saved_models":True,
        "token_probe_positions_per_condition_step":2048,
        "token_probe_snapshot_steps":len(SNAPSHOT_STEPS),
        "outputs":{
            "performance_rows":len(performance),"crossover_rows":len(crossover),
            "gap_decomposition_rows":len(decomposition),"token_summary_rows":len(token_summaries),
            "token_array_files":len(token_manifest),"aligned_internal_rows":len(internal),
            "update_event_group_summaries":update_measurements,
            "figure_files":4,"competing_explanations_file":"competing_explanations.md",
        },
        "strict_endpoint_matches":[{"seed":s,"sparsity":sp} for s,sp in sorted(EXPECTED_MATCHES)],
        "all_focused_checks_passed":checks["passed"],
        "source_model_tensors_loaded_or_modified":False,
        "test_split_inspected_or_evaluated":False,
    }
    write_json(OUT/"completion.json",completion)
    print(json.dumps(completion,indent=2))


if __name__ == "__main__":
    main()
