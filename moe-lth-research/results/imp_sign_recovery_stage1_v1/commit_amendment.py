from pathlib import Path
import csv,hashlib,subprocess,datetime,math
OUT=Path(__file__).resolve().parent;ROOT=Path.cwd();now=datetime.datetime.now(datetime.timezone.utc);date='2026-11-11'  # Date supplied in the client America/Indianapolis context
old_rel='moe-lth-research/results/imp_sign_recovery_audit_v2/preregistration.md';old=subprocess.check_output(['git','show',f'1f60d90:{old_rel}']);assert old==(ROOT/'results/imp_sign_recovery_audit_v2/preregistration.md').read_bytes()
assert not (OUT/'jobs').exists() and not (OUT/'training_started.json').exists()
checked=[]
for folder in ['imp_sign_recovery_audit_v1','imp_sign_recovery_audit_v2','imp_sign_placement_recovery_v1','imp_sign_placement_recovery_v2','imp_sign_recovery_stage1_v1']:
 path=ROOT/'results'/folder
 artifacts=[str(p) for p in path.rglob('*') if p.is_file() and (p.name=='gradient_stats.jsonl' or p.name=='interruption.pt' or p.name=='final_recovered.pt' or p.name=='attempt_ledger.jsonl')]
 assert not artifacts,(folder,artifacts)
 checked.append(dict(directory=str(path),registered_stage1_training_artifacts=0,registered_stage1_runs_started=0,registered_stage1_attempted_updates=0,checked_utc=now.isoformat(),scope='new sign-only Stage1; historical reused originals/full-shuffle training excluded'))
with (OUT/'pre_amendment_training_audit.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(checked[0]));w.writeheader();w.writerows(checked)
tcrit=math.sqrt(2*.95**2/(1-.95**2))
text=f'''# Stage-1 amendment — {date} (America/Indianapolis)

Recorded UTC: {now.isoformat()}. This prospective amendment supersedes commit `1f60d9013d2f61b9fac5dd1d5eb1f798798ee744` only where that registration used the interaction level as “reopening,” and supersedes its Stage-1 interim gate accordingly. The original registration and its file remain unchanged. It is made before any registered sign-only Stage-1 training: zero runs and zero attempted updates, with evidence in `pre_amendment_training_audit.csv`. Historical original and full-shuffle runs are not newly registered sign-only recoveries.

## Unchanged interaction and weighting

For seed s in {{7,17,29}}, order o in {{A,B}}, perturbation draw j in {{1,2,3}}, and mask m in {{IMP,random1,random2}}, let delta_m,s,o,j(t) = CE_sign_m,s,o,j(t) - CE_original_m,s,o(t). Originals are the verified compatible existing original runs. Keep the two fixed random masks distinct, paired by their actual registered draw IDs, and equally weighted:

    i_s,o,j(t) = delta_IMP,s,o,j(t)
                 - (delta_random1,s,o,j(t) + delta_random2,s,o,j(t))/2
    i_s,j(t)   = (i_s,A,j(t) + i_s,B,j(t))/2
    I_S,s(t)   = mean_j i_s,j(t).

Orders A/B share each draw's initial perturbed state and are paired blocks, not independent draws. Do not manufacture extra pairing, re-pair draws based on results, or pool the random masks before recording their individual losses and penalties. Preserve these same weights for all following calculations.

## Distinct quantities and intervals

For each source seed:

- Level L_s = I_S,s(2000).
- Reopening R_s = I_S,s(2000) - I_S,s(250).

Construct draw values before estimating uncertainty. L_s,j = i_s,j(2000). For R, first difference checkpoints **within the same recovery run**, also differencing its same-mask/order original comparator:

    d_m,s,o,j = [CE_sign_m,s,o,j(2000) - CE_sign_m,s,o,j(250)]
                - [CE_original_m,s,o(2000) - CE_original_m,s,o(250)]
    r_s,o,j   = d_IMP,s,o,j - (d_random1,s,o,j+d_random2,s,o,j)/2
    R_s,j     = (r_s,A,j+r_s,B,j)/2.

Then L_s = mean_j L_s,j and R_s = mean_j R_s,j, j=1,2,3. This preserves endpoint covariance within each run, order pairing within each draw, and actual registered draw pairing across the IMP and two random-mask arms. Report all three L_s,j and R_s,j values, plus order-specific and mask-specific losses/penalties/checkpoint changes.

For Q in {{L,R}}, use the two-sided 95% draw-level Student-t interval:

    SD_Q = sqrt(sum_j (Q_s,j-mean_j Q_s,j)^2 / (3-1))
    CI_Q = mean_j Q_s,j +/- {tcrit:.15g} * SD_Q/sqrt(3).

The t quantile is t_(0.975,df=2). For R compute this from the three **paired R_s,j differences**, never by subtracting independently calculated endpoint intervals or standard errors. These n=3 intervals are conditional on the tested source seed, two fixed random masks and two fixed orders; they do not treat orders or masks as additional independent draws or establish population coverage across untested seeds/masks.

## Per-seed decision and Stage-1 gate

A seed passes only when all four strict inequalities hold:

1. L_s > **0.04972095 nats**.
2. R_s > **0.02486047 nats**.
3. Lower bound of CI_L > 0.
4. Lower bound of CI_R > 0.

Use these exact displayed decimal thresholds. Point estimates test practical size; positive lower bounds test direction. The R threshold is a **prospective design choice**, not an empirically established noise bound for R. Equality at a boundary does not pass.

Define level_pass = (L_s > 0.04972095 and lower(CI_L)>0), increase_pass = (R_s > 0.02486047 and lower(CI_R)>0), and joint_pass = level_pass and increase_pass. Report jointly passing, level-only, increase-only and neither outcomes explicitly, including all four component booleans. Failure to pass is not proof of no effect.

The Stage-1 interim gate passes iff **the same two or more of seeds {{7,17,29}} have joint_pass=True**. Two seeds passing level and a different two passing increase is insufficient if fewer than two pass both. If a later stage is separately authorized, the eventual four-of-five sign criterion uses exactly this per-seed joint_pass definition for seeds {{7,17,29,43,61}}, rather than different seed sets for its two components.

## Hard authorization boundary

Only **54 sign-shuffle recoveries** are authorized: 3 source seeds x 2 orders x 3 fixed mask variants x 3 draws. Each is capped at **2500 attempted updates**, including AMP-skipped updates; aggregate cap **135000 attempted updates**. No compensation updates, automatic retraining, extra seeds, other perturbation arms, second-segment training, reset-and-extension, or step-5000 continuation is authorized. A first-segment failure stops dispatch rather than silently replaying attempts. Preserve attempted-update accounting.

**Stop after Stage 1 regardless of gate outcome.** The gate is reported, not an authorization to advance. Later-stage references above do not authorize those stages. Preserve the original readout protocol for Stage-1 steps 0,250,500,1000,1500,2000,2500 and the native first-segment optimization/data protocol. Recheck storage immediately before dispatch against the registered **188628797420.25-byte** first-stage requirement including planned allowance, accounting for current outputs; if it does not fit, stop without deleting historical artifacts. Preserve native snapshots and resumable first-segment state.

At completion report individual draw values and, per seed/order, absolute IMP and each random-mask CE, their separate sign-shuffle penalties, I_S(250), I_S(2000) and their paired difference. Report actual attempted/applied/skipped updates, retained bytes and elapsed slot-hours. No mechanism or no-effect claim follows solely from an interim pass/fail.
'''
path=OUT/f'amendment_{date}.md';path.write_text(text)
print(path);print('sha256',hashlib.sha256(path.read_bytes()).hexdigest())
