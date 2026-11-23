#!/usr/bin/env python3
"""Launch all 18 MOE branch continuation jobs with parallel GPU scheduling."""
import subprocess
import sys
import time
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
SEEDS = [7, 17, 29]
ORDERS = ['A', 'B']
ARMS = ['untouched', 'random-small', 'high-reinforcement-small']

def launch_branch(seed, order, arm, dry_run=False):
    """Launch a single branch continuation job."""
    cmd = ['python3', str(OUT / 'branch_v2.py'), str(seed), order, arm]
    
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return None
    
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(OUT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return proc
    except Exception as e:
        print(f"Failed to launch seed{seed}_{order} {arm}: {e}", file=sys.stderr)
        return None

def check_complete(seed, order, arm):
    """Check if branch job is complete."""
    done = OUT / 'branches' / f'seed{seed}_{order}' / arm / 'final_step_2500.json'
    return done.exists()

def check_running(seed, order, arm):
    """Check if branch job lock exists."""
    lock = OUT / 'branches' / f'seed{seed}_{order}' / arm / 'lock'
    return lock.exists()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Launch MOE branch continuations (seed 250→2500)')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without running')
    parser.add_argument('--sequential', action='store_true', help='Run jobs sequentially')
    parser.add_argument('--first-only', action='store_true', help='Only run seed7_A for all 3 arms')
    parser.add_argument('--monitor', action='store_true', help='Monitor ongoing jobs without launching')
    args = parser.parse_args()
    
    # Build job list
    jobs = []
    if args.first_only:
        jobs = [(7, 'A', arm) for arm in ARMS]
    else:
        for seed in SEEDS:
            for order in ORDERS:
                for arm in ARMS:
                    jobs.append((seed, order, arm))
    
    print(f"Loaded {len(jobs)} branch jobs (seed 250→2500)")
    
    if args.monitor:
        # Monitor mode: check status without launching
        complete_count = 0
        for seed, order, arm in jobs:
            if check_complete(seed, order, arm):
                print(f"seed{seed}_{order} {arm}: ✓ completed")
                complete_count += 1
            elif check_running(seed, order, arm):
                print(f"seed{seed}_{order} {arm}: ⏳ running")
            else:
                print(f"seed{seed}_{order} {arm}: pending")
        print(f"\nTotal: {complete_count}/{len(jobs)} complete")
        return
    
    if args.sequential:
        # Sequential execution mode
        for i, (seed, order, arm) in enumerate(jobs, 1):
            if check_complete(seed, order, arm):
                print(f"[{i:2d}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ already complete")
                continue
            
            print(f"[{i:2d}/{len(jobs)}] Launching seed{seed}_{order} {arm}...")
            proc = launch_branch(seed, order, arm, dry_run=args.dry_run)
            if proc:
                stdout, stderr = proc.communicate()
                if proc.returncode == 0:
                    print(f"       ✓ completed")
                else:
                    print(f"       ✗ failed")
                    if stderr:
                        print(f"       {stderr[:300]}")
    else:
        # Parallel mode: launch up to N at once respecting GPU admission
        active_jobs = {}
        pending_idx = 0
        max_parallel = 2  # Conservative GPU limit
        
        while pending_idx < len(jobs) or active_jobs:
            # Check completed jobs
            completed_ids = []
            for job_id, proc in list(active_jobs.items()):
                poll = proc.poll()
                if poll is not None:
                    seed, order, arm = jobs[job_id]
                    if poll == 0:
                        print(f"[{job_id+1:2d}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ completed")
                    else:
                        print(f"[{job_id+1:2d}/{len(jobs)}] seed{seed}_{order} {arm}: ✗ failed")
                    completed_ids.append(job_id)
            
            for job_id in completed_ids:
                del active_jobs[job_id]
            
            # Launch new jobs (up to parallelism limit)
            while pending_idx < len(jobs) and len(active_jobs) < max_parallel:
                seed, order, arm = jobs[pending_idx]
                
                if check_complete(seed, order, arm):
                    print(f"[{pending_idx+1:2d}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ already complete, skipping")
                    pending_idx += 1
                    continue
                
                print(f"[{pending_idx+1:2d}/{len(jobs)}] Launching seed{seed}_{order} {arm}...")
                proc = launch_branch(seed, order, arm, dry_run=args.dry_run)
                if proc:
                    active_jobs[pending_idx] = proc
                pending_idx += 1
            
            # Wait before next check
            if active_jobs:
                time.sleep(10)
        
        print(f"\n{'='*60}")
        print(f"All {len(jobs)} jobs processed")
        complete = sum(1 for s, o, a in jobs if check_complete(s, o, a))
        print(f"Complete: {complete}/{len(jobs)}")

if __name__ == '__main__':
    main()
