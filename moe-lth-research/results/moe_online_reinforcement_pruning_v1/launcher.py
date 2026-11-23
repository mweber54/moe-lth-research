#!/usr/bin/env python3
"""Launch all 18 branch jobs with GPU admission queuing and monitoring."""
import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime

OUT = Path(__file__).resolve().parent
SEEDS = [7, 17, 29]
ORDERS = ['A', 'B']
ARMS = ['untouched', 'random-small', 'high-reinforcement-small']

def launch_branch(seed, order, arm, dry_run=False):
    """Launch a single branch job."""
    cmd = ['python3', str(OUT / 'branch.py'), str(seed), order, arm]
    
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

def check_completion(seed, order, arm):
    """Check if branch is complete."""
    done_file = OUT / 'branches' / f'seed{seed}_{order}' / arm / 'final_step_2500.json'
    return done_file.exists()

def check_running(seed, order, arm):
    """Check if branch is currently running."""
    lock_file = OUT / 'branches' / f'seed{seed}_{order}' / arm / 'lock'
    return lock_file.exists()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Launch MOE branch continuations')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without running')
    parser.add_argument('--sequential', action='store_true', help='Run jobs sequentially instead of parallel')
    parser.add_argument('--first-only', action='store_true', help='Only run seed7_A for all three arms')
    parser.add_argument('--monitor', action='store_true', help='Monitor ongoing jobs without launching new ones')
    args = parser.parse_args()
    
    jobs = []
    
    # Build job list
    if args.first_only:
        jobs = [(7, 'A', arm) for arm in ARMS]
    else:
        for seed in SEEDS:
            for order in ORDERS:
                for arm in ARMS:
                    jobs.append((seed, order, arm))
    
    print(f"Loaded {len(jobs)} branch jobs")
    
    if args.monitor:
        # Just monitor existing jobs
        for seed, order, arm in jobs:
            if check_completion(seed, order, arm):
                print(f"seed{seed}_{order} {arm}: ✓ completed")
            elif check_running(seed, order, arm):
                print(f"seed{seed}_{order} {arm}: ⏳ running")
            else:
                print(f"seed{seed}_{order} {arm}: pending")
        return
    
    if args.sequential:
        # Run one at a time
        for i, (seed, order, arm) in enumerate(jobs):
            if check_completion(seed, order, arm):
                print(f"[{i+1}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ already complete")
                continue
            
            print(f"[{i+1}/{len(jobs)}] Launching seed{seed}_{order} {arm}...")
            proc = launch_branch(seed, order, arm, dry_run=args.dry_run)
            if proc:
                stdout, stderr = proc.communicate()
                if proc.returncode == 0:
                    print(f"  ✓ completed")
                else:
                    print(f"  ✗ failed (code {proc.returncode})")
                    if stderr:
                        print(f"  stderr: {stderr[:200]}")
    else:
        # Launch all in parallel (but respect GPU admission)
        active_jobs = {}
        pending_idx = 0
        
        while pending_idx < len(jobs) or active_jobs:
            # Check completed jobs
            completed = []
            for job_id, proc in list(active_jobs.items()):
                if proc.poll() is not None:
                    seed, order, arm = jobs[job_id]
                    if proc.returncode == 0:
                        print(f"[{job_id+1}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ completed")
                    else:
                        print(f"[{job_id+1}/{len(jobs)}] seed{seed}_{order} {arm}: ✗ failed")
                    completed.append(job_id)
            
            for job_id in completed:
                del active_jobs[job_id]
            
            # Launch new jobs (up to reasonable limit)
            while pending_idx < len(jobs) and len(active_jobs) < 2:  # Limit parallelism
                seed, order, arm = jobs[pending_idx]
                
                if check_completion(seed, order, arm):
                    print(f"[{pending_idx+1}/{len(jobs)}] seed{seed}_{order} {arm}: ✓ already complete, skipping")
                    pending_idx += 1
                    continue
                
                print(f"[{pending_idx+1}/{len(jobs)}] Launching seed{seed}_{order} {arm}...")
                proc = launch_branch(seed, order, arm, dry_run=args.dry_run)
                if proc:
                    active_jobs[pending_idx] = proc
                pending_idx += 1
            
            # Wait a bit before next check
            if active_jobs:
                time.sleep(10)
        
        print(f"\nAll {len(jobs)} jobs completed")

if __name__ == '__main__':
    main()
