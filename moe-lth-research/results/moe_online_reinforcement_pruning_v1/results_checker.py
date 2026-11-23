#!/usr/bin/env python3
"""Collect and analyze branch execution results."""
import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
SEEDS = [7, 17, 29]
ORDERS = ['A', 'B']
ARMS = ['untouched', 'random-small', 'high-reinforcement-small']

def collect_results():
    """Collect results from all completed branches."""
    results = {}
    
    for seed in SEEDS:
        for order in ORDERS:
            cell_key = f'{seed}_{order}'
            results[cell_key] = {}
            
            for arm in ARMS:
                complete_file = OUT / 'branches' / cell_key / arm / 'final_step_2500.json'
                
                if complete_file.exists():
                    with open(complete_file) as f:
                        data = json.load(f)
                    results[cell_key][arm] = {
                        'status': data['status'],
                        'completed_step': data['completed_step'],
                        'measurements': data.get('measurements', {})
                    }
                else:
                    results[cell_key][arm] = {'status': 'MISSING'}
    
    return results

def compute_deltas(results):
    """Compute Delta values for each cell."""
    deltas = {}
    
    for seed in SEEDS:
        for order in ORDERS:
            cell_key = f'{seed}_{order}'
            if cell_key not in results:
                continue
            
            cell_results = results[cell_key]
            
            # Check if both random and high arms are complete
            random_status = cell_results.get('random-small', {}).get('status')
            high_status = cell_results.get('high-reinforcement-small', {}).get('status')
            
            if random_status == 'COMPLETED' and high_status == 'COMPLETED':
                # Delta = CE_high - CE_random at step 2500
                # TODO: Extract actual CE values from checkpoint logs
                deltas[cell_key] = None  # Will be filled once we extract CE values
    
    return deltas

def print_summary(results):
    """Print results summary."""
    print("\n" + "="*70)
    print("MOE ONLINE REINFORCEMENT PRUNING V1 - BRANCH EXECUTION RESULTS")
    print("="*70)
    
    complete = 0
    total = 0
    
    for seed in SEEDS:
        print(f"\n[Seed {seed}]")
        for order in ORDERS:
            cell_key = f'{seed}_{order}'
            cell = results.get(cell_key, {})
            print(f"  {cell_key}:")
            
            for arm in ARMS:
                arm_result = cell.get(arm, {})
                status = arm_result.get('status', 'PENDING')
                step = arm_result.get('completed_step', '?')
                
                if status == 'COMPLETED':
                    print(f"    {arm}: ✓ step {step}")
                    complete += 1
                elif status == 'MISSING':
                    print(f"    {arm}: ⏳ in progress")
                else:
                    print(f"    {arm}: ✗ {status}")
                
                total += 1
    
    print(f"\n{'='*70}")
    print(f"Status: {complete}/{total} branches complete")
    
    if complete == total:
        print("✓ ALL BRANCHES COMPLETE - Ready for Delta analysis")
    elif complete > 0:
        print(f"⏳ {total - complete} branches still running/pending")
    else:
        print("⏳ All branches pending or in progress")
    
    return complete == total

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check branch execution status')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--complete', action='store_true', help='Only show completed branches')
    args = parser.parse_args()
    
    results = collect_results()
    
    if args.status or not (args.complete):
        all_complete = print_summary(results)
    
    if args.complete:
        complete_cells = []
        for seed in SEEDS:
            for order in ORDERS:
                cell_key = f'{seed}_{order}'
                cell = results.get(cell_key, {})
                if all(cell.get(arm, {}).get('status') == 'COMPLETED' for arm in ARMS):
                    complete_cells.append(cell_key)
        
        print(f"Completed cells: {complete_cells}")
        return 0 if len(complete_cells) == 6 else 1

if __name__ == '__main__':
    main()
