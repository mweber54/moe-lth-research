"""Consolidate previously measured tables (rerun track) into per-seed views. No new measurement."""
import pandas as pd
from fge_common import *

REP = ROOT / 'results/high_sparsity_coupling_v1/report'
O = OUT / 'consolidated_existing_metrics'

EXISTING_TABLES = (
    ('representation_trajectories', 'representation_trajectories'),
    ('model_function_trajectories', 'model_function_trajectories'),
    ('expert_function_trajectories_n128', 'expert_function_trajectories'),
    ('geometry_trajectories', 'geometry_trajectories'),
    ('update_budgets', 'update_budgets'),
    ('geometry_endpoint_summary', 'geometry_endpoint_summary'),
)


def main():
    O.mkdir(exist_ok=True)
    pc = pd.read_csv(REP / 'paired_seed_coupling_changes.csv')
    keep = ['track', 'seed', 'sparsity', 'condition', 'mode', 'selection', 'pair_definition', 'coefficient', 'aggregation',
            'n_valid_units', 'sparse_initial', 'sparse_final', 'dense_initial', 'dense_final', 'sparse_change', 'dense_change',
            'endpoint_difference', 'difference_in_changes']
    pc = pc[pc.track == 'rerun'][keep].copy()
    pc['initial_gap_sparse_minus_dense'] = pc.sparse_initial - pc.dense_initial
    pc['final_gap_sparse_minus_dense'] = pc.endpoint_difference
    pc['gap_shrinks_toward_zero'] = pc.final_gap_sparse_minus_dense.abs() < pc.initial_gap_sparse_minus_dense.abs()
    table(O / 'coupling_per_condition_initial_final.csv', pc.to_dict('records'))

    labels = pd.read_csv(REP / 'rerun_tickets.csv')
    agree = pd.read_csv(REP / 'ticket_label_agreement.csv')
    lab = labels.merge(agree[['seed', 'sparsity', 'strict_success_historical', 'common_tf32_off_gap_rerun', 'label_agreement']], on=['seed', 'sparsity'])
    lab = lab.rename(columns={'strict_success': 'matches_or_beats_matched_dense_5000_attempted_rerun',
                              'strict_success_historical': 'matches_or_beats_matched_dense_5000_attempted_historical'})
    lab['label_tf32_off'] = lab.common_tf32_off_gap <= 0
    table(O / 'ticket_labels_by_condition.csv', lab.to_dict('records'))

    sel = pc[(pc['mode'] == 'fixed_source') & (pc.selection == 'all_expert') & (pc.pair_definition == 'identical_fixed_pairs')
             & (pc.coefficient == 'pearson')]
    j = sel.merge(lab[['seed', 'sparsity', 'gap', 'matches_or_beats_matched_dense_5000_attempted_rerun']], on=['seed', 'sparsity'])
    j = j.drop(columns=['aggregation']).drop_duplicates()
    table(O / 'coupling_vs_ce_gap_by_condition.csv', j.to_dict('records'))

    for name, src in EXISTING_TABLES:
        d = pd.read_csv(REP / f'{src}.csv')
        if 'track' in d:
            d = d[d.track == 'rerun']
        table(O / f'{name}.csv', d.to_dict('records'))

    c = pd.read_csv(REP / 'coupling_trajectories.csv')
    c = c[c.track == 'rerun']
    table(O / 'coupling_trajectories_rerun.csv', c.to_dict('records'))
    print(len(pc), len(j))


if __name__ == '__main__':
    main()
