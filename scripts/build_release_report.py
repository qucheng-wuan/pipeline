#!/usr/bin/env python3
"""CPU-only report from completed, authenticated experiment artifacts.

This tool does not select another checkpoint, execute a model, read GT arrays,
or promote release.json. Missing full-roster evidence is a hard error. Large
prediction/checkpoint arrays were verified by the recorded independent gates;
this report verifies their seals, identities and the small metric artifacts.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
METHODS = ('stage1_native', 'stage1_resized', 'stage1_lidf')
REGIONS = ('all', 'transparent', 'background')
DOMAINS = ('combined', 'real', 'synthetic')
CHAINS = ('historical_cached', 'portable_cached', 'portable_live')
PROFILES = ('full_live', 'training_release')
QUALITY_FAILURES = {'failed_cached_input_numerical_parity', 'failed_cached_input_metric_parity'}
QUALITY_LIMITS = {'per_image_low_mean_abs_m': 1e-6, 'per_image_native_mean_abs_m': 1e-6,
                  'per_image_low_over_1e6_fraction': 1e-4, 'total_low_over_1e6_fraction': 1e-6,
                  'partition_mae_abs_change_mm': .01, 'refined_mask_disagree_pixels': 0}
QUALITY_AUTHORIZATION = {'basis': 'latest_user_training_release_approximate_effect_scope',
                         'numeric_limits_explicitly_specified_by_user': False,
                         'operational_limits_defined_by': 'root', 'post_failure_policy': True,
                         'automatically_relaxed': False}
DEPTH_FIELDS = {'v2/0/output', 'lap/fixed_bands/0/output/0', 'lap/applied/0/output/depth', 'final_depth'}
DIAGNOSTIC_FIELDS = {'lap/fixed_bands/0/output/1', 'lap/context/0/output', 'lap/mean_coeff/0/output',
    'lap/gates/gate_alpha', 'lap/gates/gate_delta', 'lap/gates/gate_regularization',
    'lap/applied/0/output/gate_delta', 'lap/applied/0/output/global_log_mean', 'lap/applied/0/output/log_correction'}
EXACT_FIELDS = {'dino/0/output', 'far/0/output', 'moge/0/output', 'moge/1/output',
    'teacher/0/output/confidence', 'teacher/0/output/normal_cv', 'tom/0/logits', 'tom/0/output', 'lap/gates/global_alpha'}
EXACT_FIELDS |= {'r5/0/output/' + key for key in ('anchor_disagreement', 'bounded_delta', 'delta_log', 'depth',
    'o2_delta_log', 'o2_depth', 'o4_delta_log', 'o4_depth', 'outline_logits', 'patch_roi', 'raw_delta', 'refine_raw',
    'teacher_confidence', 'teacher_edge', 'teacher_normal')}
EXACT_FIELDS |= {'v2_inputs/0/output/' + key for key in ('content', 'context_content', 'context_descriptor',
    'context_far', 'context_probability', 'context_r5', 'context_rgb', 'context_teacher_confidence',
    'context_teacher_normal', 'crop_top_left', 'far_depth', 'native_hw', 'padding_tblr', 'probability',
    'r5_depth', 'rgb', 'teacher_confidence', 'teacher_normal')}
LABELS = {'historical_cached': '历史封存 P／原生产模型',
          'portable_cached': '同一历史 P／便携 Stage2',
          'portable_live': '新 RGB 实算 E08／便携 Stage2',
          'stage1_native': 'E08 原生', 'stage1_resized': 'E08 缩放往返',
          'stage1_lidf': 'E08→LIDF-SPVNAS', 'combined': '合并', 'real': '真实',
          'synthetic': '合成', 'all': '全图', 'transparent': '透明', 'background': '背景'}
DISPLAY_METRICS = ('mae_mm', 'rmse_mm', 'abs_rel', 'normal_mean_deg', 'edge_f1')
IDENTITY = ('cache_id', 'sample_id', 'rgb_sha256', 'native_hw', 'domain',
            'official_split', 'benchmark_group')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def reference(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b, label, atol=1e-10):
    require(finite(a) and finite(b) and math.isclose(a, b, rel_tol=0., abs_tol=atol), label)


class Evidence:
    def __init__(self):
        self.files = {}

    def add(self, path):
        value = reference(path)
        previous = self.files.setdefault(value['path'], value['sha256'])
        require(previous == value['sha256'], 'Input changed during report: ' + value['path'])
        return value

    def check(self, value):
        require(self.add(value['path']) == value, 'Artifact SHA/path differs: ' + str(value['path']))
        return Path(value['path'])

    def json(self, path):
        self.add(path)
        return read(path)


def verification_profile(args):
    profile = getattr(args, 'verification_profile', 'full_live')
    require(profile in PROFILES, 'Unknown verification profile')
    require(getattr(args, 'cached_quality_acceptance', None) is None or profile == 'training_release',
            'Cached quality acceptance is only available for explicit training_release')
    if profile == 'full_live':
        require(getattr(args, 'live_validation', None) is not None, 'full_live requires --live-validation')
    else:
        require(getattr(args, 'live_validation', None) is None,
                'training_release reports exactly two full-roster chains; use full_live for a third chain')
        require(all(getattr(args, k, None) is not None for k in
                    ('stage1_acceptance', 'single_training_acceptance', 'ddp_training_acceptance')),
                'training_release requires Stage1 three-case/relocation and real single/DDP training acceptance')
    return profile


def verify_project_sources(mapping, evidence, base):
    require(isinstance(mapping, dict) and bool(mapping), 'Missing source identity')
    for name, digest in mapping.items():
        path = (base / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(base.resolve())
                and evidence.add(path)['sha256'] == digest, 'Acceptance source differs: ' + name)


def validate_policy_comparison(record, policy, policy_ref):
    require(record.get('status') == 'passed_user_policy' and record.get('policy') == policy_ref
            and record.get('numeric_tolerance_user_authorized') is True
            and record.get('numeric_tolerance_automatically_relaxed') is False
            and record.get('nonmetric_intermediate_equivalence_claimed') is False
            and record.get('depth_limits') == policy['depth_limits'], 'Stage1 comparison policy differs')
    fields = record['fields']
    require(set(fields) == EXACT_FIELDS | DEPTH_FIELDS | DIAGNOSTIC_FIELDS, 'Unknown/missing Stage1 trace field')
    for name, row in fields.items():
        require(row.get('accepted') is True and row.get('finite') is True
                and finite(row['max_abs']) and finite(row['mean_abs'])
                and row['max_abs'] >= 0 and row['mean_abs'] >= 0,
                'Stage1 field failed finite/schema acceptance')
        if name in EXACT_FIELDS:
            require(row.get('acceptance_kind') == 'exact' and row.get('exact_equal') is True
                    and row.get('exact_equivalence_required') is True
                    and row['max_abs'] == 0 and row['mean_abs'] == 0, 'Upstream Stage1 field is not exact')
        elif name in DEPTH_FIELDS:
            require(row.get('acceptance_kind') == 'metric_depth_max_and_mean'
                    and row.get('exact_equivalence_required') is False and row.get('depth_positive') is True
                    and row['max_abs'] <= 5e-5 and row['mean_abs'] <= 1e-6, 'Stage1 depth exceeds fixed user limits')
        else:
            require(row.get('acceptance_kind') == 'diagnostic_finite_schema_only'
                    and row.get('exact_equivalence_required') is False, 'Nonmetric field falsely claimed exact')


def training_release_stage1(path, config, evidence):
    marker = evidence.json(path)
    require(marker.get('schema') == 'portable_stage1_validation_complete_v2'
            and marker.get('status') == 'passed_user_policy'
            and marker.get('numeric_tolerance_user_authorized') is True
            and marker.get('numeric_tolerance_automatically_relaxed') is False
            and marker.get('zero_tolerance_equivalence_claimed') is False
            and marker.get('nonmetric_intermediate_equivalence_claimed') is False
            and marker.get('full_1226_regression_executed') is False,
            'Stage1 small-sample acceptance is incomplete or overclaims full regression')
    for value in marker['bindings'].values():
        evidence.check(value)
    bindings = marker['bindings']
    require(bindings['asset_manifest']['sha256'] == config['stage1']['assets_manifest_sha256']
            and bindings['source_inventory'] == evidence.add(PROJECT / 'provenance/stage1_runtime_sources.json'),
            'Stage1 acceptance assets/source differ from release')
    verify_project_sources(read(bindings['source_inventory']['path'])['sha256'], evidence, PROJECT)
    cases = lines(bindings['trace_cases']['path'])
    require(len(cases) == 3 and len({r['sample_id'] for r in cases}) == 3,
            'Exactly three distinct live Stage1 trace cases are required')
    policy_ref = bindings['acceptance_policy']
    policy = read(policy_ref['path'])
    require(policy.get('schema') == 'portable_stage1_user_numeric_policy_v1'
            and policy.get('numeric_tolerance_user_authorized') is True
            and policy.get('numeric_tolerance_automatically_relaxed') is False
            and policy.get('depth_limits') == {'max_abs_m': 5e-5, 'mean_abs_m': 1e-6, 'relative_tolerance': 0.}
            and policy.get('unknown_fields') == 'reject' and policy.get('missing_fields') == 'reject'
            and policy.get('nonmetric_fields_are_not_claimed_exact') is True, 'Stage1 user policy changed')
    for key, expected in (('exact_fields', EXACT_FIELDS), ('metric_depth_fields', DEPTH_FIELDS),
                          ('diagnostic_only_fields', DIAGNOSTIC_FIELDS)):
        require(set(policy[key]) == expected and len(policy[key]) == len(expected), 'Stage1 field partition changed')
    require(policy['frozen_source_inventory'] == bindings['source_inventory'], 'Policy source inventory differs')
    require(bool(policy['evidence']), 'Missing operator repeatability evidence')
    for value in policy['evidence'].values():
        evidence.check(value)
    strict = read(evidence.check(marker['original_zero_tolerance_failure']))
    require(marker['original_zero_tolerance_failure'] == policy['original_zero_tolerance_failure']
            and strict.get('status') == 'failed_numeric_comparison'
            and strict.get('declared_tolerance') == {'atol': 0., 'rtol': 0.}, 'Original zero-tolerance failure was not retained')
    traces = {value['path']: read(evidence.check(value)) for value in marker['trace_receipts']}
    require(len(traces) == 7 and len(marker['policy_comparisons']) == 4,
            'Three legacy/portable cases plus one relocated trace are required')
    comparisons = []
    for value in marker['policy_comparisons']:
        record = read(evidence.check(value)); validate_policy_comparison(record, policy, policy_ref)
        left, right = record['reference'], record['candidate']
        require(left['path'] in traces and right['path'] in traces, 'Comparison trace not in sealed roster')
        evidence.check(left); evidence.check(right)
        require(traces[left['path']]['rgb_sha256'] == traces[right['path']]['rgb_sha256'], 'Stage1 compared different RGB')
        zero = read(evidence.check(record['zero_tolerance_comparison']))
        require(zero['status'] == record['zero_tolerance_status'] and zero['reference'] == left and zero['candidate'] == right
                and zero['declared_tolerance'] == {'atol': 0., 'rtol': 0.}, 'Zero-tolerance diagnostic changed')
        comparisons.append((traces[left['path']], traces[right['path']]))
    pairs = [(a['rgb_sha256'], a['mode'], b['mode']) for a, b in comparisons]
    require(sorted(pairs[:3]) == sorted((r['rgb_sha256'], 'legacy', 'portable') for r in cases)
            and pairs[3] == (cases[0]['rgb_sha256'], 'portable', 'portable'), 'Missing legacy or relocated comparison')
    for name in ('ready_marker', 'fresh_environment_probe', 'relocation_preparation', 'executions'):
        evidence.check(marker[name])
    require(read(marker['fresh_environment_probe']['path'])['status'] == 'passed', 'Fresh Stage1 environment probe failed')
    for execution in read(marker['executions']['path']):
        require(execution['returncode'] == 0 and execution['execution_completed_normally'] is True,
                'Stage1 acceptance contains an unsuccessful execution')
        evidence.check(execution['log'])
    for trace in traces.values():
        require(trace['status'] == 'trace_complete' and trace['metadata']['force_live'] is True
                and trace['metadata']['ground_truth_access'] is False, 'Trace was not actual RGB-only inference')
        if trace['mode'] == 'portable':
            metadata = trace['metadata']; guard = metadata['file_access_guard']
            require(metadata['source_manifest_sha256'] == bindings['source_inventory']['sha256']
                    and metadata['assets_sha256'] == bindings['asset_manifest']['sha256']
                    and guard['ground_truth_read'] is False and guard['legacy_cache_read'] is False
                    and guard['strict_read_allowlist'] is True and not guard['denied_paths'],
                    'Portable trace source/asset/input isolation differs')
    for value in marker['reused_evidence']:
        reuse = read(evidence.check(value))
        require(reuse.get('execution_mode') == 'reused_verified' and reuse.get('not_fresh_gpu_execution') is True
                and reuse.get('same_source_and_weights_and_rgb') is True
                and reuse.get('all_trace_arrays_sha256_verified') is True, 'Unverified reused Stage1 trace')
        for key in ('original_attempt_status', 'original_executions', 'original_zero_tolerance_failure'):
            evidence.check(reuse[key])
    predictions = evidence.check(marker['fresh_predictions'])
    seal = read(evidence.check(marker['fresh_predictions_seal']))
    require(seal.get('status') == 'complete' and seal.get('images') == 12 and seal.get('unique_sample_ids') == 12
            and seal.get('ground_truth_access') is False and seal.get('expert_cache_read') is False
            and seal.get('all_arrays_sha256_verified') is True and seal['manifest_sha256'] == sha(predictions),
            'Real twelve-image RGB-only Stage1 batch is incomplete')
    return marker


def training_acceptance(path, world, stage1, evidence):
    marker = evidence.json(path); root = Path(path).parent; run = root / 'training'
    alphas = [1., 1., 1., 1., .5, 0., 0., 0., 0.]
    require(marker.get('status') == 'passed' and marker.get('diagnostic_smoke') is True
            and marker.get('formal_training_claimed') is False and marker.get('epochs') == 3
            and marker.get('updates') == 9 and marker.get('world_size') == world
            and marker.get('input_alphas') == alphas, 'Real single/DDP nine-update training did not complete')
    require(evidence.add(root / 'smoke.config.json')['sha256'] == marker['config_sha256']
            and evidence.add(run / 'smoke.complete.json')['sha256'] == marker['training_complete_sha256'],
            'Training acceptance bytes changed')
    cfg = read(root / 'smoke.config.json'); complete = read(run / 'smoke.complete.json')
    contract = evidence.json(run / 'contract.json')
    require(contract['config'] == cfg and contract['world_size'] == world
            and complete['contract_sha256'] == sha(run / 'contract.json')
            and complete['smoke'] is True and complete['epochs'] == 3 and complete['updates'] == 9
            and not (run / 'complete.json').exists(), 'Smoke cannot substitute for completed formal training')
    verify_project_sources(contract['sources'], evidence, PROJECT / 'src')
    prepared = evidence.json(root / 'prepared/stage1/complete.json')
    require(contract['stage1_complete_sha256'] == sha(root / 'prepared/stage1/complete.json')
            and prepared['source_predictions_sha256'] == stage1['fresh_predictions']['sha256'],
            'Training smoke used a different Stage1 prediction batch')
    require(marker.get('source_predictions') == stage1['fresh_predictions']
            and marker.get('training_contract') == evidence.add(run / 'contract.json'),
            'Training source prediction/contract receipt differs')
    interrupted = evidence.json(run / 'smoke.interrupted.json')
    require(interrupted['status'] == 'controlled_epoch_stop' and interrupted['epoch'] == 1
            and interrupted['updates'] == 3, 'Missing genuine committed-epoch interruption')
    epochs = lines(evidence.add(run / 'epochs.jsonl')['path'])
    require([r['epoch'] for r in epochs] == [1, 2, 3] and [r['updates'] for r in epochs] == [3, 6, 9],
            'Missing real training epochs')
    for index, row in enumerate(epochs):
        pure, matched = row['validation'], row['validation_curriculum']
        require(row['smoke'] is True and all(finite(v) for v in row['losses'].values())
                and pure['input_alpha'] == 0 and pure['gt_or_mask_used_as_input'] is False
                and pure['images'] == cfg['counts']['val'] and pure['images'] > 0
                and matched['input_alpha'] == alphas[index * 3 + 2]
                and matched['gt_or_mask_used_as_input'] is (matched['input_alpha'] > 0),
                'Training or dual-validation condition differs')
        require(all(finite(v[k]) for v in (pure, matched) for k in ('mae_mm', 'rmse_mm', 'abs_rel')),
                'Nonfinite validation metric')
    require([r['rank'] for r in marker['ranks']] == list(range(world)), 'Missing/duplicate restored rank')
    for rank in marker['ranks']:
        index = rank['rank']; resume_path = run / f'resume_verification_rank{index}.json'
        changes_path = run / f'optimizer_changes_rank{index}.jsonl'
        require(evidence.add(resume_path)['sha256'] == rank['resume_sha256']
                and evidence.add(changes_path)['sha256'] == rank['optimizer_changes_sha256'], 'Rank restoration/update bytes changed')
        resume, changes = read(resume_path), lines(changes_path)
        require(resume['status'] == 'passed' and resume['updates'] == 3 and resume['rank'] == index
                and all(resume[k] is True for k in ('model_exact', 'optimizer_exact', 'scheduler_exact', 'rng_exact')),
                'Model/optimizer/scheduler/RNG restoration was not exact')
        require([r['updates'] for r in changes] == list(range(1, 10)) and [r['alpha'] for r in changes] == alphas
                and all(r['loss_finite'] is True and r['gradients_finite'] is True for r in changes),
                'Real optimizer updates/alpha/finite gradients incomplete')
        for group in ('lidf', 'spvnas', 'adapters'):
            values = [r['parameter_group_max_abs_change'][group] for r in changes]
            require(all(finite(v) and v >= 0 for v in values) and max(values) > 0
                    and rank['group_max_abs_change'][group] == max(values), 'A trainable group did not update')
    require(marker.get('runner_sha256') == evidence.add(PROJECT / 'scripts/run_stage2_smoke.py')['sha256'],
            'Training acceptance runner source differs')
    require(len(marker.get('executions', [])) == 2, 'Fresh and resumed process completion evidence is required')
    for index, execution in enumerate(marker['executions']):
        require(execution['exit_code'] == 0,
                'Training process did not exit successfully')
        evidence.check(execution['log'])
        command = execution['command']
        require('transparent_depth.train_stage2' in command and '--output' in command
                and Path(command[command.index('--output') + 1]).resolve() == run.resolve(),
                'Training execution belongs to another run')
        if index == 0:
            require('--stop-after-epoch' in command and command[command.index('--stop-after-epoch') + 1] == '1'
                    and '--resume' not in command, 'First process did not perform controlled interruption')
        else:
            require('--resume' in command and '--stop-after-epoch' not in command, 'Second process did not resume')
    return marker


def validate_selection(selection, verified, resolved, selection_ref, verification_ref):
    require(selection.get('status') == 'complete' and selection.get('completed_epochs') == 80,
            'All 80 completed checkpoints are required')
    ranked = selection['ranking']
    require(len(ranked) == 80 and {r['epoch'] for r in ranked} == set(range(1, 81)),
            'Selection contains missing/duplicate epochs')
    require(all(r['images'] == 1226 and r['smoke'] is False and finite(r['transparent_mae_mm'])
                and r['transparent_mae_mm'] >= 0 for r in ranked), 'Incomplete/nonfinite ranking entry')
    require(ranked == sorted(ranked, key=lambda r: (r['transparent_mae_mm'], r['epoch'])),
            'Ranking differs from transparent image-macro MAE / earlier-epoch tie rule')
    policy = selection['selection_policy']
    require(policy.get('epochs') == list(range(1, 81)) and policy.get('selection_images') == 1226
            and policy.get('primary_metric') == 'full1226/combined/transparent/image_macro/mae_mm'
            and policy.get('tie_break') == 'smaller_epoch'
            and policy.get('empty_region') == 'null_per_image_excluded_from_macro'
            and policy.get('uses_test_labels_for_checkpoint_selection') is True
            and policy.get('independent_test_performance_claimed') is False
            and policy.get('gt_or_mask_used_as_prediction_input') is False,
            'Selection policy differs or test-informed disclosure is missing')
    require(verified.get('status') == 'passed' and verified.get('epochs') == 80
            and verified.get('predictions_verified') == 80 * 1226
            and verified.get('selection') == selection_ref,
            'All 98,080 predictions and checkpoints need independent completed verification')
    require(resolved.get('status') == 'complete' and resolved.get('original_selection') == selection_ref
            and resolved.get('verification') == verification_ref, 'Final tie-resolution provenance differs')
    winner = next((r for r in ranked if r['epoch'] == resolved.get('selected_epoch')), None)
    require(winner is not None and resolved['selected_checkpoint']['sha256'] == winner['checkpoint_sha256'],
            'Resolved winner differs from an authenticated checkpoint')
    score = resolved['scores'][str(winner['epoch'])]['transparent_mae_mm']
    close(score, winner['transparent_mae_mm'], 'Resolved score exceeds original 2e-5 mm tolerance', 2e-5)
    candidates = [r['epoch'] for r in ranked if r['transparent_mae_mm'] - ranked[0]['transparent_mae_mm'] <= 2e-5]
    require(resolved.get('near_tie_tolerance_mm') == 2e-5
            and resolved.get('candidate_epochs') == candidates
            and set(resolved['scores']) == {str(e) for e in candidates}
            and resolved.get('recomputed_on_one_gpu') is (len(candidates) > 1),
            'Near-tie resolution coverage/protocol differs')
    actual_winner = min(candidates, key=lambda e: (resolved['scores'][str(e)]['transparent_mae_mm'], e))
    require(actual_winner == winner['epoch'], 'Winner does not minimize resolved candidate scores')
    return winner


def validate_roster(rows, expected=None, count=1226, test_count=521):
    require(len(rows) == count and len({r['cache_id'] for r in rows}) == count
            and len({r['sample_id'] for r in rows}) == count, 'Incomplete/duplicate metric roster')
    require(sum(r['official_split'].endswith('test') for r in rows) == test_count,
            'Original test subset is incomplete')
    require(all(r['domain'] in ('real', 'synthetic') for r in rows), 'Unknown evaluation domain')
    if expected is not None:
        wanted = {r['cache_id']: r for r in expected}
        require(set(wanted) == {r['cache_id'] for r in rows}, 'Evaluation rosters differ')
        require(all(all(r.get(k) == wanted[r['cache_id']].get(k) for k in IDENTITY) for r in rows),
                'Evaluation image/RGB/domain identity differs')


def subset_rows(rows):
    result = {'full1226': rows, 'test521': [r for r in rows if r['official_split'].endswith('test')]}
    result.update({g: [r for r in rows if r['benchmark_group'] == g]
                   for g in sorted({r['benchmark_group'] for r in rows})})
    return result


def verify_summary(metrics, paired, rows):
    """Check reported macro/support against per-image evidence; no GT is read."""
    subsets = subset_rows(rows)
    require(set(metrics) == set(subsets), 'Missing/extra metric subset')
    for subset, part in subsets.items():
        present_domains = {'combined'} | {r['domain'] for r in part}
        require(set(metrics[subset]) == present_domains, 'Missing/extra metric domain')
        for domain, methods in metrics[subset].items():
            selected = [r for r in part if domain == 'combined' or r['domain'] == domain]
            require(set(methods) == set(METHODS), 'All three methods must be retained')
            for method, regions in methods.items():
                require(set(regions) == set(REGIONS), 'Missing metric region')
                for region, item in regions.items():
                    require(item['images'] == len(selected), 'Metric image count differs')
                    values = [r['metrics'][method][region] for r in selected]
                    require(item['pixels'] == sum(v['pixels'] for v in values), 'Metric pixel support differs')
                    require(all(k in item['image_macro'] for k in DISPLAY_METRICS), 'Missing depth/geometry metric')
                    require(all(v is None or finite(v) for v in item['pixel_pooled'].values()), 'Nonfinite pooled metric')
                    for metric, actual in item['image_macro'].items():
                        entries = [v[metric] for v in values if v[metric] is not None]
                        require(all(finite(v) for v in entries), 'Nonfinite per-image metric')
                        require(item['metric_image_counts'][metric] == len(entries), 'Metric-specific image support differs')
                        if entries:
                            close(actual, math.fsum(entries) / len(entries), 'Macro metric differs from per-image data')
                        else:
                            require(actual is None, 'Empty support must remain null')
            for row in selected:
                for region in REGIONS:
                    require(len({row['metrics'][m][region]['pixels'] for m in METHODS}) == 1,
                            'Methods use different depth GT supports')
    for subset in ('full1226', 'test521'):
        require(set(paired[subset]) == {f'{d}/{r}/{m}' for d in DOMAINS for r in REGIONS
                                      for m in ('mae_mm', 'abs_rel')}, 'Incomplete paired CI table')
        for key, result in paired[subset].items():
            domain, region, metric = key.split('/')
            selected = [r for r in subsets[subset] if domain == 'combined' or r['domain'] == domain]
            differences = []
            for row in selected:
                a, b = (row['metrics'][m][region][metric] for m in ('stage1_lidf', 'stage1_native'))
                require((a is None) == (b is None), 'Paired support differs')
                if a is not None:
                    differences.append(a - b)
            require(result['paired_images'] == len(differences), 'Paired image count differs')
            if differences:
                close(result['difference_lidf_minus_stage1'], math.fsum(differences) / len(differences),
                      'Paired mean differs from per-image metrics')
                interval = result['image_bootstrap_95pct']
                require(len(interval) == 2 and all(finite(x) for x in interval) and interval[0] <= interval[1],
                        'Invalid paired confidence interval')
            else:
                require(result['difference_lidf_minus_stage1'] is None and result['image_bootstrap_95pct'] is None,
                        'Empty paired support must remain null')


def report_payload(root, marker, evidence, historical=False):
    names = ('metrics.json', 'paired_differences.json', 'per_image.jsonl')
    paths = {}
    for name in names:
        if historical:
            path = root / 'reports' / name
            require(evidence.add(path)['sha256'] == marker['reports_sha256'][name], 'Historical report changed')
        else:
            path = evidence.check(marker['reports'][name])
            require(path.resolve() == (root / 'reports' / name).resolve(), 'Validation report path differs')
        paths[name] = path
    metrics, paired, rows = read(paths[names[0]]), read(paths[names[1]]), lines(paths[names[2]])
    validate_roster(rows)
    verify_summary(metrics, paired, rows)
    close(metrics['full1226']['combined']['stage1_lidf']['transparent']['image_macro']['mae_mm'],
          marker['transparent_mae_mm'], 'Report headline differs from metric table')
    return {'metrics': metrics, 'paired': paired, 'rows': rows, 'marker': marker}


def portable_payload(root, mode, winner, resolved_ref, metadata_ref, weights_ref, evidence, *, quality_measurement=False):
    marker = evidence.json(root / 'evaluation.complete.json')
    allowed_status = marker.get('status') in QUALITY_FAILURES if quality_measurement else marker.get('status') == 'complete'
    require(not quality_measurement or mode == 'cached', 'Quality measurement requires cached inputs')
    require(allowed_status and marker.get('mode') == mode
            and marker.get('images') == 1226 and marker.get('test_images') == 521
            and marker.get('selected_epoch') == winner['epoch']
            and marker.get('selected_checkpoint_sha256') == winner['checkpoint_sha256']
            and marker.get('model_inputs_included_gt_or_mask') is False
            and marker.get('test_informed_selection') is True
            and marker.get('independent_test_performance_claimed') is False,
            'Portable validation is incomplete, impure, or from another checkpoint')
    if mode == 'cached' and not quality_measurement:
        require(marker.get('cached_input_parity_passed') is True and marker.get('cached_transparent_mae_agrees') is True
                and marker.get('cached_transparent_mae_tolerance_mm') == 2e-5,
                'Cached-input parity did not pass its original thresholds')
        require(abs(marker['transparent_mae_change_vs_historical_mm']) <= 2e-5, 'Cached metric parity failed')
    evidence.check(marker['predictions_seal'])
    require(Path(marker['predictions_seal']['path']).resolve() == (root / 'predictions.complete.json').resolve(),
            'Portable prediction seal path differs')
    seal = read(marker['predictions_seal']['path'])
    contract_path = root / 'contract.json'
    contract = evidence.json(contract_path)
    for name, digest in contract['sources'].items():
        path = (PROJECT / name).resolve()
        require(not Path(name).is_absolute() and path.is_relative_to(PROJECT)
                and evidence.add(path)['sha256'] == digest, 'Portable validation source differs from this release')
    require(seal.get('status') == 'complete' and seal.get('images') == 1226 and seal.get('test_images') == 521
            and seal.get('all_arrays_sha256_verified') is True and seal.get('labels_used_as_prediction_input') is False
            and seal.get('contract_sha256') == sha(contract_path) and seal.get('mode') == mode
            and seal.get('selected_checkpoint_sha256') == winner['checkpoint_sha256']
            and contract['selection']['resolved_selection'] == resolved_ref
            and contract['selection']['checkpoint_sha256'] == winner['checkpoint_sha256']
            and contract['dependencies']['bundle_metadata'] == metadata_ref
            and contract['dependencies']['bundle_weights'] == weights_ref,
            'Portable model/input contract differs from final release')
    close(marker['historical_cached_winner_transparent_mae_mm'],
          contract['selection']['historical_transparent_mae_mm'], 'Historical baseline score binding differs')
    close(marker['transparent_mae_change_vs_historical_mm'],
          marker['transparent_mae_mm'] - marker['historical_cached_winner_transparent_mae_mm'],
          'Reported cached/live score change differs')
    if mode == 'cached':
        require((quality_measurement or seal.get('cached_input_parity_passed') is True)
                and contract.get('cached_parity_abs_tolerance_m') == 1e-6
                and contract.get('cached_parity_rtol') == 0., 'Cached array parity limits changed')
        if quality_measurement:
            require(marker.get('cached_input_parity_passed') == seal.get('cached_input_parity_passed')
                    and type(seal.get('cached_input_parity_passed')) is bool
                    and marker.get('cached_transparent_mae_tolerance_mm') == 2e-5,
                    'Original strict comparison facts changed')
            metric_pass = abs(marker['transparent_mae_change_vs_historical_mm']) <= 2e-5
            expected_status = ('failed_cached_input_metric_parity' if not metric_pass
                               else 'failed_cached_input_numerical_parity')
            require(marker.get('cached_transparent_mae_agrees') is metric_pass
                    and marker['status'] == expected_status
                    and (not metric_pass or seal['cached_input_parity_passed'] is False),
                    'Original strict failure is inconsistent with its unchanged thresholds')
    require(set(seal['workers']) == {str(i) for i in range(contract['world_size'])}, 'Missing prediction worker')
    require(set(marker['metric_shards']) == set(seal['workers']), 'Missing metric worker')
    for value in list(seal['workers'].values()) + list(seal['original_producer_attempts'].values()):
        evidence.check(value)
    for rank, value in seal['workers'].items():
        worker = read(value['path'])
        require(worker.get('status') == 'complete' and worker['rank'] == int(rank)
                and worker['contract_sha256'] == seal['contract_sha256']
                and worker.get('labels_used_as_prediction_input') is False
                and worker.get('compute_loss') is False and worker.get('input_alpha') == 0.
                and worker['file_access_guard']['ground_truth_read'] is False
                and not worker['file_access_guard']['denied_paths'], 'Prediction process isolation audit failed')
    for value in marker['metric_shards'].values():
        shard = read(evidence.check(value))
        require(shard['status'] == 'complete' and shard['model_executed'] is False
                and shard['labelled_process'] is True and shard['predictions_seal'] == marker['predictions_seal'],
                'Independent metric process evidence is incomplete')
        evidence.check(shard['records'])
    for value in contract['dependencies'].values():
        evidence.check(value)
    require(sha(evidence.check(marker['labels'])) == contract['selection']['label_manifest_sha256'],
            'Portable labels differ from authenticated selection labels')
    predictions = lines(evidence.check(seal['predictions']))
    validate_roster(predictions)
    result = report_payload(root, marker, evidence)
    validate_roster(result['rows'], predictions)
    by_id = {r['cache_id']: r for r in predictions}
    require(all(r['prediction_sha256'] == by_id[r['cache_id']]['prediction']['sha256'] for r in result['rows']),
            'Metrics reference different prediction bytes')
    result['contract'] = contract
    result['predictions'] = predictions
    return result


def cached_quality_statistics(predictions, historical_rows, portable_rows):
    """Small sealed JSON only: no model, prediction arrays, or GT arrays read."""
    validate_roster(predictions)
    validate_roster(historical_rows, predictions)
    validate_roster(portable_rows, predictions)
    old = {r['cache_id']: r for r in historical_rows}
    new = {r['cache_id']: r for r in portable_rows}
    per_image, total_over, total_pixels, mask_errors = [], 0, 0, 0
    for row in predictions:
        identity = {key: row[key] for key in IDENTITY}
        comparison = row['parity_against_cached_winner']
        low, native = comparison['low'], comparison['native_vs_reference_cpu_resize']
        for diff in (low, native):
            require(diff.get('abs_tolerance_m') == 1e-6 and diff.get('rtol') == 0.
                    and all(finite(diff[k]) and diff[k] >= 0 for k in ('max_abs_m', 'mae_m', 'rmse_m'))
                    and 0 <= diff['bitwise_equal_fraction'] <= 1
                    and type(diff['pixels_over_tolerance']) is int and diff['pixels_over_tolerance'] >= 0
                    and diff['mae_m'] <= diff['max_abs_m'], 'Invalid sealed array-difference statistics')
        low_pixels, native_pixels = 240 * 320, math.prod(row['native_hw'])
        require(low['pixels_over_tolerance'] <= low_pixels and native['pixels_over_tolerance'] <= native_pixels,
                'Pixel difference counts exceed array size')
        mask = comparison['refined_mask_disagree_pixels']
        require(type(mask) is int and 0 <= mask <= low_pixels
                and comparison['passed'] is (low['pixels_over_tolerance'] == 0 and mask == 0),
                'Strict per-image comparison facts are inconsistent')
        total_over += low['pixels_over_tolerance']; total_pixels += low_pixels; mask_errors += mask
        a, b = old[row['cache_id']], new[row['cache_id']]
        verify_cross_chain_labels(a, b)
        require(b['prediction_sha256'] == row['prediction']['sha256'], 'Quality metrics refer to other prediction bytes')
        for region in REGIONS:
            for method in METHODS:
                aa, bb = a['metrics'][method][region], b['metrics'][method][region]
                require(aa['pixels'] == bb['pixels'] and (aa['mae_mm'] is None) == (bb['mae_mm'] is None),
                        'Quality comparison has different GT support')
        per_image.append({**identity, 'low': low, 'native_vs_reference_cpu_resize': native,
                          'low_pixels': low_pixels, 'native_pixels': native_pixels,
                          'low_over_1e6_fraction': low['pixels_over_tolerance'] / low_pixels,
                          'refined_mask_disagree_pixels': mask})
    partitions = {}
    for subset in ('full1226', 'test521'):
        for domain in DOMAINS:
            rows = [r for r in portable_rows if (subset == 'full1226' or r['official_split'].endswith('test'))
                    and (domain == 'combined' or r['domain'] == domain)]
            for region in REGIONS:
                pairs = [(old[r['cache_id']]['metrics']['stage1_lidf'][region]['mae_mm'],
                          r['metrics']['stage1_lidf'][region]['mae_mm']) for r in rows]
                pairs = [(a, b) for a, b in pairs if a is not None]
                require(all(finite(a) and finite(b) for a, b in pairs), 'Nonfinite paired quality metric')
                before = math.fsum(a for a, _ in pairs) / len(pairs) if pairs else None
                after = math.fsum(b for _, b in pairs) / len(pairs) if pairs else None
                deltas = [b - a for a, b in pairs]
                partitions[f'{subset}/{domain}/{region}'] = {
                    'images': len(rows), 'metric_images': len(pairs),
                    'pixels': sum(r['metrics']['stage1_lidf'][region]['pixels'] for r in rows),
                    'historical_mae_mm': before, 'portable_mae_mm': after,
                    'signed_mae_change_mm': None if not pairs else math.fsum(deltas) / len(deltas),
                    'mean_abs_per_image_mae_change_mm': None if not pairs else math.fsum(map(abs, deltas)) / len(deltas),
                    'max_abs_per_image_mae_change_mm': None if not pairs else max(map(abs, deltas))}
    summary = {'images': len(per_image), 'test_images': sum(r['official_split'].endswith('test') for r in per_image),
               'low_pixels': total_pixels, 'low_pixels_over_1e6': total_over,
               'low_images_over_1e6': sum(r['low']['pixels_over_tolerance'] > 0 for r in per_image),
               'total_low_over_1e6_fraction': total_over / total_pixels,
               'max_per_image_low_over_1e6_fraction': max(r['low_over_1e6_fraction'] for r in per_image),
               'max_per_image_low_mean_abs_m': max(r['low']['mae_m'] for r in per_image),
               'max_per_image_native_mean_abs_m': max(r['native_vs_reference_cpu_resize']['mae_m'] for r in per_image),
               'max_low_abs_m': max(r['low']['max_abs_m'] for r in per_image),
               'max_native_abs_m': max(r['native_vs_reference_cpu_resize']['max_abs_m'] for r in per_image),
               'refined_mask_disagree_pixels': mask_errors,
               'native_comparison_definition': 'portable_native_vs_cpu_resize_of_historical_low',
               'single_pixel_max_is_disclosed_without_relaxing_original_strict_gate': True}
    return {'summary': summary, 'partitions': partitions, 'per_image': per_image}


def quality_limit_checks(statistics):
    s = statistics['summary']; limits = QUALITY_LIMITS
    return {'per_image_low_mean': s['max_per_image_low_mean_abs_m'] <= limits['per_image_low_mean_abs_m'],
            'per_image_native_mean': s['max_per_image_native_mean_abs_m'] <= limits['per_image_native_mean_abs_m'],
            'per_image_low_fraction': s['max_per_image_low_over_1e6_fraction'] <= limits['per_image_low_over_1e6_fraction'],
            'total_low_fraction': s['total_low_over_1e6_fraction'] <= limits['total_low_over_1e6_fraction'],
            'mask_exact': s['refined_mask_disagree_pixels'] == 0,
            'all_partition_mae': all(r['signed_mae_change_mm'] is None or
                abs(r['signed_mae_change_mm']) <= limits['partition_mae_abs_change_mm']
                for r in statistics['partitions'].values())}


def assess_cached_quality(cached_root, historical_root, evidence=None):
    """Return a proposed assessment, never an acceptance or release promotion."""
    cached_root, historical_root = Path(cached_root), Path(historical_root)
    evidence = evidence if evidence is not None else Evidence()
    contract_ref = evidence.add(cached_root / 'contract.json'); contract = read(contract_ref['path'])
    selection = contract['selection']
    selected = read(evidence.check(selection['selection']))
    verified = read(evidence.check(selection['verification']))
    resolved = read(evidence.check(selection['resolved_selection']))
    winner = validate_selection(selected, verified, resolved, selection['selection'], selection['verification'])
    historical_ref = evidence.add(historical_root / 'evaluation.complete.json'); historical = read(historical_ref['path'])
    require(historical.get('status') == 'complete' and historical.get('images') == 1226
            and historical.get('test_images') == 521 and historical.get('selected_epoch') == winner['epoch']
            and historical['selected_checkpoint'] == resolved['selected_checkpoint']
            and historical['resolved_selection'] == selection['resolved_selection']
            and historical['sweep_verification'] == selection['verification']
            and historical.get('ground_truth_used_as_prediction_input') is False,
            'Quality baseline is not the authenticated selected historical evaluation')
    baseline = report_payload(historical_root, historical, evidence, historical=True)
    dependencies = contract['dependencies']
    portable = portable_payload(cached_root, 'cached', winner, selection['resolved_selection'],
        dependencies['bundle_metadata'], dependencies['bundle_weights'], evidence, quality_measurement=True)
    sweep = read(evidence.check(selection['sweep_contract']))
    require(dependencies['input_manifest'] == sweep['input_manifest']
            and dependencies['stage1_seal'] == sweep['stage1_seal']
            and portable['marker']['labels']['sha256'] == sweep['label_manifest']['sha256'],
            'Quality comparison used different inputs or labels from the selected sweep')
    statistics = cached_quality_statistics(portable['predictions'], baseline['rows'], portable['rows'])
    seal = read(cached_root / 'predictions.complete.json')
    failed_ids = [r['cache_id'] for r in portable['predictions'] if not r['parity_against_cached_winner']['passed']]
    require(seal['different_image_ids'] == failed_ids
            and seal['different_from_historical_cached_input_images'] == len(failed_ids)
            and seal['cached_input_parity_passed'] is (not failed_ids), 'Strict seal differs from per-image receipts')
    proposal = {'schema': 'transparent_depth_cached_quality_statistics_v1', 'status': 'proposed',
                'root_reviewed': False, 'model_executed': False, 'gt_arrays_read': False,
                'authorization': QUALITY_AUTHORIZATION, 'limits': QUALITY_LIMITS,
                'strict_numerical_parity_passed': False, 'strict_failure': evidence.add(cached_root / 'evaluation.complete.json'),
                'original_strict_results': {'status': portable['marker']['status'],
                    'cached_input_parity_passed': portable['marker']['cached_input_parity_passed'],
                    'cached_transparent_mae_agrees': portable['marker']['cached_transparent_mae_agrees']},
                'predictions_seal': evidence.add(cached_root / 'predictions.complete.json'),
                'historical_evaluation': historical_ref, 'contract': contract_ref,
                'selected_epoch': winner['epoch'], 'checkpoint_sha256': winner['checkpoint_sha256'],
                'statistics': statistics, 'candidate_limit_checks': quality_limit_checks(statistics),
                'evidence_sha256': dict(evidence.files)}
    return proposal, portable


def validate_cached_quality_acceptance(path, cached_root, historical_root, evidence=None):
    """Shared report/external-CLI gate. Requires explicit root review, no final report."""
    evidence = evidence if evidence is not None else Evidence()
    record = evidence.json(path)
    require(record.get('schema') == 'transparent_depth_cached_quality_acceptance_v1'
            and record.get('status') == 'accepted' and record.get('root_reviewed') is True
            and record.get('verification_profile') == 'training_release'
            and record.get('strict_numerical_parity_passed') is False
            and record.get('authorization') == QUALITY_AUTHORIZATION and record.get('limits') == QUALITY_LIMITS,
            'Missing explicit training_release root-reviewed quality acceptance')
    review = record.get('root_review', {})
    require(review.get('reviewer') == '/root' and review.get('decision') == 'accept_training_release_approximate_quality'
            and finite(review.get('reviewed_unix')) and review['reviewed_unix'] > 0
            and isinstance(review.get('note'), str) and bool(review['note'].strip()),
            'Quality acceptance requires a recorded affirmative root review')
    proposal = read(evidence.check(record['proposed_statistics']))
    actual, portable = assess_cached_quality(cached_root, historical_root)
    require(proposal == actual, 'Quality statistics/inputs changed or were not generated from the complete sealed outputs')
    require(record.get('strict_failure') == actual['strict_failure'], 'Original strict failure is not explicitly retained')
    for source, digest in actual['evidence_sha256'].items():
        require(evidence.add(source)['sha256'] == digest, 'Quality input evidence changed')
    diagnostics = record.get('repeatability_evidence')
    require(isinstance(diagnostics, list) and bool(diagnostics), 'Actual old self-repeatability evidence is required')
    observed_comparison = False
    for diagnostic in diagnostics:
        value = read(evidence.check(diagnostic))
        if (value.get('status') == 'completed_diagnostic_not_acceptance'
                and value.get('original_tolerance_m') == 1e-6
                and (value.get('original_strict_numerical_failure_preserved') is True
                     or value.get('original_failed_acceptance_not_overridden') is True)):
            observed_comparison = True
            if value.get('schema') == 'stage2_hard_argmax_repeatability_observation_v1':
                require(value.get('labels_read') is False and value.get('model_state_exact_to_original_checkpoint') is True
                        and len(value['execution']['runs']) == 2
                        and all(r['returncode'] == 0 for r in value['execution']['runs']),
                        'Repeatability evidence lacks completed pure model execution')
                for mode in ('old', 'portable'):
                    detail = value['runtime_details'][mode]
                    require(detail['checkpoint']['sha256'] == actual['checkpoint_sha256']
                            and detail['contract'] == actual['contract']
                            and detail.get('model_state_exact_to_original_checkpoint') is True
                            and detail.get('labels_read') is False and detail.get('mathematical_operators_changed') is False,
                            'Repeatability observation belongs to a different model/input contract')
    require(observed_comparison, 'Repeatability diagnostic must preserve the original strict failure and tolerance')
    require(review.get('repeatability_evidence_reviewed') is True,
            'Root must review actual repeatability evidence, not assume a cause')
    recoveries = record.get('orchestration_recovery_evidence', [])
    require(isinstance(recoveries, list), 'Orchestration recovery evidence must be explicit artifact references')
    for recovery in recoveries:
        evidence.check(recovery)
    require(all(actual['candidate_limit_checks'].values()), 'A partition/mean/fraction/mask quality limit failed')
    return {'acceptance': record, 'reference': evidence.add(path), 'statistics': actual['statistics'],
            'portable_payload': portable, 'status': 'accepted_approximate_quality',
            'strict_numerical_parity_passed': False}


def training_rows(root, ranked, evidence):
    complete = evidence.json(root / 'complete.json')
    contract = evidence.json(root / 'contract.json')
    require(complete.get('smoke') is False and complete.get('epochs') == 80
            and complete.get('epoch_checkpoints_count') == 80
            and complete.get('contract_sha256') == sha(root / 'contract.json'), 'Training has not formally completed')
    epochs_path, receipts_path = root / 'epochs.jsonl', root / 'epoch_checkpoints.jsonl'
    evidence.add(epochs_path)
    require(evidence.add(receipts_path)['sha256'] == complete['epoch_checkpoints_sha256'], 'Epoch receipt index changed')
    epochs, receipts = lines(epochs_path), lines(receipts_path)
    require([r['epoch'] for r in epochs] == [r['epoch'] for r in receipts] == list(range(1, 81)),
            'All 80 ordered training epochs and receipts are required')
    cfg = contract['config']
    migration = contract['migration']
    parent_cfg = read(evidence.check(migration['parent_config']))
    require(cfg['epochs'] == 80 and cfg['curriculum']['stage_epochs'] == [20, 30, 30]
            and cfg['counts'] == {'train': 45453, 'val': 704, 'evaluation': 1226}
            and cfg['migration'] == {k: v for k, v in migration.items() if k != 'allowed_config_changes'},
            'Training configuration/provenance differs')
    require(migration['parent_completed_epochs'] == 29 and migration['parent_committed_updates'] == 41209
            and migration['final_updates'] == 118525
            and (parent_cfg['expected_world_size'], parent_cfg['batch_per_gpu'], parent_cfg['gradient_accumulation_steps'],
                 parent_cfg['curriculum']['steps_per_epoch']) == (8, 2, 2, 1421)
            and (cfg['expected_world_size'], cfg['batch_per_gpu'], cfg['gradient_accumulation_steps'],
                 cfg['curriculum']['steps_per_epoch']) == (3, 2, 5, 1516),
            'Registered 8-to-3 GPU migration/batch protocol differs')
    parent_epochs = migration['parent_completed_epochs']
    by_epoch = {r['epoch']: r for r in ranked}
    cumulative, previous_alpha, output = 0, 1., []
    best, best_epoch = math.inf, None
    for row, receipt in zip(epochs, receipts):
        epoch = row['epoch']
        current = parent_cfg if epoch <= parent_epochs else cfg
        world, micro, accum = current['expected_world_size'], current['batch_per_gpu'], current['gradient_accumulation_steps']
        steps = current['curriculum']['steps_per_epoch']
        cumulative += steps
        require(row['smoke'] is False and row['updates'] == cumulative == receipt['updates']
                and row['optimizer_steps'] == steps and row['train_unique_expected'] == 45453
                and row['train_presentations'] == current['sampler_protocol']['presentations_per_epoch']
                and receipt['sha256'] == by_epoch[epoch]['checkpoint_sha256'], 'Epoch coverage/update/checkpoint mismatch')
        for key, expected in (('world_size', world), ('batch_per_gpu', micro),
                              ('gradient_accumulation_steps', accum), ('effective_global_batch', world * micro * accum)):
            require(row.get(key, expected) == expected, 'Training batch protocol mismatch')
        require(Path(receipt['path']).resolve() == (root / f'epoch{epoch:03d}.pt').resolve(), 'Epoch checkpoint path mismatch')
        val, matched = row['validation'], row['validation_curriculum']
        require(val['images'] == matched['images'] == 704 and val['input_alpha'] == 0.
                and val['gt_or_mask_used_as_input'] is False, 'Pure-P validation protocol mismatch')
        alpha = row['input_alpha_last']
        require(0 <= alpha <= row['input_alpha_first'] <= previous_alpha + 1e-12
                and 0 <= row['input_alpha_mean'] <= 1, 'Curriculum alpha is invalid or discontinuous')
        require((epoch > 20 or alpha == 1.) and (epoch <= 50 or alpha == 0.), 'Curriculum phase mismatch')
        previous_alpha = alpha
        close(matched['input_alpha'], alpha, 'Matched validation alpha differs')
        require(matched['gt_or_mask_used_as_input'] is (alpha > 0.), 'Label-conditioned validation disclosure missing')
        close(receipt['input_alpha'], alpha, 'Receipt alpha differs')
        for name in ('mae_mm', 'rmse_mm', 'abs_rel'):
            require(all(finite(v[name]) and v[name] >= 0 for v in (val, matched)), 'Invalid validation metric')
            if alpha == 0.:
                close(val[name], matched[name], 'Alpha-zero validation streams must agree')
        close(receipt['validation_abs_rel'], val['abs_rel'], 'Receipt validation differs')
        close(receipt['validation_curriculum_abs_rel'], matched['abs_rel'], 'Receipt matched validation differs')
        eligible = epoch >= 51
        improved = eligible and val['abs_rel'] < best
        require(row['selection_eligible'] is eligible and row['selected'] is improved,
                'Original val704 selection history differs')
        if improved:
            best, best_epoch = val['abs_rel'], epoch
        require(all(finite(v) for v in row['losses'].values()), 'Nonfinite training loss')
        record = {'epoch': epoch, 'checkpoint_sha256': receipt['sha256'], 'updates': cumulative,
                  'optimizer_steps': steps, 'microsteps': row['microsteps'], 'world_size': world,
                  'batch_per_gpu': micro, 'gradient_accumulation_steps': accum,
                  'effective_global_batch': world * micro * accum, 'global_microbatch': world * micro,
                  'train_unique': row['train_unique_expected'], 'train_presentations': row['train_presentations'],
                  'train_seconds': row['train_seconds'], 'alpha_first': row['input_alpha_first'],
                  'alpha_last': alpha, 'alpha_mean': row['input_alpha_mean'],
                  'validation_matched_reuses_pure_p': row['validation_curriculum_reuses_pure_p'],
                  'original_val_selection_eligible': eligible, 'original_val_selected': row['selected']}
        record.update({'train_loss_' + k: v for k, v in row['losses'].items()})
        for prefix, result in (('pure_p', val), ('phase_matched', matched)):
            record.update({prefix + '_' + k: result[k] for k in ('mae_mm', 'rmse_mm', 'abs_rel', 'input_alpha', 'seconds')})
        record.update({'lr_' + k: v for k, v in row['learning_rates'].items()})
        output.append(record)
    require(complete['updates'] == cumulative == migration['final_updates'], 'Final update count differs')
    close(complete['best_score'], best, 'Original val704 best differs')
    live = evidence.json(root.parent / 'migration_live_verification.json')
    require(live.get('status') == 'verified_actual_training' and live.get('world_size') == 3
            and live.get('gpu_ids') == [5, 6, 7] and live['checkpoint'] == migration['parent_checkpoint']
            and live.get('exact_state_restored') is True
            and live.get('identical_eight_rank_training_trajectory_claimed') is False
            and live['progress']['epoch'] == 30 and live['progress']['step'] > 0
            and live['progress']['updates'] == 41209 + live['progress']['step']
            and len(live['restore_receipts']) == 3, 'Actual three-rank migration evidence is incomplete')
    ranks = []
    for value in live['restore_receipts']:
        audit = read(evidence.check(value))
        ranks.append(audit['new_rank'])
        for observation, phase in ((audit, 'after_ddp_before_training'),
                                   (audit['load_observation'], 'after_load_before_ddp')):
            require(observation.get('status') == 'passed' and observation['phase'] == phase
                    and observation['parent_checkpoint'] == migration['parent_checkpoint']
                    and observation['old_rank'] == observation['new_rank'] + 5
                    and observation['physical_gpu'] == observation['old_rank']
                    and observation.get('checkpoint_fields_edited') is False
                    and all(observation['checks'][k]['exact'] is True for k in ('model', 'optimizer', 'scheduler', 'rng')),
                    'Migration model/Adam/scheduler/RNG exact-load audit failed')
    require(sorted(ranks) == [0, 1, 2], 'Migration rank receipts are duplicated or missing')
    preflight = read(evidence.check(migration['parent_preflight']))
    require(preflight.get('status') == 'passed', 'Original implementation preflight was not complete')
    exact = read(evidence.check(preflight['resume_audit']))
    numeric = read(evidence.check(preflight['strict_numeric_reproducibility']))
    require(exact.get('status') == 'completed' and exact.get('train_run_completed_normally') is True
            and exact.get('exact_restoration_observation') == 'before first resumed forward'
            and numeric.get('status') == 'failed_strict_numeric_reproducibility'
            and numeric.get('same_anchor_trajectory_within_original_tolerance') is False
            and numeric.get('tolerance_relaxed') is False
            and numeric.get('original_floating_tolerance') == {'rtol': 1e-5, 'atol': 1e-6},
            'Resume disclosure/evidence differs')
    return output, {'original_val_best_epoch': best_epoch, 'original_val_best_abs_rel': best,
                    'migration': migration, 'config': cfg, 'complete': complete,
                    'strict_numeric_reproducibility': preflight['strict_numeric_reproducibility'],
                    'exact_restore': preflight['resume_audit'], 'numerical_diagnostic': numeric}


def metric_tables(chains):
    flat, paired = [], []
    for chain, payload in chains.items():
        for subset, domains in payload['metrics'].items():
            for domain, methods in domains.items():
                for method, regions in methods.items():
                    for region, item in regions.items():
                        for averaging in ('image_macro', 'pixel_pooled'):
                            for metric, value in item[averaging].items():
                                flat.append({'chain': chain, 'subset': subset, 'domain': domain,
                                             'method': method, 'region': region, 'averaging': averaging,
                                             'metric': metric, 'value': value, 'images': item['images'],
                                             'pixels': item['pixels'], 'normal_pixels': item['normal_pixels'],
                                             'metric_images': item['metric_image_counts'].get(metric),
                                             'normal_reference_counts': json.dumps(item['normal_reference_counts'], sort_keys=True)})
        for subset, entries in payload['paired'].items():
            for key, value in entries.items():
                domain, region, metric = key.split('/')
                interval = value['image_bootstrap_95pct']
                paired.append({'chain': chain, 'subset': subset, 'domain': domain, 'region': region,
                               'metric': metric, 'comparison': 'stage1_lidf_minus_stage1_native',
                               'paired_images': value['paired_images'], 'difference': value['difference_lidf_minus_stage1'],
                               'ci95_low': interval[0] if interval else None, 'ci95_high': interval[1] if interval else None})
    return flat, paired


def verify_cross_chain_labels(original, current):
    # Historical native_fusion.benchmark calls official normals independent_normal;
    # the portable label adapter names their file normal_path.
    for old_key, new_key in (('depth', 'target_depth_path'), ('transparent_mask', 'transparent_mask_path'),
                             ('independent_normal', 'normal_path')):
        before, after = original['label_assets'].get(old_key), current['label_assets'].get(new_key)
        require((before is None) == (after is None) and (before is None or before['sha256'] == after['sha256']),
                'Historical and portable metrics used different GT assets')
    require(current['normal_reference'] == original['normal_reference'], 'Normal reference type differs')
    for region in REGIONS:
        require(current['metrics']['stage1_lidf'][region]['pixels'] == original['metrics']['stage1_lidf'][region]['pixels'],
                'Historical and portable depth support differs')


def selection_table(ranked, resolved, evidence):
    result = []
    for order, row in enumerate(ranked, 1):
        metrics = read(evidence.check(row['metrics']))
        macro = metrics['full1226']['combined']
        close(macro['transparent']['image_macro']['mae_mm'], row['transparent_mae_mm'], 'Sweep metric score changed')
        value = {'epoch': row['epoch'], 'original_sweep_rank': order,
                 'final_selected': row['epoch'] == resolved['selected_epoch'],
                 'checkpoint_sha256': row['checkpoint_sha256'], 'images': row['images'],
                 'transparent_mae_mm': row['transparent_mae_mm'],
                 'transparent_metric_images': macro['transparent']['metric_image_counts']['mae_mm'],
                 'same_gpu_resolution_mae_mm': resolved['scores'].get(str(row['epoch']), {}).get('transparent_mae_mm')}
        for subset in ('full1226', 'test521'):
            for domain in DOMAINS:
                for region in REGIONS:
                    item = metrics[subset][domain][region]['image_macro']
                    value.update({f'{subset}_{domain}_{region}_{m}': item[m] for m in ('mae_mm', 'rmse_mm', 'abs_rel')})
        result.append(value)
    return sorted(result, key=lambda r: r['epoch'])


def collect(args):
    evidence = Evidence()
    selection = evidence.json(args.selection_root / 'selection.complete.json')
    profile = verification_profile(args)
    verified = evidence.json(args.selection_root / 'publication_verification.complete.json')
    resolved = evidence.json(args.historical_geometry / 'resolved_selection.json')
    selection_ref = evidence.add(args.selection_root / 'selection.complete.json')
    verification_ref = evidence.add(args.selection_root / 'publication_verification.complete.json')
    resolved_ref = evidence.add(args.historical_geometry / 'resolved_selection.json')
    winner = validate_selection(selection, verified, resolved, selection_ref, verification_ref)
    sweep_contract_path = evidence.check(selection['contract'])
    require(sweep_contract_path.resolve() == (args.selection_root / 'contract.json').resolve(), 'Sweep contract path differs')
    sweep_contract = read(sweep_contract_path)
    config = evidence.json(args.release_config)
    require(config.get('status') in ('selection_complete_portability_pending', 'verified_release'),
            'Release candidate selection is not complete')
    require(config['selection']['checkpoint_epoch'] == winner['epoch']
            and config['selection']['checkpoint_sha256'] == winner['checkpoint_sha256']
            and config['selection'].get('independent_test_estimate') is False,
            'Release configuration does not declare the authenticated test-selected winner')
    require(config['selection'].get('selection_sha256') == selection_ref['sha256']
            and config['selection'].get('verification_sha256') == verification_ref['sha256']
            and config['selection'].get('resolved_selection_sha256') == resolved_ref['sha256']
            and config['selection'].get('winner_geometry_sha256') == sha(args.historical_geometry / 'evaluation.complete.json'),
            'Release configuration audit identities differ')
    # Public CLI checks the actual release bundle; this import is CPU-only.
    import sys
    sys.path.insert(0, str(PROJECT / 'src'))
    from transparent_depth.cli import project_path, validate_release_assets
    from transparent_depth.assets import destination
    assets = (Path(args.asset_root).expanduser().resolve() if getattr(args, 'asset_root', None)
              else project_path(PROJECT, config['asset_root']))
    validate_release_assets(PROJECT, config, assets)
    bundle = destination(assets, config['stage2']['bundle'])
    metadata_ref, weights_ref = evidence.add(bundle / 'model.json'), evidence.add(bundle / 'model.pt')
    evidence.add(project_path(PROJECT, config['stage1']['assets_manifest']))
    historical_marker = evidence.json(args.historical_geometry / 'evaluation.complete.json')
    require(historical_marker.get('status') == 'complete' and historical_marker.get('images') == 1226
            and historical_marker.get('test_images') == 521 and historical_marker.get('selected_epoch') == winner['epoch']
            and historical_marker['selected_checkpoint'] == resolved['selected_checkpoint']
            and historical_marker['resolved_selection'] == resolved_ref
            and historical_marker['sweep_verification'] == verification_ref
            and historical_marker.get('ground_truth_used_as_prediction_input') is False,
            'Historical winner geometry is incomplete or from another selection')
    close(historical_marker['transparent_mae_mm'], resolved['scores'][str(winner['epoch'])]['transparent_mae_mm'],
          'Historical geometry score differs', 2e-5)
    chains = {'historical_cached': report_payload(args.historical_geometry, historical_marker, evidence, historical=True)}
    quality = None
    if getattr(args, 'cached_quality_acceptance', None) is not None:
        quality = validate_cached_quality_acceptance(args.cached_quality_acceptance,
            args.cached_validation, args.historical_geometry, evidence)
    portable = [('portable_cached', args.cached_validation, 'cached')]
    if profile == 'full_live':
        portable.append(('portable_live', args.live_validation, 'live'))
    for name, root, mode in portable:
        chains[name] = (quality['portable_payload'] if mode == 'cached' and quality is not None else
                       portable_payload(root, mode, winner, resolved_ref, metadata_ref, weights_ref, evidence))
        require(chains[name]['marker']['selected_epoch'] == winner['epoch']
                and chains[name]['marker']['selected_checkpoint_sha256'] == winner['checkpoint_sha256']
                and chains[name]['contract']['selection']['resolved_selection'] == resolved_ref
                and chains[name]['contract']['dependencies']['bundle_metadata'] == metadata_ref
                and chains[name]['contract']['dependencies']['bundle_weights'] == weights_ref,
                'Cached quality acceptance does not match this final release bundle')
        validate_roster(chains[name]['rows'], chains['historical_cached']['rows'])
        dependency = chains[name]['contract']['dependencies']
        require(dependency['input_manifest'] == sweep_contract['input_manifest']
                and chains[name]['marker']['labels']['sha256'] == sweep_contract['label_manifest']['sha256'],
                'Portable input/label roster differs from selected sweep')
        original_rows = {r['cache_id']: r for r in chains['historical_cached']['rows']}
        for row in chains[name]['rows']:
            verify_cross_chain_labels(original_rows[row['cache_id']], row)
        if mode == 'cached':
            require(dependency['stage1_seal'] == sweep_contract['stage1_seal'], 'Cached validation used a different Stage1 seal')
    acceptance = {}
    if profile == 'training_release':
        stage1 = training_release_stage1(args.stage1_acceptance, config, evidence)
        single = training_acceptance(args.single_training_acceptance, 1, stage1, evidence)
        ddp = training_acceptance(args.ddp_training_acceptance, 2, stage1, evidence)
        require(single.get('stage1_acceptance') == evidence.add(args.stage1_acceptance)
                and ddp.get('stage1_acceptance') == evidence.add(args.stage1_acceptance),
                'Training smoke was not gated on this Stage1 policy acceptance')
        acceptance = {'stage1': evidence.add(args.stage1_acceptance),
                      'single_training': evidence.add(args.single_training_acceptance),
                      'ddp_training': evidence.add(args.ddp_training_acceptance),
                      'trace_cases': 3, 'relocation_cases': 1, 'fresh_stage1_images': 12,
                      'training_updates_per_run': 9, 'training_world_sizes': [single['world_size'], ddp['world_size']]}
    selection_rows = selection_table(selection['ranking'], resolved, evidence)
    epochs, training = training_rows(args.training_root, selection['ranking'], evidence)
    # Bind the table's archived epoch bytes to the verifier's full rehash evidence.
    receipts = lines(args.training_root / 'epoch_checkpoints.jsonl')
    for receipt in receipts:
        require(verified['evidence_sha256'].get(str(Path(receipt['path']).resolve())) == receipt['sha256'],
                'Sweep verification did not authenticate this training checkpoint')
    return {'evidence': evidence, 'selection': selection, 'resolved': resolved, 'winner': winner,
            'release_config': config, 'chains': chains, 'selection_rows': selection_rows,
            'epochs': epochs, 'training': training, 'verification_profile': profile,
            'training_release_acceptance': acceptance, 'cached_quality_acceptance': quality,
            'asset_root': str(assets)}


def number(value, digits=6):
    return '—' if value is None else f'{value:.{digits}f}'


def delta_description(current, baseline, lower=True):
    if current is None or baseline is None:
        return '无共同有效支持'
    delta = current - baseline
    change = 100 * delta / abs(baseline) if baseline else None
    label = '改善' if (delta < 0 if lower else delta > 0) else ('持平' if delta == 0 else '退化')
    return f'{label}，Δ={delta:+.6f}' + (f'（{change:+.2f}%）' if change is not None else '（基线为零，不算百分比）')


def markdown(info, flat, paired):
    winner, training, chains = info['winner'], info['training'], info['chains']
    lines_out = ['# RGB-only 两阶段深度估计：完整实验与发布验证', '',
        '**本报告使用完整 1226 张图（含 521 张测试图）的标签，从 80 个 checkpoint 中事后选模。'
        '所有 1226／521 数字均不是独立测试泛化估计。**', '',
        f"最终发布候选为第 **{winner['epoch']}** 轮，原 checkpoint SHA256：`{winner['checkpoint_sha256']}`。",
        '冻结第一阶段为 V2 + 已训练的 E08 拉普拉斯适配器；第二阶段为 LIDF stage1 + SPVNAS20G。'
        '推理输入仅为 RGB 与已知标定 K，深度 P 由 RGB 模型生成，预测阶段不使用 GT 深度或透明掩码。', '',
        '## 选择协议和数据覆盖', '',
        '选模主指标是原生分辨率透明区域的逐图宏平均 MAE（mm），越低越好。'
        '无有效透明像素的图保持 null，不计入该指标宏平均；不是补零。'
        '原 sweep 同分优先较早 epoch；差距不超过 2×10⁻⁵ mm 的候选按封存裁决在同一 GPU 重算。'
        '最终选择取自 resolved_selection.json，不假定 E63 或原 sweep 排名第一永远获胜。',
        f"原训练协议只在第 51–80 轮按 704 张纯 P 验证的全图 AbsRel 选模："
        f"第 {training['original_val_best_epoch']} 轮，AbsRel={training['original_val_best_abs_rel']:.9f}。"
        '此结果和原 checkpoint 合同保留，与本次测试集参与选模分开。', '',
        f"## {len(chains)} 条完整 1226 张证据链", '',
        '| 链 | 意义 | 全1226透明MAE（mm） |', '|---|---|---:|']
    for name in chains:
        meaning = {'historical_cached': '选模所依据的历史封存预测',
                   'portable_cached': '同 P 输入，验证便携 Stage2 一致性',
                   'portable_live': 'RGB 实算第一阶段后的实际两阶段发布效果'}[name]
        lines_out.append(f"| {LABELS[name]} | {meaning} | {number(chains[name]['marker']['transparent_mae_mm'], 9)} |")
    cached = chains['portable_cached']['marker']
    quality = info.get('cached_quality_acceptance')
    if quality is None:
        lines_out += ['', f"同 P 全量验证通过：低分辨率深度绝对容差 10⁻⁶ m、refined mask 精确一致，"
                      f"透明 MAE 差 {cached['transparent_mae_change_vs_historical_mm']:+.9g} mm，原上限 2×10⁻⁵ mm。"]
    else:
        stats = quality['statistics']; summary = stats['summary']
        lines_out += ['', '**Stage2 原严格数值验收失败，原失败 marker 和阈值完整保留；没有改写为严格通过。**',
            '本 training_release 采用独立的近似质量接受记录，依据用户最新“后续训练效果差不多、重点整理上传”的范围。'
            '下列操作界限由 root 在严格失败后提出并审核，用户没有逐项指定这些数值；不是自动放宽单像素最大误差。',
            f"原 10⁻⁶ m 阈值下，低分辨率共有 {summary['low_images_over_1e6']} 张、"
            f"{summary['low_pixels_over_1e6']}/{summary['low_pixels']} 像素超限；"
            f"最大单像素差 {summary['max_low_abs_m']:.9g} m，原生对照最大差 {summary['max_native_abs_m']:.9g} m。",
            f"逐图平均绝对差的最大值：低分辨率 {summary['max_per_image_low_mean_abs_m']:.9g} m，"
            f"原生对照 {summary['max_per_image_native_mean_abs_m']:.9g} m，均不超过 10⁻⁶ m。"
            '原生数组差的参照是历史低分辨率预测的 CPU resize；GT 质量另由两边实际原生评测逐图对齐。',
            '每图低分辨率超限比例≤10⁻⁴、总体≤10⁻⁶，refined mask 精确一致；'
            '完整1226／521×合并／真实／合成×全图／透明／背景的宏平均 MAE 差均≤0.01 mm。'
            '旧版自身重复的原始证据单独封存；近似质量接受不宣称逐像素相等或未来训练轨迹相等。', '',
            '| 子集/域/区域 | 原 MAE mm | 新 MAE mm | 新−原 mm | 有效图数 |', '|---|---:|---:|---:|---:|']
        for name, row in stats['partitions'].items():
            lines_out.append(f"| {name} | {number(row['historical_mae_mm'], 9)} | {number(row['portable_mae_mm'], 9)} | "
                             f"{number(row['signed_mae_change_mm'], 9)} | {row['metric_images']} |")
    if 'portable_live' in chains:
        live_delta = chains['portable_live']['marker']['transparent_mae_mm'] - chains['historical_cached']['marker']['transparent_mae_mm']
        lines_out += [f"真实新 RGB 链相对历史胜者的透明 MAE 差为 {live_delta:+.9g} mm。"
                      '此链独立列出实测值；同 P 通过不能替代新 RGB 链的全量验证。']
    else:
        lines_out += ['**验证范围为 training_release：完整 1226 张实时 RGB→E08→Stage2 链未重算。**'
                      '上表仅包含历史封存 P 和同一 P 的便携 Stage2 两条全量链，没有第三条 live 全量指标。',
                      '另有 3 例实时旧/新 Stage1 对照、1 例迁移目录对照、12 张新 RGB 预测，'
                      '以及单卡和 2 卡各 9 次实际优化更新、真实中断/恢复的验收。'
                      '这些证据支持训练入口和已测配置的功能一致性，不保证将来重新训练得到相同分数。',
                      '第一阶段保留原算子与权重。用户确认的固定验收为：42 项专家/V2输入等字段 exact；'
                      '4 项米制 V2/后续深度分别满足最大绝对差≤5×10⁻⁵ m、平均绝对差≤10⁻⁶ m；'
                      '9 项非米制中间量只要求有限值和相同 schema，不宣称 exact。'
                      'CUDA scatter 归约重复性证据与原零容差失败保留；Stage1 规则不用于替代 Stage2 验收，'
                      'Stage2 严格结果与任何独立附加质量接受记录均分别披露。']
    lines_out += ['',
                  '## 域与几何差异', '',
                  '以下比较均为 LIDF 输出减去同一链的原生 E08；MAE、RMSE、AbsRel、法线角误差越低越好，边缘 F1 越高越好。'
                  '合并均值不能代表真实、合成或背景分别改善。', '',
                  '| 链 | 子集 | 域 | 区域 | MAE变化（mm） | 法线角误差变化（°） | 边缘F1变化 |',
                  '|---|---|---|---|---|---|---|']
    for name in chains:
        for subset in ('full1226', 'test521'):
            for domain in DOMAINS:
                for region in REGIONS:
                    methods = chains[name]['metrics'][subset][domain]
                    a, b = (methods[m][region]['image_macro'] for m in ('stage1_lidf', 'stage1_native'))
                    changes = [delta_description(a[m], b[m], m != 'edge_f1') for m in ('mae_mm', 'normal_mean_deg', 'edge_f1')]
                    lines_out.append('| ' + ' | '.join([LABELS[name], subset, LABELS[domain], LABELS[region], *changes]) + ' |')
    lines_out += ['', '## 完整三方法指标表（逐图宏平均）', '',
                  '深度单位 mm，AbsRel 与 F1 无量纲；法线角误差单位度。'
                  '“—”表示无有效支持，不等于 0。各指标有效图数、像素数、法线参考类型及像素池化结果在 metrics_all.csv 中完整保留。'
                  '合成法线使用官方 camera normals；真实法线为 GT 深度诱导代理。混合域法线只作汇总，不能当作同一种独立真值。', '']
    for name in chains:
        lines_out += [f'### {LABELS[name]}', '',
                      '| 子集 | 域 | 区域 | 方法 | 图数 | MAE | RMSE | AbsRel | 法线角误差 | 边缘F1 |',
                      '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
        for subset in ('full1226', 'test521'):
            for domain in DOMAINS:
                for region in REGIONS:
                    for method in METHODS:
                        item = chains[name]['metrics'][subset][domain][method][region]
                        values = [number(item['image_macro'][m]) for m in DISPLAY_METRICS]
                        lines_out.append('| ' + ' | '.join([subset, LABELS[domain], LABELS[region], LABELS[method], str(item['images']), *values]) + ' |')
    lines_out += ['', '## 配对差值与置信区间', '',
                  '沿用既定按图配对 bootstrap（2000 次，seed=23），差值方向为 LIDF − 原生 E08。'
                  '负 MAE／AbsRel 表示平均误差降低；区间跨零时不据此声称稳定改善。'
                  '区间不计入场景相关性、80 次 checkpoint 选择或历史测试曝光的不确定性。'
                  '法线与边缘没有伪造额外 CI，使用上表与各自支持量。', '',
                  '| 链 | 子集 | 域 | 区域 | 指标 | 配对图数 | 平均差 | 95%区间 |',
                  '|---|---|---|---|---|---:|---:|---|']
    for row in paired:
        interval = '—' if row['ci95_low'] is None else f"[{number(row['ci95_low'])}, {number(row['ci95_high'])}]"
        lines_out.append('| ' + ' | '.join([LABELS[row['chain']], row['subset'], LABELS[row['domain']],
            LABELS[row['region']], row['metric'], str(row['paired_images']), number(row['difference']), interval]) + ' |')
    first, last = info['epochs'][0], info['epochs'][-1]
    lines_out += ['', '## 训练过程与恢复边界', '',
        '45453 张训练图，80 轮，课程分段 20／30／30。前段 GT 背景+预测透明；中段背景按 alpha 混合 GT 与预测；后段全 P。'
        '监督目标全程为 GT。纯 P 验证始终 alpha=0；课程匹配验证 alpha 等于该轮最后实际训练 step，含标签的验证不当作 RGB-only 部署性能。',
        '早期不含本次 SPVNAS 扩展的 60 轮、20／20／20 课程属于前一实验；本次后续获准协议为 80 轮、20／30／30，未将两次实验混作同一训练记录。',
        f"记录的第 1／80 轮训练 total loss：{first['train_loss_total']:.9f}／{last['train_loss_total']:.9f}。"
        '该值是原记录中的按样本数汇总的 microbatch 支持归一化损失，不能声称与单次大 batch loss 完全等价。',
        '第 1–29 轮：8 卡，micro2×累积2，有效 batch32，每轮45456次呈现（3次 padding）／1421次更新。'
        '第 30–80 轮：3 卡，micro2×累积5，有效 batch30，每轮45453次呈现／1516次更新；总计118525次更新。'
        '实际末累积窗按真实 microbatch 数归一化；SyncBN 同步 microbatch，不同步累计的有效 batch。'
        '这次迁移改变 batch／SyncBN 和采样轨迹，不声称等价于完整8卡训练。',
        '第29轮→三卡迁移的三份现场恢复审计确认模型、Adam、调度器与所映射 RNG 精确加载。'
        '另，训练前八卡真实恢复状态审计已通过，但其同锚点 CUDA 训练轨迹未通过原 rtol=1e-5、atol=1e-6 严格数值重复性阈值，失败记录保留且未放宽容差。'
        '“精确加载状态”与“后续训练轨迹逐数值复现”是两件事。',
        '本实验同时改变 RGB ImageNet 初始化、SPVNAS20G、课程、80轮、差分学习率与梯度累积等；不是单因素因果消融。', '',
        '![训练与两种验证](training_curves.png)', '', '![80轮透明MAE选模](selection_curve.png)', '',
        '## 可审查产物', '',
        '- `selection_80_epochs.csv`：全部80轮深度指标、原排序、最终裁决、checkpoint SHA。',
        '- `training_80_epochs.csv`：loss、双验证、alpha、LR、batch、呈现数、实际更新数。',
        f'- `metrics_all.csv`：{len(chains)} 条完整链／三方法／域／区域／宏平均与像素池化，包括原始完整几何指标。',
        '- `paired_ci.csv`：完整1226及521、真实与合成、各区域配对误差区间。',
        '- `report.complete.json`：全部输入与输出 SHA、最终候选身份、完整性边界；不修改发布配置状态。',
        '- 图同时提供 PNG 与 SVG。所有曲线来自上述原始80轮记录，无平滑、无插值补点。', '']
    return '\n'.join(lines_out)


def write_csv(path, rows):
    require(rows, 'Cannot write an empty table')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plots(output, info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 300})
    epochs = info['epochs']
    x = [r['epoch'] for r in epochs]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), constrained_layout=True)
    specifications = [('train_loss_total', None, 'Training total loss'),
                      ('pure_p_mae_mm', 'phase_matched_mae_mm', 'Validation MAE (mm)'),
                      ('pure_p_abs_rel', 'phase_matched_abs_rel', 'Validation AbsRel'),
                      ('alpha_last', 'alpha_mean', 'Training alpha'),
                      ('effective_global_batch', 'global_microbatch', 'Batch size'),
                      ('lr_lidf', 'lr_adapters', 'Learning rate')]
    for ax, (a, b, title) in zip(axes.flat, specifications):
        ax.plot(x, [r[a] for r in epochs], color='#0072B2', lw=1.4, label=a.replace('_', ' '))
        if b:
            ax.plot(x, [r[b] for r in epochs], color='#D55E00', lw=1.2, ls='--', label=b.replace('_', ' '))
        for start, end, color in ((1, 20.5, '#f5e7c6'), (20.5, 50.5, '#e1eaf2'), (50.5, 80, '#dcecdc')):
            ax.axvspan(start, end, color=color, alpha=.28, zorder=-2)
        ax.axvline(29.5, color='#6b6b6b', lw=.7, ls=':')
        ax.set(xlim=(1, 80), xlabel='Epoch', ylabel=title)
        ax.grid(alpha=.18)
        if b:
            ax.legend(fontsize=6.7, frameon=False)
        if title == 'Learning rate':
            ax.set_yscale('log')
    fig.suptitle('Observed training history: 20 / 30 / 30 curriculum; 8-to-3 GPU migration after E29', fontsize=11)
    for extension in ('png', 'svg'):
        fig.savefig(output / ('training_curves.' + extension))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    rows = info['selection_rows']
    ax.plot([r['epoch'] for r in rows], [r['transparent_mae_mm'] for r in rows], color='#0072B2', lw=1.3)
    winner = info['resolved']['selected_epoch']
    score = info['resolved']['scores'][str(winner)]['transparent_mae_mm']
    ax.scatter([winner], [score], color='#D55E00', marker='*', s=120, zorder=3, label=f'Resolved E{winner}: {score:.6f} mm')
    ax.set(xlabel='Checkpoint epoch', ylabel='Native transparent MAE (mm)', xlim=(1, 80),
           title='Post-hoc selection on all 1226 images (includes 521 test images)')
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    for extension in ('png', 'svg'):
        fig.savefig(output / ('selection_curve.' + extension))
    plt.close(fig)


def build(args):
    info = collect(args)  # No output directory/files are created before all gates pass.
    output = args.output.resolve()
    require(not output.exists() or not any(output.iterdir()), 'Preserve existing reports; choose an empty output directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-build-', dir=output.parent) as temporary:
        stage = Path(temporary)
        flat, paired = metric_tables(info['chains'])
        write_csv(stage / 'selection_80_epochs.csv', info['selection_rows'])
        write_csv(stage / 'training_80_epochs.csv', info['epochs'])
        write_csv(stage / 'metrics_all.csv', flat)
        write_csv(stage / 'paired_ci.csv', paired)
        quality = info.get('cached_quality_acceptance')
        if quality is not None:
            (stage / 'cached_quality_statistics.json').write_text(json.dumps(
                quality['statistics'], ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        (stage / 'FINAL_REPORT.zh-CN.md').write_text(markdown(info, flat, paired), encoding='utf-8')
        plots(stage, info)
        evidence = info['evidence']
        evidence.add(Path(__file__))
        # Catch concurrent replacements of any already-read small evidence file.
        for path, digest in evidence.files.items():
            require(sha(path) == digest, 'Evidence changed while building report: ' + path)
        marker = {'schema': 'transparent_depth_final_release_report_v1', 'status': 'complete',
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'cpu_only': True,
                  'epochs': 80, 'images_per_chain': 1226, 'test_images_per_chain': 521,
                  'verification_profile': info['verification_profile'],
                  'full_roster_chains': list(info['chains']),
                  'full_live_1226_executed': 'portable_live' in info['chains'],
                  'training_release_acceptance': info['training_release_acceptance'],
                  'cached_acceptance_kind': 'strict_original_thresholds' if quality is None else 'accepted_approximate_quality',
                  'cached_strict_numerical_parity_passed': quality is None,
                  'cached_quality_acceptance': None if quality is None else quality['reference'],
                  'asset_root': info['asset_root'],
                  'selected_epoch': info['winner']['epoch'], 'checkpoint_sha256': info['winner']['checkpoint_sha256'],
                  'selection_is_test_informed': True, 'independent_test_estimate': False,
                  'release_config_status_observed': info['release_config']['status'],
                  'does_not_promote_release_status': True, 'original_val704_best_epoch': info['training']['original_val_best_epoch'],
                  'same_anchor_trajectory_within_original_tolerance': False,
                  'large_array_authentication': 'inherits completed independent seals; does not reread 98080 prediction arrays or 80 checkpoints',
                  'input_sha256': evidence.files,
                  'outputs': {p.name: sha(p) for p in sorted(stage.iterdir())}}
        (stage / 'report.complete.json').write_text(json.dumps(marker, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        output.mkdir(exist_ok=True)
        for path in sorted(stage.iterdir(), key=lambda p: p.name == 'report.complete.json'):
            os.replace(path, output / path.name)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection-root', 'training-root', 'historical-geometry', 'cached-validation',
                 'release-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--verification-profile', choices=PROFILES, default='full_live')
    parser.add_argument('--live-validation', type=Path, help='Required by default full_live profile')
    parser.add_argument('--stage1-acceptance', type=Path, help='Three-case + relocation + fresh12 complete.json')
    parser.add_argument('--single-training-acceptance', type=Path)
    parser.add_argument('--ddp-training-acceptance', type=Path)
    parser.add_argument('--cached-quality-acceptance', type=Path,
                        help='Explicit root-reviewed additional quality acceptance; training_release only, preserves strict failures')
    parser.add_argument('--asset-root', type=Path, help='Explicit external asset directory; overrides release config default')
    args = parser.parse_args()
    try:
        marker = build(args)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, 'Final report refused: ' + str(error) + '\n')
    print(json.dumps({key: marker[key] for key in ('status', 'selected_epoch', 'checkpoint_sha256', 'cpu_only')}))


if __name__ == '__main__':
    main()
