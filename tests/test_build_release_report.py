"""CPU-only adversarial report fixtures; these are not experiment results."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_report', PROJECT / 'scripts/build_release_report.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')
    return report.reference(path)


def selection():
    ranks = [dict(epoch=i, checkpoint_sha256=f'{i:064d}', images=1226, smoke=False,
                  transparent_mae_mm=10. + i) for i in range(1, 81)]
    ranks[1]['transparent_mae_mm'] = ranks[0]['transparent_mae_mm'] + 1e-5
    policy = dict(epochs=list(range(1, 81)), selection_images=1226,
        primary_metric='full1226/combined/transparent/image_macro/mae_mm', tie_break='smaller_epoch',
        empty_region='null_per_image_excluded_from_macro', uses_test_labels_for_checkpoint_selection=True,
        independent_test_performance_claimed=False, gt_or_mask_used_as_prediction_input=False)
    selected = dict(status='complete', completed_epochs=80, ranking=ranks, selection_policy=policy)
    sr, vr = dict(path='/selection', sha256='s'), dict(path='/verification', sha256='v')
    verified = dict(status='passed', epochs=80, predictions_verified=98080, selection=sr)
    resolved = dict(status='complete', original_selection=sr, verification=vr,
        selected_epoch=2, selected_checkpoint={'sha256': ranks[1]['checkpoint_sha256']},
        candidate_epochs=[1, 2], near_tie_tolerance_mm=2e-5, recomputed_on_one_gpu=True,
        scores={'1': {'transparent_mae_mm': 11.00001}, '2': {'transparent_mae_mm': 11.}})
    return selected, verified, resolved, sr, vr


def test_all80_and_final_near_tie_winner_not_original_rank_one():
    values = selection()
    assert report.validate_selection(*values)['epoch'] == 2
    bad = copy.deepcopy(values)
    bad[2]['selected_epoch'] = 1
    bad[2]['selected_checkpoint']['sha256'] = bad[0]['ranking'][0]['checkpoint_sha256']
    with pytest.raises(ValueError, match='does not minimize'):
        report.validate_selection(*bad)
    bad = copy.deepcopy(values)
    bad[0]['ranking'].pop()
    with pytest.raises(ValueError, match='missing/duplicate'):
        report.validate_selection(*bad)
    bad = copy.deepcopy(values)
    bad[0]['selection_policy']['independent_test_performance_claimed'] = True
    with pytest.raises(ValueError, match='disclosure'):
        report.validate_selection(*bad)


def test_references_fail_on_changed_bytes(tmp_path):
    value = write(tmp_path / 'audit.json', {'status': 'complete'})
    evidence = report.Evidence()
    assert evidence.check(value) == tmp_path / 'audit.json'
    write(tmp_path / 'audit.json', {'status': 'incomplete'})
    with pytest.raises(ValueError, match='changed'):
        evidence.check(value)


def small_metrics():
    # Two test images: real deteriorates, synthetic improves more. The third
    # validation image has no transparent support. Never replace null by zero.
    rows = []
    for index, (domain, base, final, split) in enumerate((('real', 10., 12., 'real-test'),
            ('synthetic', 30., 10., 'synthetic-test'), ('real', 20., 22., 'real-val'))):
        row = dict(cache_id=str(index), sample_id=str(index), rgb_sha256=f'{index:064d}',
                   native_hw=[2, 3], domain=domain, official_split=split, benchmark_group=domain + '_group',
                   metrics={}, normal_reference='depth_proxy' if domain == 'real' else 'independent')
        for method, value in zip(report.METHODS, (base, base + .5, final)):
            row['metrics'][method] = {}
            for region in report.REGIONS:
                empty = index == 2 and region == 'transparent'
                item = {'pixels': 0 if empty else 6, 'mae_mm': None if empty else value,
                        'rmse_mm': None if empty else value + 1, 'abs_rel': None if empty else value / 100,
                        'normal_mean_deg': None if empty else value / 2,
                        'edge_f1': None if empty else 1 - value / 100}
                row['metrics'][method][region] = item
        rows.append(row)
    summary, paired = {}, {}
    for subset, part in report.subset_rows(rows).items():
        summary[subset] = {}
        for domain in ('combined', *sorted({r['domain'] for r in part})):
            selected = [r for r in part if domain == 'combined' or r['domain'] == domain]
            summary[subset][domain] = {}
            for method in report.METHODS:
                summary[subset][domain][method] = {}
                for region in report.REGIONS:
                    values = [r['metrics'][method][region] for r in selected]
                    entries = {k: [v[k] for v in values if v[k] is not None] for k in report.DISPLAY_METRICS}
                    macro = {k: sum(v)/len(v) if v else None for k, v in entries.items()}
                    summary[subset][domain][method][region] = dict(images=len(selected),
                        pixels=sum(v['pixels'] for v in values), normal_pixels=sum(v['pixels'] for v in values),
                        image_macro=macro, pixel_pooled=macro.copy(),
                        metric_image_counts={k: len(v) for k, v in entries.items()},
                        normal_reference_counts={'fixture': len(selected)})
        if subset in ('full1226', 'test521'):
            paired[subset] = {}
            for domain in report.DOMAINS:
                selected = [r for r in part if domain == 'combined' or r['domain'] == domain]
                for region in report.REGIONS:
                    for metric in ('mae_mm', 'abs_rel'):
                        delta = [r['metrics']['stage1_lidf'][region][metric] - r['metrics']['stage1_native'][region][metric]
                                 for r in selected if r['metrics']['stage1_native'][region][metric] is not None]
                        mean = sum(delta)/len(delta) if delta else None
                        paired[subset][f'{domain}/{region}/{metric}'] = dict(paired_images=len(delta),
                            difference_lidf_minus_stage1=mean,
                            image_bootstrap_95pct=[min(delta), max(delta)] if delta else None)
    return dict(rows=rows, metrics=summary, paired=paired,
        marker={'transparent_mae_mm': 11., 'transparent_mae_change_vs_historical_mm': 0.})


def test_macro_domain_support_and_nulls_are_preserved():
    payload = small_metrics()
    report.validate_roster(payload['rows'], count=3, test_count=2)
    report.verify_summary(payload['metrics'], payload['paired'], payload['rows'])
    assert payload['metrics']['full1226']['combined']['stage1_lidf']['transparent']['image_macro']['mae_mm'] == 11.
    assert payload['metrics']['full1226']['real']['stage1_lidf']['all']['image_macro']['mae_mm'] == 17.
    altered = copy.deepcopy(payload)
    altered['metrics']['full1226']['real']['stage1_lidf']['all']['image_macro']['mae_mm'] = 11.
    with pytest.raises(ValueError, match='Macro metric'):
        report.verify_summary(altered['metrics'], altered['paired'], altered['rows'])
    altered = copy.deepcopy(payload)
    altered['paired']['full1226']['combined/transparent/mae_mm']['paired_images'] = 3
    with pytest.raises(ValueError, match='Paired image count'):
        report.verify_summary(altered['metrics'], altered['paired'], altered['rows'])
    changed_rgb = copy.deepcopy(payload['rows'])
    changed_rgb[0]['rgb_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        report.validate_roster(changed_rgb, payload['rows'], count=3, test_count=2)


def test_actual_historical_independent_normal_alias_is_bound_to_portable_labels():
    row = small_metrics()['rows'][1]
    original, portable = copy.deepcopy(row), copy.deepcopy(row)
    original['label_assets'] = {key: {'sha256': digest} for key, digest in
                                (('depth', 'd'), ('transparent_mask', 'm'), ('independent_normal', 'n'))}
    portable['label_assets'] = {key: {'sha256': digest} for key, digest in
                                (('target_depth_path', 'd'), ('transparent_mask_path', 'm'), ('normal_path', 'n'))}
    report.verify_cross_chain_labels(original, portable)
    portable['label_assets']['normal_path']['sha256'] = 'different_normals'
    with pytest.raises(ValueError, match='different GT assets'):
        report.verify_cross_chain_labels(original, portable)


def info_fixture():
    chains = {name: small_metrics() for name in report.CHAINS}
    epochs = []
    for epoch in range(1, 81):
        epochs.append(dict(epoch=epoch, train_loss_total=3 - epoch / 100,
            pure_p_mae_mm=50 - epoch / 10, phase_matched_mae_mm=30 + epoch / 10,
            pure_p_abs_rel=.08 - epoch / 10000, phase_matched_abs_rel=.05 + epoch / 10000,
            alpha_last=1. if epoch <= 20 else max(0., (50 - epoch) / 30),
            alpha_mean=1. if epoch <= 20 else max(0., (50 - epoch) / 30),
            effective_global_batch=32 if epoch <= 29 else 30,
            global_microbatch=16 if epoch <= 29 else 6,
            lr_lidf=1e-4 if epoch <= 60 else 1e-5, lr_adapters=1e-3 if epoch <= 60 else 1e-4))
    return dict(chains=chains, winner={'epoch': 2, 'checkpoint_sha256': 'a'*64},
        resolved={'selected_epoch': 2, 'scores': {'2': {'transparent_mae_mm': 11.}}},
        training={'original_val_best_epoch': 63, 'original_val_best_abs_rel': .057}, epochs=epochs,
        selection_rows=[{'epoch': i, 'transparent_mae_mm': 11 + i / 100} for i in range(1, 81)])


def test_tables_keep_three_chains_domains_and_geometry_degradation(tmp_path):
    info = info_fixture()
    flat, paired = report.metric_tables(info['chains'])
    assert {r['chain'] for r in flat} == set(report.CHAINS)
    assert {r['method'] for r in flat} == set(report.METHODS)
    assert {r['averaging'] for r in flat} == {'image_macro', 'pixel_pooled'}
    assert len(paired) == 3 * 2 * 3 * 3 * 2
    text = report.markdown(info, flat, paired)
    assert '不是独立测试' in text and '严格数值重复性' in text
    assert '| 真实 | 全图 | 退化' in text
    assert '新 RGB 实算 E08' in text and '2000 次' in text
    assert '第 **2** 轮' in text and '第 63 轮' in text
    report.write_csv(tmp_path / 'metrics.csv', flat)
    assert 'normal_reference_counts' in (tmp_path / 'metrics.csv').read_text()


def test_training_release_does_not_invent_third_chain_or_hide_real_degradation():
    info = info_fixture()
    del info['chains']['portable_live']
    info['verification_profile'] = 'training_release'
    flat, paired = report.metric_tables(info['chains'])
    text = report.markdown(info, flat, paired)
    assert {r['chain'] for r in flat} == {'historical_cached', 'portable_cached'}
    assert len(paired) == 2 * 2 * 3 * 3 * 2
    assert '完整 1226 张实时 RGB→E08→Stage2 链未重算' in text
    assert '没有第三条 live 全量指标' in text and '3 例实时' in text
    assert '9 次实际优化更新' in text and '| 真实 | 全图 | 退化' in text
    assert '### 新 RGB 实算' not in text and '不保证将来重新训练得到相同分数' in text


def test_additional_quality_report_preserves_strict_failure_and_large_pixel_difference():
    info = info_fixture(); del info['chains']['portable_live']
    info['verification_profile'] = 'training_release'
    info['cached_quality_acceptance'] = {'statistics': {'summary': {
        'low_images_over_1e6': 5, 'low_pixels_over_1e6': 5, 'low_pixels': 1226 * 76800,
        'max_low_abs_m': .05646, 'max_native_abs_m': .055,
        'max_per_image_low_mean_abs_m': 8e-7, 'max_per_image_native_mean_abs_m': 9e-7},
        'partitions': {'full1226/real/background': {'historical_mae_mm': 30.,
            'portable_mae_mm': 30.001, 'signed_mae_change_mm': .001, 'metric_images': 100}}}}
    flat, paired = report.metric_tables(info['chains'])
    text = report.markdown(info, flat, paired)
    assert '原严格数值验收失败' in text and '没有改写为严格通过' in text
    assert '0.05646 m' in text and '用户没有逐项指定这些数值' in text
    assert '同 P 全量验证通过' not in text
    assert 'CPU resize' in text and 'full1226/real/background' in text


def test_training_profile_must_be_explicit_and_complete():
    with pytest.raises(ValueError, match='full_live requires'):
        report.verification_profile(argparse.Namespace())
    args = argparse.Namespace(verification_profile='training_release', live_validation=None,
        stage1_acceptance=Path('/stage1'), single_training_acceptance=Path('/single'), ddp_training_acceptance=None)
    with pytest.raises(ValueError, match='real single/DDP'):
        report.verification_profile(args)
    args.ddp_training_acceptance = Path('/ddp')
    assert report.verification_profile(args) == 'training_release'
    args.live_validation = Path('/live')
    with pytest.raises(ValueError, match='exactly two'):
        report.verification_profile(args)


def policy_fixture():
    policy = {'depth_limits': {'max_abs_m': 5e-5, 'mean_abs_m': 1e-6, 'relative_tolerance': 0.}}
    fields = {}
    for name in report.EXACT_FIELDS | report.DEPTH_FIELDS | report.DIAGNOSTIC_FIELDS:
        exact, depth = name in report.EXACT_FIELDS, name in report.DEPTH_FIELDS
        fields[name] = dict(accepted=True, finite=True, max_abs=0. if exact else 1e-7,
            mean_abs=0. if exact else 1e-9, exact_equal=exact, exact_equivalence_required=exact,
            depth_positive=True if depth else None,
            acceptance_kind='exact' if exact else 'metric_depth_max_and_mean' if depth else 'diagnostic_finite_schema_only')
    ref = {'path': '/fixture/policy', 'sha256': 'policy'}
    record = dict(status='passed_user_policy', policy=ref, numeric_tolerance_user_authorized=True,
        numeric_tolerance_automatically_relaxed=False, nonmetric_intermediate_equivalence_claimed=False,
        depth_limits=policy['depth_limits'], fields=fields)
    return record, policy, ref


def test_stage1_fixed_policy_both_depth_limits_and_exact_upstream_are_enforced():
    record, policy, ref = policy_fixture()
    assert (len(report.EXACT_FIELDS), len(report.DEPTH_FIELDS), len(report.DIAGNOSTIC_FIELDS)) == (42, 4, 9)
    report.validate_policy_comparison(record, policy, ref)
    for field, key, value, message in (
        ('final_depth', 'max_abs', 5.001e-5, 'fixed user limits'),
        ('final_depth', 'mean_abs', 1.001e-6, 'fixed user limits'),
        ('dino/0/output', 'exact_equal', False, 'not exact'),
        ('lap/context/0/output', 'exact_equivalence_required', True, 'falsely claimed exact')):
        altered = copy.deepcopy(record); altered['fields'][field][key] = value
        with pytest.raises(ValueError, match=message):
            report.validate_policy_comparison(altered, policy, ref)
    record['fields']['unapproved'] = record['fields']['final_depth']
    with pytest.raises(ValueError, match='Unknown/missing'):
        report.validate_policy_comparison(record, policy, ref)


def smoke_fixture(tmp_path, monkeypatch, world=2):
    project = tmp_path / 'project'; source = project / 'src/transparent_depth/model_fixture.py'
    source.parent.mkdir(parents=True); source.write_text('# CPU fixture, not real model evidence\n')
    runner = project / 'scripts/run_stage2_smoke.py'; runner.parent.mkdir(); runner.write_text('# fixture\n')
    monkeypatch.setattr(report, 'PROJECT', project)
    root = tmp_path / 'smoke'; run = root / 'training'
    cfg = {'counts': {'val': 2}}
    cfg_ref = write(root / 'smoke.config.json', cfg)
    stage1 = {'fresh_predictions': write(tmp_path / 'predictions.jsonl', {'fixture': True})}
    producer_ref = write(root / 'prepared/stage1/complete.json',
                         {'source_predictions_sha256': stage1['fresh_predictions']['sha256']})
    contract_ref = write(run / 'contract.json', dict(config=cfg, world_size=world,
        stage1_complete_sha256=producer_ref['sha256'],
        sources={'transparent_depth/model_fixture.py': report.sha(source)}))
    complete_ref = write(run / 'smoke.complete.json', dict(smoke=True, epochs=3, updates=9,
                                                         contract_sha256=contract_ref['sha256']))
    write(run / 'smoke.interrupted.json', dict(status='controlled_epoch_stop', epoch=1, updates=3))
    epochs = []
    for e in range(1, 4):
        pure = dict(input_alpha=0., gt_or_mask_used_as_input=False, images=2, mae_mm=30., rmse_mm=40., abs_rel=.1)
        matched = {**pure, 'input_alpha': 1. if e == 1 else 0., 'gt_or_mask_used_as_input': e == 1}
        epochs.append(dict(epoch=e, updates=3*e, smoke=True, losses={'total': 1.},
                           validation=pure, validation_curriculum=matched))
    (run / 'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in epochs))
    alphas = [1., 1., 1., 1., .5, 0., 0., 0., 0.]; ranks = []
    for rank in range(world):
        resume = write(run / f'resume_verification_rank{rank}.json', dict(status='passed', updates=3,
            rank=rank, model_exact=True, optimizer_exact=True, scheduler_exact=True, rng_exact=True))
        changes_path = run / f'optimizer_changes_rank{rank}.jsonl'
        changes = [dict(updates=i+1, alpha=a, loss_finite=True, gradients_finite=True,
                       parameter_group_max_abs_change={g: .1 for g in ('lidf', 'spvnas', 'adapters')})
                   for i, a in enumerate(alphas)]
        changes_path.write_text(''.join(json.dumps(r)+'\n' for r in changes))
        ranks.append(dict(rank=rank, resume_sha256=resume['sha256'], optimizer_changes_sha256=report.sha(changes_path),
                          group_max_abs_change={g: .1 for g in ('lidf', 'spvnas', 'adapters')}))
    executions = []
    for phase, extra in [('fresh', ['--stop-after-epoch', '1']), ('resume', ['--resume'])]:
        log = root / (phase + '.log'); log.write_text('fixture success\n')
        executions.append(dict(exit_code=0, log=report.reference(log),
            command=['python', '-m', 'transparent_depth.train_stage2', '--output', str(run), *extra]))
    path = root / 'acceptance.complete.json'
    marker = dict(status='passed', diagnostic_smoke=True, formal_training_claimed=False, epochs=3, updates=9,
        world_size=world, input_alphas=alphas, config_sha256=cfg_ref['sha256'],
        training_complete_sha256=complete_ref['sha256'], source_predictions=stage1['fresh_predictions'],
        training_contract=contract_ref, ranks=ranks, runner_sha256=report.sha(runner), executions=executions)
    write(path, marker)
    return path, marker, stage1


def test_real_training_gate_requires_complete_ranks_same_prediction_source_and_exact_resume(tmp_path, monkeypatch):
    path, marker, stage1 = smoke_fixture(tmp_path, monkeypatch)
    report.training_acceptance(path, 2, stage1, report.Evidence())
    changed = copy.deepcopy(stage1); changed['fresh_predictions']['sha256'] = 'different'
    with pytest.raises(ValueError, match='different Stage1'):
        report.training_acceptance(path, 2, changed, report.Evidence())
    altered = {**marker, 'ranks': marker['ranks'][:1]}; write(path, altered)
    with pytest.raises(ValueError, match='Missing/duplicate'):
        report.training_acceptance(path, 2, stage1, report.Evidence())
    write(path, marker)
    rank_path = path.parent / 'training/resume_verification_rank1.json'
    resume = report.read(rank_path); resume['optimizer_exact'] = False
    updated = write(rank_path, resume); marker['ranks'][1]['resume_sha256'] = updated['sha256']; write(path, marker)
    with pytest.raises(ValueError, match='was not exact'):
        report.training_acceptance(path, 2, stage1, report.Evidence())


def test_matplotlib_writes_actual_cpu_png_and_svg(tmp_path):
    report.plots(tmp_path, info_fixture())
    assert (tmp_path / 'training_curves.png').read_bytes().startswith(b'\x89PNG')
    assert '<svg' in (tmp_path / 'training_curves.svg').read_text()
    assert (tmp_path / 'selection_curve.png').stat().st_size > 1000
    assert 'Post-hoc selection' in (tmp_path / 'selection_curve.svg').read_text()


def test_missing_complete_evidence_refuses_without_writing_output(tmp_path):
    args = argparse.Namespace(selection_root=tmp_path / 'unfinished', output=tmp_path / 'report')
    with pytest.raises(FileNotFoundError):
        report.build(args)
    assert not args.output.exists()


def test_failed_cached_parity_or_wrong_winner_refused_before_payload_reads(tmp_path):
    winner = dict(epoch=2, checkpoint_sha256='b'*64)
    marker = dict(status='complete', mode='cached', images=1226, test_images=521,
        selected_epoch=2, selected_checkpoint_sha256='b'*64, model_inputs_included_gt_or_mask=False,
        test_informed_selection=True, independent_test_performance_claimed=False,
        cached_input_parity_passed=False, cached_transparent_mae_agrees=True, cached_transparent_mae_tolerance_mm=2e-5)
    write(tmp_path / 'evaluation.complete.json', marker)
    with pytest.raises(ValueError, match='Cached-input parity'):
        report.portable_payload(tmp_path, 'cached', winner, {}, {}, {}, report.Evidence())
    marker['selected_checkpoint_sha256'] = 'c'*64
    write(tmp_path / 'evaluation.complete.json', marker)
    with pytest.raises(ValueError, match='another checkpoint'):
        report.portable_payload(tmp_path, 'cached', winner, {}, {}, {}, report.Evidence())


def test_training_csv_preserves_ancestral_batch_and_two_validation_conditions(tmp_path):
    root = tmp_path / 'training'
    root.mkdir()
    parent = dict(expected_world_size=8, batch_per_gpu=2, gradient_accumulation_steps=2,
                  curriculum={'steps_per_epoch': 1421}, sampler_protocol={'presentations_per_epoch': 45456})
    parent_ref = write(tmp_path / 'parent_config.json', parent)
    exact_ref = write(tmp_path / 'exact.json', dict(status='completed', train_run_completed_normally=True,
                     exact_restoration_observation='before first resumed forward'))
    numeric_ref = write(tmp_path / 'numeric.json', dict(status='failed_strict_numeric_reproducibility',
        same_anchor_trajectory_within_original_tolerance=False, tolerance_relaxed=False,
        original_floating_tolerance={'rtol': 1e-5, 'atol': 1e-6}))
    preflight_ref = write(tmp_path / 'preflight.json', dict(status='passed', resume_audit=exact_ref,
                                                         strict_numeric_reproducibility=numeric_ref))
    migration = dict(parent_config=parent_ref, parent_preflight=preflight_ref,
        parent_completed_epochs=29, parent_committed_updates=41209, final_updates=118525,
        parent_checkpoint={'path': '/fixture/epoch029.pt', 'sha256': '9'*64})
    cfg = dict(epochs=80, curriculum={'stage_epochs': [20, 30, 30], 'steps_per_epoch': 1516},
        counts={'train': 45453, 'val': 704, 'evaluation': 1226}, migration=migration,
        expected_world_size=3, batch_per_gpu=2, gradient_accumulation_steps=5,
        sampler_protocol={'presentations_per_epoch': 45453})
    write(root / 'contract.json', dict(config=cfg, migration={**migration, 'allowed_config_changes': []}))
    records, receipts, ranking = [], [], []
    updates, previous_alpha = 0, 1.
    for epoch in range(1, 81):
        steps = 1421 if epoch <= 29 else 1516
        updates += steps
        alpha = 1. if epoch <= 20 else max(0., (50-epoch)/30)
        pure = dict(images=704, input_alpha=0., gt_or_mask_used_as_input=False,
                    mae_mm=40., rmse_mm=50., abs_rel=.06, seconds=1.)
        matched = {**pure, 'input_alpha': alpha, 'gt_or_mask_used_as_input': alpha > 0}
        row = dict(epoch=epoch, updates=updates, smoke=False, optimizer_steps=steps,
            microsteps=2841 if epoch <= 29 else 7576, train_unique_expected=45453,
            train_presentations=45456 if epoch <= 29 else 45453, train_seconds=10.,
            validation=pure, validation_curriculum=matched, input_alpha_first=previous_alpha,
            input_alpha_last=alpha, input_alpha_mean=(previous_alpha+alpha)/2,
            selection_eligible=epoch>=51, selected=epoch==51,
            validation_curriculum_reuses_pure_p=alpha==0., learning_rates={'lidf': 1e-4},
            losses={'total': 1., 'position': .1, 'probability': .1, 'normal': .1})
        previous_alpha = alpha
        receipt = dict(epoch=epoch, path=str(root / f'epoch{epoch:03d}.pt'), sha256=f'{epoch:064d}',
            updates=updates, input_alpha=alpha, validation_abs_rel=.06, validation_curriculum_abs_rel=.06)
        records.append(row); receipts.append(receipt)
        ranking.append(dict(epoch=epoch, checkpoint_sha256=receipt['sha256']))
    (root / 'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (root / 'epoch_checkpoints.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in receipts))
    write(root / 'complete.json', dict(smoke=False, epochs=80, updates=updates,
        epoch_checkpoints_count=80, contract_sha256=report.sha(root / 'contract.json'),
        epoch_checkpoints_sha256=report.sha(root / 'epoch_checkpoints.jsonl'), best_score=.06))
    restored = []
    for rank in range(3):
        audit = dict(status='passed', parent_checkpoint=migration['parent_checkpoint'], new_rank=rank,
            old_rank=rank+5, physical_gpu=rank+5, checkpoint_fields_edited=False,
            checks={key: {'exact': True} for key in ('model', 'optimizer', 'scheduler', 'rng')})
        restored.append(write(tmp_path / f'restore_rank{rank}.json', {**audit,
            'phase': 'after_ddp_before_training', 'load_observation': {**audit, 'phase': 'after_load_before_ddp'}}))
    write(tmp_path / 'migration_live_verification.json', dict(status='verified_actual_training', world_size=3,
        gpu_ids=[5, 6, 7], checkpoint=migration['parent_checkpoint'], exact_state_restored=True,
        identical_eight_rank_training_trajectory_claimed=False, restore_receipts=restored,
        progress={'epoch': 30, 'step': 1, 'updates': 41210}))
    rows, info = report.training_rows(root, ranking, report.Evidence())
    assert rows[28]['effective_global_batch'] == 32 and rows[29]['effective_global_batch'] == 30
    assert rows[-1]['updates'] == 118525 and info['original_val_best_epoch'] == 51
    assert rows[0]['pure_p_input_alpha'] == 0 and rows[0]['phase_matched_input_alpha'] == 1
    records[-1]['validation']['gt_or_mask_used_as_input'] = True
    (root / 'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError, match='Pure-P validation'):
        report.training_rows(root, ranking, report.Evidence())
