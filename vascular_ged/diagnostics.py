"""Read-only model diagnostics before another label-generation experiment."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from .atomic import write_csv, write_json
from .metadata import _resolve
from .training import _atomic_text, _rank_metrics


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _metrics(rows: list[dict[str, str]]) -> dict:
    lower, upper, pred = (np.asarray([float(r[key]) for r in rows]) for key in ('lower_bound', 'upper_bound', 'prediction'))
    exact = np.asarray([r['is_exact'].lower() in {'true', '1'} for r in rows])
    violation = np.maximum(lower-pred, 0) + np.maximum(pred-upper, 0)
    bounded = ~exact
    spearman, _ = _rank_metrics(lower[exact], pred[exact]) if exact.sum() > 1 else (float('nan'), float('nan'))
    return {
        'rows': len(rows), 'exact_rows': int(exact.sum()), 'bounded_rows': int(bounded.sum()),
        'interval_loss': float(np.mean(violation**2)), 'mean_interval_violation': float(violation.mean()),
        'exact_mae': float(np.mean(abs(pred[exact]-lower[exact]))) if exact.any() else None,
        'exact_rmse': float(np.sqrt(np.mean((pred[exact]-lower[exact])**2))) if exact.any() else None,
        'exact_spearman': float(spearman) if np.isfinite(spearman) else None,
        'bounded_coverage': float(np.mean(violation[bounded] == 0)) if bounded.any() else None,
        'bounded_mean_violation': float(violation[bounded].mean()) if bounded.any() else None,
        'bounded_mean_width': float((upper-lower)[bounded].mean()) if bounded.any() else None,
    }


def test_pool_inventory(index: Path, patient_split: Path) -> dict:
    used = {r['patient_id'] for r in _read(patient_split)}
    by_split: dict[str, set[str]] = defaultdict(set)
    graph_counts: Counter = Counter()
    with index.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        patient_key = _resolve(fields, None, 'patient_id')
        split_key = _resolve(fields, None, 'relationformer_split', required=False)
        for row in reader:
            patient = row[patient_key].strip()
            split = row.get(split_key or '', 'train').strip().lower() or 'train'
            split = {'training': 'train', 'validation': 'val', 'testing': 'test'}.get(split, split)
            if not patient or split not in {'train', 'val', 'test'}:
                raise ValueError('invalid patient or split in test-pool inventory')
            by_split[split].add(patient)
            graph_counts[split] += 1
    candidates = set().union(*by_split.values()) - used
    return {
        'patch_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
        'used_greed_patient_count': len(used),
        'metadata_graph_counts': dict(graph_counts),
        'metadata_patient_counts': {k: len(v) for k, v in by_split.items()},
        'candidate_patient_ids_outside_greed_split': sorted(candidates),
        'candidate_count': len(candidates),
        'overlap_with_greed_by_relationformer_split': {k: sorted(v & used) for k, v in by_split.items()},
        'status': 'candidate_patients_require_provenance_review' if candidates else 'no_untouched_patients_in_patch_index',
        'note': 'Metadata inventory only. Candidates are not a verified independent test set. RelationFormer evaluation patients require an agreed evaluation protocol before use.',
    }


def diagnose_50k(experiment_root: str | Path, validation_labels: str | Path,
                 patient_split: str | Path, patch_index: str | Path, report_root: str | Path) -> dict:
    experiment, labels_path, split_path, report = map(Path, (experiment_root, validation_labels, patient_split, report_root))
    labels = _read(labels_path)
    by_id = {r['pair_id']: r for r in labels}
    if len(labels) != 2000 or len(by_id) != len(labels):
        raise ValueError('diagnostics require 2,000 unique validation labels')
    exact_ids = [r['pair_id'] for r in labels if r['is_exact'].lower() in {'true', '1'}]
    exact_cuts = np.quantile([float(by_id[k]['lower_bound']) for k in exact_ids], [1/3, 2/3])
    exact_id_set = set(exact_ids)
    bounded_rows = [r for r in labels if r['pair_id'] not in exact_id_set]
    width_cuts = np.quantile([float(r['upper_bound'])-float(r['lower_bound']) for r in bounded_rows], [1/3, 2/3]) if bounded_rows else []
    summaries, curves, breakdown, worst = [], [], [], []
    for seed in (161803, 271828, 314159):
        directory = experiment / f'pairs_50000/seed_{seed}'
        metadata = json.loads((directory / 'model_metadata.json').read_text())
        cfg = yaml.safe_load((directory / 'resolved_config.yaml').read_text())
        if metadata['training_seed'] != seed or metadata['train_pair_count'] != 50000:
            raise ValueError('diagnostics training run mismatch')
        for key, path in (('validation_labels_sha256', labels_path), ('patient_split_sha256', split_path)):
            if metadata[key] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'diagnostics input hash mismatch: {key}')
        history = _read(directory / 'training_history.csv')
        if not history or len(history) != metadata['epochs_completed'] or [int(r['epoch']) for r in history] != list(range(1, len(history)+1)):
            raise ValueError('training history is incomplete')
        losses = np.asarray([float(r['validation_interval_loss']) for r in history])
        train_losses = np.asarray([float(r['training_loss']) for r in history])
        if not (np.isfinite(losses).all() and np.isfinite(train_losses).all()):
            raise ValueError('non-finite training history')
        best_epoch = int(metadata['best_metrics']['epoch'])
        if not 1 <= best_epoch <= len(history):
            raise ValueError('best epoch is outside training history')
        window = min(15, max(1, len(history)//2))
        old_best = float(losses[:-window].min()) if len(losses) > window else float(losses[0])
        recent_best = float(losses[-window:].min())
        recent_improvement = (old_best-recent_best)/old_best if old_best > 0 else 0.0
        hit_cap = len(history) >= int(cfg['epochs']) and not metadata['stopped_early']
        review = hit_cap and (len(history)-best_epoch < window or recent_improvement > 0.01)
        summary = {'seed': seed, 'epochs_completed': len(history), 'epoch_limit': int(cfg['epochs']),
                   'best_epoch': best_epoch, 'stopped_early': bool(metadata['stopped_early']),
                   'hit_epoch_cap': hit_cap, 'recent_window_epochs': window,
                   'recent_best_relative_improvement': recent_improvement,
                   'final_training_loss': float(train_losses[-1]), 'final_validation_loss': float(losses[-1]),
                   'best_validation_loss': float(metadata['best_validation_interval_loss']),
                   'extend_training_review': bool(review)}
        summaries.append(summary)
        curves.extend(dict(seed=seed, **r) for r in history)
        predictions = _read(directory / 'validation_predictions.csv')
        if len(predictions) != 2000 or {r['pair_id'] for r in predictions} != set(by_id):
            raise ValueError('prediction rows do not match validation labels')
        normalized = []
        for prediction in predictions:
            source = by_id[prediction['pair_id']]
            for field in ('sample_a', 'sample_b', 'patient_a', 'patient_b', 'size_stratum', 'sampling_strategy', 'is_exact'):
                if prediction[field] != source[field]:
                    raise ValueError('prediction identity metadata mismatch')
            lower, upper, value = float(source['lower_bound']), float(source['upper_bound']), float(prediction['prediction'])
            if not all(np.isfinite([lower, upper, value])) or lower > upper:
                raise ValueError('invalid prediction or label bounds')
            normalized.append(dict(source, prediction=str(value)))
        groups = defaultdict(list)
        for row in normalized:
            groups['all'].append(row)
            groups[f"size:{row['size_stratum']}"].append(row)
            groups[f"strategy:{row['sampling_strategy']}"].append(row)
            groups[f"cell:{row['sampling_strategy']}:{row['size_stratum']}"].append(row)
            if row['is_exact'].lower() in {'true', '1'}:
                index = int(np.searchsorted(exact_cuts, float(row['lower_bound']), side='right'))
                groups[f'exact_ged:{("low", "medium", "high")[index]}'].append(row)
            else:
                width = float(row['upper_bound'])-float(row['lower_bound'])
                index = int(np.searchsorted(width_cuts, width, side='right'))
                groups[f'bounded_width:{("narrow", "medium", "wide")[index]}'].append(row)
        for name, rows in sorted(groups.items()):
            breakdown.append(dict(seed=seed, group=name, **_metrics(rows)))
        ranked = sorted(normalized, key=lambda r: max(float(r['lower_bound'])-float(r['prediction']), float(r['prediction'])-float(r['upper_bound']), 0), reverse=True)
        for row in ranked[:20]:
            worst.append(dict(seed=seed, pair_id=row['pair_id'], size_stratum=row['size_stratum'],
                              sampling_strategy=row['sampling_strategy'], is_exact=row['is_exact'],
                              lower_bound=row['lower_bound'], upper_bound=row['upper_bound'], prediction=row['prediction'],
                              interval_violation=max(float(row['lower_bound'])-float(row['prediction']), float(row['prediction'])-float(row['upper_bound']), 0)))
    inventory = test_pool_inventory(Path(patch_index), split_path)
    result = {'convergence': summaries, 'review_training_extension_before_100k': any(r['extend_training_review'] for r in summaries),
              'exact_ged_tertile_cuts': exact_cuts.tolist(), 'bounded_width_tertile_cuts': list(width_cuts),
              'test_pool_inventory': inventory,
              'interpretation': 'Epoch-cap flag is a review heuristic, not proof of convergence. GED-range groups use exact labels; bounded widths are analyzed separately.'}
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / 'diagnostics.json', result)
    write_csv(report / 'convergence.csv', summaries, list(summaries[0]))
    write_csv(report / 'epoch_curves.csv', curves, list(curves[0]))
    write_csv(report / 'validation_breakdown.csv', breakdown, list(breakdown[0]))
    write_csv(report / 'worst_validation_pairs.csv', worst, list(worst[0]))
    lines = ['# 50k GREED diagnostics before 100k', '', '| Seed | Epochs | Best epoch | Early stopped | Epoch cap | Final train loss | Final val loss | Review extension |', '|---:|---:|---:|---|---|---:|---:|---|']
    for r in summaries:
        lines.append(f"| {r['seed']} | {r['epochs_completed']} | {r['best_epoch']} | {r['stopped_early']} | {r['hit_epoch_cap']} | {r['final_training_loss']:.4f} | {r['final_validation_loss']:.4f} | {r['extend_training_review']} |")
    lines += ['', 'Extension review is flagged when training reaches its epoch limit with a recent best epoch or more than 1% recent best-loss improvement. It is a heuristic.', '',
              '| Group | Pairs per seed | Exact MAE (mean ± seed SD) | Interval violation (mean ± seed SD) | Bounded coverage (mean ± seed SD) | Mean bound width |', '|---|---:|---:|---:|---:|---:|']
    for name in sorted({r['group'] for r in breakdown}):
        group = [r for r in breakdown if r['group'] == name]
        def display(key):
            values = [r[key] for r in group if r[key] is not None]
            return f'{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}' if len(values) == 3 else 'n/a'
        lines.append(f"| {name} | {group[0]['rows']} | {display('exact_mae')} | {display('mean_interval_violation')} | {display('bounded_coverage')} | {display('bounded_mean_width')} |")
    lines += ['', f"Untouched patient candidates in patch metadata: {inventory['candidate_count']}.",
              f"Inventory status: {inventory['status']}.", inventory['note'], '',
              f"The existing {inventory['used_greed_patient_count']} GREED split patients have been allocated to training or validation. This job does not select a test set or compute new GED labels.", '']
    _atomic_text(report / 'diagnostics.md', '\n'.join(lines))
    return result
