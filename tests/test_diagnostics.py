import hashlib
import json

import pytest

from vascular_ged.atomic import write_csv, write_json
from vascular_ged.diagnostics import diagnose_50k, test_pool_inventory as inventory


def test_inventory_excludes_all_used_patients_even_in_external_splits(tmp_path):
    index, split = tmp_path / 'index.csv', tmp_path / 'split.csv'
    index.write_text('patient_id,split\n1,train\n2,validation\n3,test\n1,test\n')
    split.write_text('patient_id,greed_split\n1,train\n2,validation\n')
    result = inventory(index, split)
    assert result['candidate_patient_ids_outside_greed_split'] == ['3']
    assert result['overlap_with_greed_by_relationformer_split']['test'] == ['1']


def test_diagnostics_recompute_errors_and_flag_epoch_cap(tmp_path):
    root, reports = tmp_path / 'models', tmp_path / 'reports'
    labels, split, index = (tmp_path / name for name in ('labels.csv', 'split.csv', 'index.csv'))
    split.write_text('patient_id,greed_split\n1,train\n2,validation\n')
    index.write_text('patient_id,split\n1,train\n2,train\n')
    rows = [dict(pair_id=str(i), lower_bound=str(i+1), upper_bound=str(i+1 if i%2 == 0 else i+3),
                 is_exact=str(i%2 == 0), sample_a=f'a{i}', sample_b=f'b{i}', patient_a='2', patient_b='3',
                 size_stratum=('small', 'medium', 'large')[i%3],
                 sampling_strategy=('corresponding_region', 'size_matched', 'unrestricted_random')[i%3]) for i in range(2000)]
    write_csv(labels, rows, list(rows[0]))
    for seed in (161803, 271828, 314159):
        directory = root / f'pairs_50000/seed_{seed}'
        write_json(directory / 'model_metadata.json', dict(training_seed=seed, train_pair_count=50000,
                   validation_labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
                   patient_split_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
                   epochs_completed=20, stopped_early=False, best_metrics={'epoch': 20}, best_validation_interval_loss=1.0))
        (directory / 'resolved_config.yaml').write_text('epochs: 20\n')
        history = [dict(epoch=i, training_loss=21-i, validation_interval_loss=21-i) for i in range(1, 21)]
        write_csv(directory / 'training_history.csv', history, list(history[0]))
        predictions = [dict(r, prediction=float(r['lower_bound'])+1, interval_violation=999) for r in rows]
        write_csv(directory / 'validation_predictions.csv', predictions, list(predictions[0]))
    result = diagnose_50k(root, labels, split, index, reports)
    assert result['review_training_extension_before_100k'] is True
    assert result['test_pool_inventory']['candidate_count'] == 0
    assert (reports / 'diagnostics.md').is_file()
    with (reports / 'validation_breakdown.csv').open() as handle:
        import csv
        all_rows = [r for r in csv.DictReader(handle) if r['group'] == 'all']
    assert float(all_rows[0]['exact_mae']) == pytest.approx(1)
    assert float(all_rows[0]['bounded_coverage']) == pytest.approx(1)
    for seed in (161803, 271828, 314159):
        path = root / f'pairs_50000/seed_{seed}/model_metadata.json'
        metadata = json.loads(path.read_text())
        metadata['stopped_early'] = True
        write_json(path, metadata)
    assert diagnose_50k(root, labels, split, index, reports)['review_training_extension_before_100k'] is False
    metadata_path = root / 'pairs_50000/seed_161803/model_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['validation_labels_sha256'] = 'wrong'
    write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match='hash mismatch'):
        diagnose_50k(root, labels, split, index, reports)
