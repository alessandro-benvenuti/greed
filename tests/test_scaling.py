import csv
import hashlib

import pytest

from vascular_ged.atomic import write_csv, write_json
from vascular_ged.metadata import GraphRecord
from vascular_ged.sampling import sample_pairs, write_manifest
from vascular_ged.scaling import prepare_50k, prepare_expansion
from vascular_ged.split import patient_split, write_patient_split


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def test_prepare_50k_preserves_labels_validation_and_split(tmp_path, monkeypatch):
    dataset, source, output = (tmp_path / name for name in ('dataset', 'original', 'scaled'))
    dataset.mkdir()
    index = dataset / 'patch_index.csv'
    index.write_text('fixture')
    records = [GraphRecord(f'p{p}_x{x}', str(p), str(x), f'train/vtp/p{p}_x{x}.vtp', 2+x, 1+x)
               for p in range(95) for x in range(20)]
    monkeypatch.setattr('vascular_ged.scaling.load_patch_index', lambda _: records)
    assignment = patient_split(records, 314159)
    write_patient_split(source / 'metadata/patient_split.csv', assignment, 314159)
    write_json(source / 'metadata/dataset_audit.json', {
        'dataset_root': str(dataset.resolve()),
        'patch_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
    })
    for split, count in (('train', 8000), ('validation', 2000)):
        rows, _ = sample_pairs(records, assignment, split, count, 271828, 'dataset', 'vascular_3d_raw_v1')
        write_manifest(source / f'manifests/greed_{split}_pairs.csv', rows)
        labels = [dict(r, solver_status='ok', solver='f2', time_limit_seconds=60,
                       lower_bound=1, upper_bound=1, is_exact=True) for r in rows]
        write_csv(source / f'results/production/ged_labels_{split}.csv', labels, list(labels[0]))
    result = prepare_50k(dataset, source, output)
    assert result['new_ged_pairs'] == 42000
    old = read(source / 'manifests/greed_train_pairs.csv')
    expanded = read(output / 'manifests/greed_train_pairs.csv')
    new = read(output / 'manifests/new_train_pairs.csv')
    assert expanded[:8000] == old
    assert len(expanded) == 50000
    assert len(new) == 42000
    assert {r['pair_id'] for r in old}.isdisjoint(r['pair_id'] for r in new)
    assert read(source / 'results/production/ged_labels_validation.csv') == read(output / 'results/production/ged_labels_validation.csv')
    assert read(source / 'metadata/patient_split.csv') == read(output / 'metadata/patient_split.csv')
    assert read(source / 'results/production/ged_labels_train.csv') == read(output / 'results/production/train_shards/retained_8000.csv')
    with pytest.raises(FileExistsError):
        prepare_50k(dataset, source, output)
    labels_50k = [dict(r, solver_status='ok', solver='f2', time_limit_seconds=60,
                       lower_bound=1, upper_bound=1, is_exact=True) for r in expanded]
    write_csv(output / 'results/production/ged_labels_train.csv', labels_50k, list(labels_50k[0]))
    output_100k = tmp_path / 'scaled100'
    expansion = prepare_expansion(dataset, output, output_100k, 50000, 100000, audit_root=source)
    assert expansion['new_ged_pairs'] == 50000
    full = read(output_100k / 'manifests/greed_train_pairs.csv')
    assert len(full) == 100000
    assert full[:50000] == expanded
    assert len({r['pair_id'] for r in full}) == 100000
    from collections import Counter
    cells = Counter((r['sampling_strategy'], r['size_stratum']) for r in full)
    assert max(cells.values()) - min(cells.values()) == 1
    assert read(output_100k / 'results/production/train_shards/retained_50000.csv') == read(output / 'results/production/ged_labels_train.csv')
    assert read(output_100k / 'results/production/ged_labels_validation.csv') == read(output / 'results/production/ged_labels_validation.csv')
