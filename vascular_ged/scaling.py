"""Extend the audited production dataset without replacing pilot artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from dataclasses import replace
from pathlib import Path

from .atomic import write_csv, write_json
from .metadata import load_patch_index
from .results import combine_partitions
from .sampling import MANIFEST_FIELDS, sample_pairs, write_manifest
from .split import read_patient_split


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def prepare_50k(dataset_root: str | Path, source_root: str | Path, output_root: str | Path) -> dict:
    return prepare_expansion(dataset_root, source_root, output_root, 8000, 50000)


def prepare_expansion(dataset_root: str | Path, source_root: str | Path, output_root: str | Path,
                      source_count: int, target_count: int, audit_root: str | Path | None = None) -> dict:
    if (source_count, target_count) not in {(8000, 50000), (50000, 100000)}:
        raise ValueError('supported expansions are 8k to 50k and 50k to 100k')
    dataset, source, output = Path(dataset_root).resolve(), Path(source_root).resolve(), Path(output_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('scaled output must be a separate sibling of the existing production root')
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing scaled experiment: {output}')
    index = dataset / 'patch_index.csv'
    audit = json.loads((Path(audit_root or source) / 'metadata/dataset_audit.json').read_text())
    if audit['dataset_root'] != str(dataset) or audit['patch_index_sha256'] != hashlib.sha256(index.read_bytes()).hexdigest():
        raise ValueError('dataset audit does not match the current dataset')
    records = load_patch_index(index)
    records = [replace(r,
        node_count=int(audit.get('normalized_node_count_corrections', {}).get(r.sample_id, r.node_count)),
        edge_count=int(audit.get('normalized_edge_count_corrections', {}).get(r.sample_id, r.edge_count)),
    ) for r in records]
    assignment, _ = read_patient_split(source / 'metadata/patient_split.csv')
    if set(assignment) != {r.patient_id for r in records}:
        raise ValueError('patient assignment does not cover the dataset')
    train_manifest = _read(source / 'manifests/greed_train_pairs.csv')
    validation_manifest = _read(source / 'manifests/greed_validation_pairs.csv')
    train_labels = _read(source / 'results/production/ged_labels_train.csv')
    validation_labels = _read(source / 'results/production/ged_labels_validation.csv')
    for manifest, labels, expected in ((train_manifest, train_labels, source_count), (validation_manifest, validation_labels, 2000)):
        if len(manifest) != expected or len(labels) != expected:
            raise ValueError('source manifest/label counts are incorrect')
        if len({r['pair_id'] for r in labels}) != expected:
            raise ValueError('duplicate source labels')
        by_id = {r['pair_id']: r for r in labels}
        for row in manifest:
            label = by_id.get(row['pair_id'])
            if label is None or any(row[field] != label[field] for field in MANIFEST_FIELDS):
                raise ValueError('source label and manifest metadata disagree')
            if label['solver_status'] not in {'ok', 'analytic_empty_graph'} or label['solver'] != 'f2' or float(label['time_limit_seconds']) != 60:
                raise ValueError('source labels are unsuccessful or use a different solver configuration')
            lower, upper = float(label['lower_bound']), float(label['upper_bound'])
            if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
                raise ValueError('source label has invalid bounds')
    with tempfile.TemporaryDirectory() as temporary:
        combine_partitions(source / 'results/production/ged_labels_train.csv',
                           source / 'results/production/ged_labels_validation.csv', Path(temporary) / 'source_labels.csv', source_count, 2000)
    dataset_id, cost_id = train_manifest[0]['dataset_id'], train_manifest[0]['cost_config_id']
    if cost_id != 'vascular_3d_raw_v1':
        raise ValueError('expansion requires the raw-coordinate cost configuration')
    # Validate the retained validation metadata against the same audited graphs.
    sample_pairs(records, assignment, 'validation', 2000, 271828, dataset_id, cost_id, initial_rows=validation_manifest)
    extended, availability = sample_pairs(records, assignment, 'train', target_count, 271828, dataset_id, cost_id, initial_rows=train_manifest)
    new_rows = extended[source_count:]
    output.mkdir(parents=True)
    write_manifest(output / 'manifests/greed_train_pairs.csv', extended)
    write_manifest(output / 'manifests/new_train_pairs.csv', new_rows)
    write_manifest(output / 'manifests/greed_validation_pairs.csv', validation_manifest)
    write_csv(output / f'results/production/train_shards/retained_{source_count}.csv', train_labels, list(train_labels[0]))
    write_csv(output / 'results/production/ged_labels_validation.csv', validation_labels, list(validation_labels[0]))
    split_path = source / 'metadata/patient_split.csv'
    write_csv(output / 'metadata/patient_split.csv', _read(split_path), list(_read(split_path)[0]))
    summary = {'train_pairs': target_count, 'retained_train_pairs': source_count, 'new_ged_pairs': len(new_rows),
               'validation_pairs': 2000, 'source_root': str(source), 'output_root': str(output),
               'patient_split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
               'source_train_labels_sha256': hashlib.sha256((source / 'results/production/ged_labels_train.csv').read_bytes()).hexdigest(),
               'source_validation_labels_sha256': hashlib.sha256((source / 'results/production/ged_labels_validation.csv').read_bytes()).hexdigest(),
               'sampling_seed': 271828, 'availability': availability}
    write_json(output / 'metadata/expansion.json', summary)
    return summary
