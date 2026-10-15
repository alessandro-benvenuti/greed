import copy
import hashlib
import json

import pytest

torch = pytest.importorskip('torch')

from vascular_ged.training import _initialize_continuation


def source_run(tmp_path):
    source, output = tmp_path / 'original', tmp_path / 'extended'
    source.mkdir(); output.mkdir()
    config = dict(epochs=100, output_dir=str(source), report_path=str(source / 'report.md'),
                  early_stopping_patience=15, learning_rate=0.001, seed=161803)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(torch.ones(1, 1)).sum().backward(); optimizer.step()
    state = dict(epoch=100, config=config, input_hashes={'labels': 'same'},
                 epochs_without_improvement=8, best_metrics={'epoch': 92},
                 model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                 torch_rng_state=torch.get_rng_state(), history=[{'epoch': 100}])
    torch.save(state, source / 'latest_checkpoint.pt')
    best = copy.deepcopy(state); best['epoch'] = 92
    torch.save(best, source / 'best_model.pt')
    (source / 'validation_predictions.csv').write_text('pair_id,prediction\na,1\n')
    target = dict(config, epochs=200, output_dir=str(output), report_path=str(output / 'report.md'))
    return source, output, target


def test_continuation_preserves_optimizer_rng_best_and_original_files(tmp_path):
    source, output, target = source_run(tmp_path)
    old_bytes = (source / 'latest_checkpoint.pt').read_bytes()
    _initialize_continuation(torch, source, output, target, {'labels': 'same'})
    old = torch.load(source / 'latest_checkpoint.pt', weights_only=False)
    new = torch.load(output / 'latest_checkpoint.pt', weights_only=False)
    assert new['config'] == target
    assert new['epoch'] == 100
    assert new['epochs_without_improvement'] == 8
    assert new['history'] == old['history']
    assert torch.equal(new['torch_rng_state'], old['torch_rng_state'])
    for key, state in old['optimizer_state']['state'].items():
        for name, value in state.items():
            assert torch.equal(new['optimizer_state']['state'][key][name], value)
    assert torch.load(output / 'best_model.pt', weights_only=False)['epoch'] == 92
    assert (output / 'validation_predictions.csv').read_bytes() == (source / 'validation_predictions.csv').read_bytes()
    assert (source / 'latest_checkpoint.pt').read_bytes() == old_bytes
    assert json.loads((output / 'continuation.json').read_text())['source_checkpoint_sha256'] == hashlib.sha256(old_bytes).hexdigest()


@pytest.mark.parametrize('change', ['learning_rate', 'labels', 'early_stopped', 'same_output'])
def test_continuation_rejects_changes_and_stopped_sources(tmp_path, change):
    source, output, target = source_run(tmp_path)
    hashes = {'labels': 'same'}
    if change == 'learning_rate':
        target['learning_rate'] = 0.1
    elif change == 'labels':
        hashes['labels'] = 'different'
    elif change == 'same_output':
        output = source
    else:
        state = torch.load(source / 'latest_checkpoint.pt', weights_only=False)
        state['epochs_without_improvement'] = 15
        torch.save(state, source / 'latest_checkpoint.pt')
    with pytest.raises(ValueError):
        _initialize_continuation(torch, source, output, target, hashes)
    if output != source:
        assert not list(output.iterdir())


def test_train_continues_in_separate_directory_and_resumes_that_directory(tmp_path):
    pytest.importorskip('torch_geometric')
    import shutil
    import yaml
    from pathlib import Path
    from vascular_ged.atomic import write_csv
    from vascular_ged.training import train

    repository = Path(__file__).parents[1]
    dataset = tmp_path / 'dataset/train/vtp'
    dataset.mkdir(parents=True)
    rows = []
    for partition, patients in (('train', ('1', '2')), ('validation', ('3', '4'))):
        for patient in patients:
            shutil.copyfile(repository / 'tests/fixtures/line_dhw.vtp', dataset / f'{patient}.vtp')
        a, b = patients
        row = dict(pair_id=partition, greed_split=partition, patient_a=a, patient_b=b,
                   graph_a_relative_path=f'train/vtp/{a}.vtp', graph_b_relative_path=f'train/vtp/{b}.vtp',
                   node_count_a=3, node_count_b=3, edge_count_a=2, edge_count_b=2,
                   lower_bound=1, upper_bound=1, is_exact=True, solver_status='ok',
                   cost_config_id='vascular_3d_raw_v1', size_stratum='small')
        write_csv(tmp_path / f'{partition}.csv', [row], list(row))
        rows.extend(dict(patient_id=p, greed_split=partition, relationformer_split='train') for p in patients)
    write_csv(tmp_path / 'split.csv', rows, list(rows[0]))
    original, continued = tmp_path / 'run', tmp_path / 'continuation'
    cfg = yaml.safe_load((repository / 'configs/greed_training_config.yaml').read_text())
    cfg.update(dataset_root=str(tmp_path / 'dataset'), train_labels=str(tmp_path / 'train.csv'),
               validation_labels=str(tmp_path / 'validation.csv'), patient_split=str(tmp_path / 'split.csv'),
               cost_config_path=str(repository / 'configs/cost_config.yaml'), train_count=1, validation_count=1,
               epochs=1, output_dir=str(original), report_path=str(original / 'report.md'), workers=0, torch_threads=1)
    cfg['model'].update(layers=2, hidden_dim=8, embedding_dim=8)
    path = tmp_path / 'config.yaml'; path.write_text(yaml.safe_dump(cfg))
    train(path)
    source_bytes = (original / 'latest_checkpoint.pt').read_bytes()
    train(original / 'resolved_config.yaml', epochs=2, output_dir=continued, resume_from=original)
    assert json.loads((continued / 'model_metadata.json').read_text())['epochs_completed'] == 2
    assert (original / 'latest_checkpoint.pt').read_bytes() == source_bytes
    train(original / 'resolved_config.yaml', epochs=2, output_dir=continued, resume_from=original)
    assert json.loads((continued / 'model_metadata.json').read_text())['epochs_completed'] == 2
