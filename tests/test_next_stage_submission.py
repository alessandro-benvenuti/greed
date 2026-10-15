import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize('fail_at', [0, 3])
def test_submission_dependencies_and_partial_failure_record(tmp_path, fail_at):
    repo = Path(__file__).parents[1]
    root = tmp_path / '50k'
    (root / 'metadata').mkdir(parents=True)
    (root / 'results/production').mkdir(parents=True)
    (root / 'results/production/ged_labels_train.csv').touch()
    for seed in (161803, 271828):
        directory = root / f'checkpoints/greed_3d_50k/pairs_50000/seed_{seed}'
        directory.mkdir(parents=True)
        (directory / 'latest_checkpoint.pt').touch()
    binary = tmp_path / 'bin'; binary.mkdir()
    fake = binary / 'sbatch'
    fake.write_text('#!/usr/bin/env bash\n'
                    'count=0\n'
                    'if [[ -f "$MOCK_COUNT" ]]; then read -r count < "$MOCK_COUNT"; fi\n'
                    'count=$((count+1))\n'
                    'printf "%s\\n" "$count" > "$MOCK_COUNT"\n'
                    'printf "%s\\n" "$*" >> "$MOCK_ARGS"\n'
                    'if [[ "$count" == "$MOCK_FAIL_AT" ]]; then exit 1; fi\n'
                    'printf "%s\\n" "$((1000+count))"\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", GREED_REPOSITORY=str(repo),
               GREED_50K_ROOT=str(root), GREED_100K_ROOT=str(tmp_path / '100k'),
               MOCK_COUNT=str(tmp_path / 'count'), MOCK_ARGS=str(tmp_path / 'args'), MOCK_FAIL_AT=str(fail_at))
    result = subprocess.run(['bash', str(repo / 'scripts/jean_zay/submit_next_stage.sh')], env=env, capture_output=True, text=True)
    record = (root / 'metadata/next_stage_submission.env').read_text()
    args = (tmp_path / 'args').read_text().splitlines()
    if fail_at:
        assert result.returncode != 0
        assert len(args) == 3
        assert 'EXTEND_REPORT_JOB_ID=1002' in record
        assert 'GED_100K_JOB_ID=' not in record
    else:
        assert result.returncode == 0, result.stderr
        assert len(args) == 7
        assert '--dependency=afterok:1001' in args[1]
        assert '--dependency=afterok:1004' in args[4]
        assert '--dependency=afterok:1005' in args[5]
        assert '--dependency=afterok:1006' in args[6]
        assert 'FINALIZE_100K_JOB_ID=1007' in record
    again = subprocess.run(['bash', str(repo / 'scripts/jean_zay/submit_next_stage.sh')], env=env, capture_output=True)
    assert again.returncode != 0
    assert (tmp_path / 'args').read_text().splitlines() == args
