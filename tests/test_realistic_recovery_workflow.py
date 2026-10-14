from pathlib import Path
import yaml


def test_recovery_only_aggregates_within_remaining_stage_budget():
    root=Path(__file__).parents[1];p=root/'.github/workflows/realistic-interaction-recovery.yml'
    assert p.exists(),'no-fit recovery workflow missing'
    v=yaml.safe_load(p.read_text());assert len(v['jobs'])==1
    j=v['jobs']['recover'];assert j['timeout-minutes']==45
    # Original aggregate lasted13m01s; combined <= original60-minute stage cap.
    assert 13*60+1+45*60<=60*60
    commands='\n'.join(str(s.get('run','')) for s in j['steps'])
    assert 'realistic_interaction_recovery.py' in commands
    assert ' shard ' not in commands and 'freeze --write' not in commands
    assert '38482368226' in str(j['steps'])
