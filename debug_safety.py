from src.safety import check
import json, pathlib

data = json.loads(pathlib.Path('fixtures/test_queries/safety_pairs.json').read_text())
for case in data['queries']:
    if case['should_block']:
        v = check(case['query'])
        if not v.blocked:
            print('NOT BLOCKED:', case['query'])