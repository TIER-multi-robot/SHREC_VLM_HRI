import json, csv, pickle, random
from collections import Counter

def load_pydantic(model, task_pickle, run):
    p = 'output_pydantic/{}_{}_shrec_empathic_{}.pickle.json'.format(model, run, task_pickle)
    with open(p) as f: return json.load(f)

# --- detection_error_only: GPT4oMiniCheck vs ER ---
print('=== detection_error_only ===')
for model, run in [('GPT4oMiniCheck', 'condA_seed0'), ('GeminiRoboticsER', 'condA_seed0'), ('GeminiRoboticsER', 'condB_seed0')]:
    data = load_pydantic(model, 'detection_error_only', run)
    golds, preds = [], []
    for s in data:
        answer = s['response'][0] if s['response'] else ''
        gold = 'error' if s['label'] is True else 'competence'
        pred = None
        if 'social competence' in answer.lower() or 'A' in answer: pred = 'competence'
        elif 'social error' in answer.lower() or 'B' in answer: pred = 'error'
        elif 'C' in answer or 'none' in answer.lower(): pred = 'none'
        golds.append(gold); preds.append(pred)
    cm = Counter((g, p) for g, p in zip(golds, preds))
    acc = sum(1 for g, p in zip(golds, preds) if g == p) / len(data)
    print(model, run, 'acc={:.3f}'.format(acc), 'gold dist:', Counter(golds), 'confusion:', dict(cm))

# label distribution of the selected error_only samples (seed 0 replay)
with open('output_datasets/shrec_empathic_detection_error_only.pickle', 'rb') as f: ds = pickle.load(f)
random.seed('0'); d = ds[:]; random.shuffle(d); d = d[:200]
print('error_only selected-200 label dist:', Counter('error' if s['error'] is True else 'competence' if s['error'] is False else 'none' for s in d))
print('error_only FULL pickle label dist:', Counter('error' if s['error'] is True else 'competence' if s['error'] is False else 'none' for s in ds))

# 10 examples for GPT4oMiniCheck detection_error_only
data = load_pydantic('GPT4oMiniCheck', 'detection_error_only', 'condA_seed0')
print('--- 10 examples (raw -> parsed -> gold) ---')
for s in data[:10]:
    gold = 'error' if s['label'] is True else 'competence'
    print('raw={:.40}... parsed={} gold={}'.format(s['pre-response'].replace(chr(10), ' ')[:60], s['response'], gold))

# --- correction ---
print('=== correction ===')
for model, run in [('GPT4oMiniCheck', 'condA_seed0'), ('GeminiRoboticsER', 'condA_seed0')]:
    data = load_pydantic(model, 'correction', run)
    correct = 0; n_parsable = 0
    digits = Counter()
    for s in data:
        answer = s['response'][0] if s['response'] else ''
        final = None
        for v in ['1', '2', '3', '4', '5']:
            if answer.find(v) != -1: final = v
        if final is None: continue
        n_parsable += 1
        digits[final] += 1
        if final in s['label']: correct += 1
    print(model, run, 'acc={:.3f}'.format(correct/len(data)), 'unparsable:', len(data)-n_parsable, 'pred digit dist:', dict(digits))
    golds = Counter(s['label'] for s in data)
    print(' gold dist:', dict(golds))
print('--- 10 correction examples (GPT4oMiniCheck) ---')
data = load_pydantic('GPT4oMiniCheck', 'correction', 'condA_seed0')
for s in data[:10]:
    print('raw={:.50}... parsed={} gold={}'.format(s['pre-response'].replace(chr(10), ' ')[:70], s['response'], s['label']))
