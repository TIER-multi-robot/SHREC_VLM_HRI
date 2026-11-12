"""Build the ER sample map: recover sample_id and video_id (session) for every
ER output row by replaying the deterministic sample selection.

Selection in main_vlm_exp.py: random.seed(seed); random.shuffle(pickle dataset);
take the first --sample_size samples. Identical across conditions.

Output: analysis/er_sample_map.csv
Columns: task, condition, seed, sample_index, sample_id, video_id, start, end
sample_index is the 0-based row index in the run's output JSON, so analysis can
join ER outputs to sessions by (task, condition, seed, sample_index).
"""

import csv
import os
import pickle
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SHREC = os.path.dirname(HERE)
DATA_PATH = os.path.join(SHREC, 'output_datasets')
OUT_CSV = os.path.join(HERE, 'er_sample_map.csv')

TASKS = {
    'detection': ('shrec_empathic_detection.pickle', 200),
    'detection_error_only': ('shrec_empathic_detection_error_only.pickle', 200),
    'attribute': ('shrec_empathic_attribute.pickle', 100),
    'attribute_agreed_multiple_subj': ('shrec_empathic_attribute_agreed_multiple_subj.pickle', 100),
    'rationale': ('shrec_empathic_rationale.pickle', 200),
    'correction': ('shrec_empathic_correction.pickle', 200),
    'pre': ('shrec_empathic_pre.pickle', 100),
    'post': ('shrec_empathic_post.pickle', 100),
}
SEEDS = (0, 1, 2)
CONDITIONS = ('A', 'B')


def main():
    rows = []
    for task, (pickle_name, n) in TASKS.items():
        with open(os.path.join(DATA_PATH, pickle_name), 'rb') as f:
            dataset = pickle.load(f)
        for seed in SEEDS:
            random.seed(seed)
            shuffled = dataset[:]
            random.shuffle(shuffled)
            for idx, sample in enumerate(shuffled[:n]):
                for cond in CONDITIONS:
                    rows.append({
                        'task': task,
                        'condition': cond,
                        'seed': seed,
                        'sample_index': idx,
                        'sample_id': sample.get('id') or '',
                        'video_id': sample.get('video_id') or sample.get('id') or '',
                        'start': sample['timestamp']['start'],
                        'end': sample['timestamp']['end'],
                    })
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote {} rows to {}'.format(len(rows), OUT_CSV))


if __name__ == '__main__':
    main()
