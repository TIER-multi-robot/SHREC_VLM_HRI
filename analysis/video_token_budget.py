"""Compute visual-token distribution across all 1,804 segments under a proposed
explicit pixel budget, and count samples affected by 1-frame segments per task.

Qwen3-VL visual tokens = total_video_pixels / (patch 16 x merge 2)^2 / temporal 2
= pixels / 2048 (after rounding to factor 32 and even frame count).
"""
import csv, pickle, random, statistics, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from utils import segment_video_name

BUDGET = 49152000  # 24,000 visual tokens max; leaves ~7k for text+output in 32k ctx

with open('data/videos_1hz/manifest.csv') as f:
    rows = list(csv.DictReader(f))

# verify frame dims on a sample
dims = set()
for r in rows[::300]:
    cap = cv2.VideoCapture('data/videos_1hz/' + r['file'])
    ret, frame = cap.read()
    if ret: dims.add((frame.shape[1], frame.shape[0]))  # (w, h)
    cap.release()
print('distinct frame dims (w,h):', dims)

native = 896*512  # verified dataset frame size
tokens = []
for r in rows:
    t = int(r['frames'])
    px = min(t * native, BUDGET)
    tokens.append(px / 2048)
tokens.sort()
print('visual tokens per video: min={:.0f} median={:.0f} max={:.0f} mean={:.0f}'.format(
    tokens[0], tokens[len(tokens)//2], tokens[-1], statistics.mean(tokens)))
n_native = sum(1 for r in rows if int(r['frames']) * native <= BUDGET)
print('segments at native resolution:', n_native, 'of', len(rows))
t573 = min(573*native, BUDGET) / 2048
print('573-frame segment: {} tokens; effective per-frame resolution ~{:.0f} px (native 458,752)'.format(
    int(t573), BUDGET / 573))
print('per-frame resolution short segments (<=107 frames): native 896x512 = 458,752 px')

# samples affected by 1-frame segments per task (selection replay)
TASKS = {'detection': 200, 'detection_error_only': 200, 'attribute': 100,
         'attribute_agreed_multiple_subj': 100, 'rationale': 200, 'correction': 200,
         'pre': 100, 'post': 100}
one_files = {r['file'] for r in rows if int(r['frames']) == 1}
from utils import segment_video_name
print('--- 1-frame samples per task (summed over cond A/B and seeds 0-2) ---')
for task, n in TASKS.items():
    with open('output_datasets/shrec_empathic_{}.pickle'.format(task), 'rb') as f:
        ds = pickle.load(f)
    cnt = tot = 0
    for seed in (0, 1, 2):
        random.seed(str(seed)); d = ds[:]; random.shuffle(d); d = d[:n]
        for s in d:
            tot += 1
            key = segment_video_name(s['video_id'], s['timestamp']['start'], s['timestamp']['end'])
            if key in one_files: cnt += 1
    print('{}: {}/{} draws are 1-frame segments'.format(task, cnt, tot))
