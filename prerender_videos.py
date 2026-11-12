"""Pre-render every Cond B segment video once, so all models receive
byte-identical video files.

Replays the deterministic sample selection (seed -> shuffle -> first N) for all
8 tasks x 3 seeds, dedupes segments by (video_id, start, end), and renders each
with write_segment_video_1hz() to data/videos_1hz/<video_id>_<start>_<end>.mp4.

A manifest CSV records file, sha256, frame count for every segment so that
byte-identity across runs can be verified.
"""

import argparse
import csv
import hashlib
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

from utils import get_frame_paths, write_segment_video_1hz, segment_video_name

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./data/videos_1hz')
    parser.add_argument('--limit', type=int, default=0, help='Render only the first N segments (for testing)')
    args = parser.parse_args()

    csv_path = '../shrec_empathic.csv'
    images_dir = '../shrec_empathic'
    data_path = './output_datasets'
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    fr = dict(zip(df['file_name'], df['framerate']))

    # Collect unique segments across all selected samples of all tasks/seeds
    segments = {}
    for task, (pickle_name, n) in TASKS.items():
        with open(os.path.join(data_path, pickle_name), 'rb') as f:
            dataset = pickle.load(f)
        for seed in SEEDS:
            random.seed(seed)
            d = dataset[:]
            random.shuffle(d)
            for s in d[:n]:
                key = segment_video_name(s['video_id'], s['timestamp']['start'], s['timestamp']['end'])
                if key not in segments:
                    segments[key] = (s['video_id'], s['timestamp']['start'], s['timestamp']['end'])

    keys = sorted(segments.keys())
    if args.limit:
        keys = keys[:args.limit]
    print('Rendering {} unique segments to {}'.format(len(keys), out_dir))

    manifest = []
    for idx, key in enumerate(keys):
        video_id, start, end = segments[key]
        frames_dir = os.path.join(images_dir, video_id)
        rate = fr[video_id]
        img_paths = get_frame_paths(frames_dir, start * rate, end * rate)
        # Same fallback as main_vlm_exp.py: if no frames in segment range,
        # use the nearest frame to the segment midpoint.
        if len(img_paths) == 0:
            import cv2  # noqa: F401 (fallback mirrors main_vlm_exp.py)
            all_pngs = sorted(
                [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                 if f.endswith('.png') and f[:-4].isdigit()],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
            if len(all_pngs) == 0:
                print('[WARN] No frames at all for {} - skipping'.format(key))
                continue
            mid_idx = int((start + end) / 2 * rate)
            img_paths = [min(all_pngs, key=lambda p: abs(int(os.path.splitext(os.path.basename(p))[0]) - mid_idx))]
        out_path = os.path.join(out_dir, key)
        skip = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        if not skip:
            write_segment_video_1hz(img_paths, out_dir, output_path=out_path)
        with open(out_path, 'rb') as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        manifest.append({
            'file': key,
            'sha256': digest,
            'frames': len(img_paths),
            'video_id': video_id,
            'start': start,
            'end': end,
        })
        if (idx + 1) % 200 == 0:
            print('  {}/{}'.format(idx + 1, len(keys)))

    manifest_path = os.path.join(out_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print('Manifest: {} ({} segments)'.format(manifest_path, len(manifest)))


if __name__ == '__main__':
    main()
