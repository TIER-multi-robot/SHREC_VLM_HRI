"""Gemini 3.6 Flash full-job orchestrator (48 runs: 8 tasks x 2 conds x 3 seeds).

- Resumable: skips runs whose output JSON already has the full sample count.
- Up to 4 parallel workers (subprocesses of main_vlm_exp.py).
- Atomic outputs: main_vlm_exp.py now writes via temp file + os.replace.
- Per-run error gate: >1% [ERROR] responses stops the whole job.
- Hard stop on any 402 / RESOURCE_EXHAUSTED error (out of credits): a
  watchdog scans the sidecar JSONLs and kills running subprocesses.

Usage (from SHREC/):
  python run_flash_job.py --workers 4
  python run_flash_job.py --dry-run
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

MODEL_KEY = 'Gemini35Flash'
MODEL_ID = 'gemini-3.6-flash'
BASE_CMD = [
    sys.executable, 'main_vlm_exp.py',
    '--context_window', '15',
    '--model', MODEL_KEY,
    '--data_path', './output_datasets',
    '--csv_path', '../shrec_empathic.csv',
    '--images_dir', '../shrec_empathic',
    '--videos_dir', './data/videos_1hz',
    '--backend', 'gemini',
    '--model_id', MODEL_ID,
]

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
CONDITIONS = {'A': False, 'B': True}

OUTPUT_DIR = './output'
CALLS_DIR = './logs/calls'
JOB_LOG = './logs/flash_job.log'
PROGRESS_FILE = './logs/flash_job_progress.json'

_lock = threading.Lock()
_procs = []          # running subprocess.Popen objects
_state = {}          # run_key -> status
_stop = threading.Event()


def output_file(task_pickle, cond, seed):
    return os.path.join(OUTPUT_DIR, '{}_cond{}_seed{}_{}.json'.format(MODEL_KEY, cond, seed, task_pickle))


def sidecar_file(task_pickle, cond, seed):
    return os.path.join(CALLS_DIR, '{}_cond{}_seed{}_{}.jsonl'.format(MODEL_KEY, cond, seed, task_pickle))


def is_complete(task_pickle, n, cond, seed):
    f = output_file(task_pickle, cond, seed)
    if not os.path.exists(f):
        return False
    try:
        return len(json.load(open(f))) >= n
    except Exception:
        return False


def log(line):
    with _lock:
        with open(JOB_LOG, 'a') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + line + '\n')
    print(line, flush=True)


def save_progress():
    with _lock:
        with open(PROGRESS_FILE, 'w') as f:
            json.dump({'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'runs': _state}, f, indent=2)


def error_rate(out_file):
    try:
        rows = json.load(open(out_file))
    except Exception:
        return 0.0, 0
    errs = sum(1 for r in rows if r['response'] == '[ERROR]')
    return errs / max(len(rows), 1), errs


def scan_for_402():
    """Return True if any Flash sidecar error mentions 402 / RESOURCE_EXHAUSTED."""
    for f in glob.glob(os.path.join(CALLS_DIR, MODEL_KEY + '_*.jsonl')):
        try:
            with open(f) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    e = rec.get('error')
                    if e and ('RESOURCE_EXHAUSTED' in e or '402' in e):
                        return f, e
        except OSError:
            continue
    return None


def watchdog():
    while not _stop.is_set():
        hit = scan_for_402()
        if hit:
            log('ABORT: 402/RESOURCE_EXHAUSTED detected in {} -> {}'.format(hit[0], hit[1]))
            _stop.set()
            with _lock:
                for p in _procs:
                    try:
                        p.kill()
                    except Exception:
                        pass
            return
        time.sleep(15)


def run_one(run_key, task_name, task_pickle, n, cond, seed, has_video):
    f = output_file(task_pickle, cond, seed)
    cmd = list(BASE_CMD) + [
        '--task_type', task_pickle,
        '--seed', str(seed),
        '--run_name', 'cond{}_seed{}'.format(cond, seed),
        '--sample_size', str(n),
    ]
    if has_video:
        cmd += ['--video', '--er_video_mode', 'video']

    with _lock:
        _state[run_key] = 'RUNNING'
    save_progress()
    log('START {} (n={})'.format(run_key, n))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=open('./logs/flash_{}_{}_seed{}.log'.format(task_name, cond, seed), 'w'),
                            stderr=subprocess.STDOUT)
    with _lock:
        _procs.append(proc)
    rc = proc.wait()
    with _lock:
        if proc in _procs:
            _procs.remove(proc)

    if _stop.is_set():
        _state[run_key] = 'ABORTED'
        save_progress()
        return

    rate, errs = error_rate(f)
    if rc != 0 or not os.path.exists(f):
        log('FAILED {} rc={} ({:.0f}s)'.format(run_key, rc, time.time() - t0))
        _state[run_key] = 'FAILED'
        _stop.set()  # rc!=0 means the run crashed; stop launching new work
    elif rate > 0.01:
        log('ERROR-GATE {} errors={}/{} ({:.1%} > 1%) — stopping job'.format(run_key, errs, n, rate))
        _state[run_key] = 'ERROR_GATE'
        _stop.set()
    else:
        log('DONE {} ({:.0f}s) errors={}'.format(run_key, time.time() - t0, errs))
        _state[run_key] = 'DONE'
    save_progress()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    os.makedirs('./logs', exist_ok=True)
    run_list = []
    for cond in CONDITIONS:                      # A block first, then B
        for task_name, (task_pickle, n) in TASKS.items():
            for seed in SEEDS:
                run_list.append((task_name, task_pickle, n, cond, seed))

    pending = []
    for task_name, task_pickle, n, cond, seed in run_list:
        key = '{}_{}_seed{}'.format(task_name, cond, seed)
        if is_complete(task_pickle, n, cond, seed):
            _state[key] = 'DONE(pre-existing)'
            log('SKIP {} (already complete)'.format(key))
        else:
            pending.append((key, task_name, task_pickle, n, cond, seed))

    log('Job plan: {} runs total, {} pending, {} pre-complete'.format(len(run_list), len(pending), len(run_list) - len(pending)))
    if args.dry_run:
        for key, *_ in pending:
            print('  pending:', key)
        save_progress()
        return

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for key, task_name, task_pickle, n, cond, seed in pending:
            if _stop.is_set():
                _state[key] = 'NOT_STARTED(stop)'
                continue
            futures.append(ex.submit(run_one, key, task_name, task_pickle, n, cond, seed, CONDITIONS[cond]))
        for fu in futures:
            fu.result()

    _stop.set()
    save_progress()
    log('Job finished. States: ' + json.dumps({k: sum(1 for v in _state.values() if v == k) for k in set(_state.values())}))


if __name__ == '__main__':
    main()
