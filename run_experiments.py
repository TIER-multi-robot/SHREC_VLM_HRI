"""
SHREC Experiment Runner — Gemini Robotics-ER
=============================================
Runs 8 tasks x 2 conditions x 3 seeds = 48 experiment configurations.

Conditions:
  A = Language-only (transcript, no video)
  B = Language + Visual (1Hz stitched video via Files API)

Tasks & sample sizes:
  detection                    (200)   3-class: competence / error / none
  detection_error_only         (200)   binary: error / competence
  attribute                    (100)   7-class multi-label
  attribute_agreed_multiple    (100)   binary: True / False
  rationale                    (200)   5-way multiple choice
  correction                   (200)   5-way multiple choice
  pre                          (100)   5-way multiple choice
  post                         (100)   5-way multiple choice

Usage:
  python run_experiments.py                          # run all 48
  python run_experiments.py --dry-run                # print commands only
  python run_experiments.py --task detection         # one task
  python run_experiments.py --seed 0                 # one seed
  python run_experiments.py --condition A            # one condition
  python run_experiments.py --skip-eval              # skip pydantic parsing + eval
  python run_experiments.py --sample-override 2      # override all sample sizes (for testing)
"""

import subprocess
import os
import sys
import json
import time
import argparse
from datetime import datetime

# ============================================================
# Configuration
# ============================================================

MODEL = 'GeminiRoboticsER'
CSV_PATH = '../shrec_empathic.csv'
IMAGES_DIR = '../shrec_empathic'
DATA_PATH = './output_datasets'
CONTEXT_WINDOW = 15

TASKS = {
    'detection': {
        'pickle': 'shrec_empathic_detection.pickle',
        'sample_size': 200,
    },
    'detection_error_only': {
        'pickle': 'shrec_empathic_detection_error_only.pickle',
        'sample_size': 200,
    },
    'attribute': {
        'pickle': 'shrec_empathic_attribute.pickle',
        'sample_size': 100,
    },
    'attribute_agreed_multiple_subj': {
        'pickle': 'shrec_empathic_attribute_agreed_multiple_subj.pickle',
        'sample_size': 100,
    },
    'rationale': {
        'pickle': 'shrec_empathic_rationale.pickle',
        'sample_size': 200,
    },
    'correction': {
        'pickle': 'shrec_empathic_correction.pickle',
        'sample_size': 200,
    },
    'pre': {
        'pickle': 'shrec_empathic_pre.pickle',
        'sample_size': 100,
    },
    'post': {
        'pickle': 'shrec_empathic_post.pickle',
        'sample_size': 100,
    },
}

SEEDS = [0, 1, 2]

CONDITIONS = {
    'A': {
        'video': False,
        'er_video_mode': None,
    },
    'B': {
        'video': True,
        'er_video_mode': 'video',
    },
}

OUTPUT_DIR = './output'
PYDANTIC_DIR = './output_pydantic'
LOG_DIR = './logs'

# ============================================================
# Helpers
# ============================================================

def output_filename(run_name, task_pickle):
    """Compute the output filename that main_vlm_exp.py will produce."""
    return os.path.join(OUTPUT_DIR, '{}_{}_{}.json'.format(MODEL, run_name, task_pickle))


def run_name_for(condition, seed):
    return 'cond{}_seed{}'.format(condition, seed)


def build_command(task_name, task_config, condition_config, seed, run_name, sample_override=None):
    """Build the main_vlm_exp.py command for a single run."""
    cmd = [
        sys.executable, 'main_vlm_exp.py',
        '--context_window', str(CONTEXT_WINDOW),
        '--model', MODEL,
        '--data_path', DATA_PATH,
        '--task_type', task_config['pickle'],
        '--csv_path', CSV_PATH,
        '--images_dir', IMAGES_DIR,
        '--seed', str(seed),
        '--run_name', run_name,
    ]

    sample_size = sample_override if sample_override else task_config['sample_size']
    cmd.extend(['--sample_size', str(sample_size)])

    if condition_config['video']:
        cmd.append('--video')
        cmd.extend(['--er_video_mode', condition_config['er_video_mode']])

    return cmd


def is_run_complete(run_name, task_pickle, expected_count):
    """Check if a run's output file exists and has enough samples."""
    out_file = output_filename(run_name, task_pickle)
    if not os.path.exists(out_file):
        return False
    try:
        with open(out_file, 'r') as f:
            data = json.load(f)
        return len(data) >= expected_count
    except (json.JSONDecodeError, IOError):
        return False


def run_one(task_name, task_config, condition_name, condition_config, seed,
            sample_override=None, dry_run=False):
    """Execute a single experiment configuration."""
    run_name = run_name_for(condition_name, seed)
    sample_size = sample_override if sample_override else task_config['sample_size']
    out_file = output_filename(run_name, task_config['pickle'])
    log_file = os.path.join(LOG_DIR, '{}_{}.log'.format(run_name, task_name))

    # Check if already done
    if is_run_complete(run_name, task_config['pickle'], sample_size):
        try:
            with open(out_file, 'r') as f:
                n = len(json.load(f))
            print('  [SKIP] {} -> already has {}/{} samples'.format(
                os.path.basename(out_file), n, sample_size))
            return 'SKIPPED'
        except:
            pass

    cmd = build_command(task_name, task_config, condition_config, seed, run_name, sample_override)

    print('  [RUN] {} samples, seed={}'.format(sample_size, seed))
    print('        {}'.format(' '.join(cmd)))

    if dry_run:
        return 'DRY_RUN'

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    start_time = time.time()
    with open(log_file, 'w') as lf:
        lf.write('Command: {}\n'.format(' '.join(cmd)))
        lf.write('Started: {}\n'.format(datetime.now().isoformat()))
        lf.write('=' * 80 + '\n\n')
        lf.flush()
        result = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print('  [FAILED] exit code {} ({:.0f}s)'.format(result.returncode, elapsed))
        print('           See log: {}'.format(log_file))
        return 'FAILED'

    # Verify output
    if not os.path.exists(out_file):
        print('  [FAILED] output file not created ({:.0f}s)'.format(elapsed))
        print('           See log: {}'.format(log_file))
        return 'FAILED'

    try:
        with open(out_file, 'r') as f:
            data = json.load(f)
        n = len(data)
    except:
        n = 0

    print('  [DONE] {} samples ({:.0f}s)'.format(n, elapsed))
    return 'DONE'


def run_pydantic_parsing():
    """Run eval_pydantic.py to parse all unprocessed output files."""
    print('\n' + '=' * 60)
    print('Running pydantic parsing (eval_pydantic.py)...')
    print('=' * 60)
    os.makedirs(PYDANTIC_DIR, exist_ok=True)
    result = subprocess.run(
        [sys.executable, 'eval_pydantic.py'],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print('  [WARNING] eval_pydantic.py exited with code {}'.format(result.returncode))
        print('  stderr:', result.stderr[-500:] if result.stderr else '(empty)')
    else:
        print('  [DONE] pydantic parsing complete')
    return result.returncode


def run_final_eval():
    """Run eval.py to compute all metrics."""
    print('\n' + '=' * 60)
    print('Running final evaluation (eval.py)...')
    print('=' * 60)
    result = subprocess.run(
        [sys.executable, 'eval.py'],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print('stderr:', result.stderr[-500:])
    return result.returncode


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='SHREC Experiment Runner')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--task', type=str, default=None, help='Filter to one task name')
    parser.add_argument('--seed', type=int, default=None, help='Filter to one seed')
    parser.add_argument('--condition', type=str, default=None, help='Filter to one condition (A or B)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip pydantic parsing and final eval')
    parser.add_argument('--sample-override', type=int, default=None,
                        help='Override all sample sizes (for testing)')
    args = parser.parse_args()

    # Filter tasks
    tasks_to_run = TASKS
    if args.task:
        if args.task not in TASKS:
            print('ERROR: Unknown task "{}". Available: {}'.format(args.task, ', '.join(TASKS.keys())))
            sys.exit(1)
        tasks_to_run = {args.task: TASKS[args.task]}

    # Filter seeds
    seeds_to_run = SEEDS
    if args.seed is not None:
        seeds_to_run = [args.seed]

    # Filter conditions
    conditions_to_run = CONDITIONS
    if args.condition:
        if args.condition not in CONDITIONS:
            print('ERROR: Unknown condition "{}". Available: {}'.format(args.condition, ', '.join(CONDITIONS.keys())))
            sys.exit(1)
        conditions_to_run = {args.condition: CONDITIONS[args.condition]}

    # Build the full run list
    run_list = []
    for task_name, task_config in tasks_to_run.items():
        for condition_name, condition_config in conditions_to_run.items():
            for seed in seeds_to_run:
                run_list.append((task_name, task_config, condition_name, condition_config, seed))

    total = len(run_list)
    print('=' * 60)
    print('SHREC Experiment Runner — Gemini Robotics-ER')
    print('=' * 60)
    print('Configurations: {} task(s) x {} condition(s) x {} seed(s) = {} runs'.format(
        len(tasks_to_run), len(conditions_to_run), len(seeds_to_run), total))
    if args.sample_override:
        print('Sample size override: {} (all tasks)'.format(args.sample_override))
    print('Output dir:  {}'.format(OUTPUT_DIR))
    print('Pydantic dir: {}'.format(PYDANTIC_DIR))
    print('Log dir:     {}'.format(LOG_DIR))
    print('Dry run:     {}'.format(args.dry_run))
    print('=' * 60)

    # Execute
    results = {}
    for idx, (task_name, task_config, condition_name, condition_config, seed) in enumerate(run_list):
        run_name = run_name_for(condition_name, seed)
        print('\n[{}/{}] {} | {} | seed={}'.format(
            idx + 1, total, task_name, condition_name, seed))

        status = run_one(
            task_name, task_config, condition_name, condition_config, seed,
            sample_override=args.sample_override,
            dry_run=args.dry_run,
        )
        key = '{}_{}_seed{}'.format(task_name, condition_name, seed)
        results[key] = status

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for key, status in results.items():
        marker = '[OK]' if status in ('DONE', 'SKIPPED', 'DRY_RUN') else '[!!]'
        print('  {} {} — {}'.format(marker, key, status))

    done = sum(1 for s in results.values() if s in ('DONE', 'SKIPPED'))
    failed = sum(1 for s in results.values() if s == 'FAILED')
    skipped = sum(1 for s in results.values() if s == 'DRY_RUN')
    print('\n  Done: {}  Failed: {}  Dry-run: {}  Total: {}'.format(done, failed, skipped, total))

    # Run eval pipeline
    if not args.skip_eval and not args.dry_run and failed == 0:
        run_pydantic_parsing()
        run_final_eval()
    elif failed > 0:
        print('\nSkipping eval — {} run(s) failed. Fix and re-run.'.format(failed))

    print('\nFinished at {}'.format(datetime.now().isoformat()))


if __name__ == '__main__':
    main()
