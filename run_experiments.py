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

# Model registry: run key -> backend config. run key is the output-file prefix.
MODELS = {
    'GeminiRoboticsER': {'backend': 'gemini', 'model_id': 'gemini-robotics-er-2-preview', 'conditions': ['A', 'B']},
    'Gemini35Flash':    {'backend': 'gemini', 'model_id': 'gemini-3.6-flash', 'conditions': ['A', 'B']},  # gemini-3.5-flash requests are served by gemini-3.6-flash (confirmed in sidecar logs); pinned to the stable endpoint
    'GPT4oMiniCheck':   {'backend': 'openai', 'model_id': 'gpt-4o-mini', 'conditions': ['A']},
    'CosmosReason2':    {'backend': 'vllm', 'model_id': 'nvidia/Cosmos-Reason2-32B', 'conditions': ['A', 'B']},
    'Qwen3VL':          {'backend': 'vllm', 'model_id': 'Qwen/Qwen3-VL-32B-Instruct', 'conditions': ['A', 'B']},
    'CosmosReason2_2B': {'backend': 'vllm', 'model_id': 'nvidia/Cosmos-Reason2-2B', 'conditions': ['A', 'B']},
    'Qwen3VL_2B':       {'backend': 'vllm', 'model_id': 'Qwen/Qwen3-VL-2B-Instruct', 'conditions': ['A', 'B']},
}

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


def build_command(task_name, task_config, condition_config, seed, run_name, sample_override=None, model_cfg=None, extra_args=None):
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

    if model_cfg:
        cmd.extend(['--backend', model_cfg['backend']])
        cmd.extend(['--model_id', model_cfg['model_id']])
    if extra_args:
        cmd.extend(extra_args)

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
            sample_override=None, dry_run=False, model_cfg=None, extra_args=None):
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

    cmd = build_command(task_name, task_config, condition_config, seed, run_name, sample_override, model_cfg=model_cfg, extra_args=extra_args)

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
    parser.add_argument('--model', type=str, default='GeminiRoboticsER',
                        help='Model run key. Available: ' + ', '.join(MODELS.keys()))
    parser.add_argument('--precision', type=str, default='bf16', help='Precision for local backends')
    parser.add_argument('--max-output-tokens', type=int, default=4096, help='Max output tokens for local backends')
    parser.add_argument('--vllm-url', type=str, default='http://localhost:8000/v1', help='vLLM server base URL')
    parser.add_argument('--videos-dir', type=str, default='./data/videos_1hz', help='Pre-rendered 1Hz segment videos dir')
    args = parser.parse_args()

    global MODEL
    MODEL = args.model
    if MODEL not in MODELS:
        print('ERROR: Unknown model "{}". Available: {}'.format(MODEL, ', '.join(MODELS.keys())))
        sys.exit(1)
    model_cfg = MODELS[MODEL]

    extra_args = []
    if model_cfg['backend'] == 'vllm':
        extra_args += ['--precision', args.precision, '--max_output_tokens', str(args.max_output_tokens), '--vllm_url', args.vllm_url]
    if MODEL != 'GeminiRoboticsER':
        extra_args += ['--videos_dir', args.videos_dir]

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
            if 'conditions' in model_cfg and condition_name not in model_cfg['conditions']:
                continue
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
            model_cfg=model_cfg,
            extra_args=extra_args,
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
