"""
SHREC Experiment Setup Script
=============================
Run this FIRST before run_experiments.py.

This script:
  1. Verifies Python venv and installs dependencies
  2. Checks that shrec_empathic.csv and frame data exist
  3. Runs preprocessing for all 8 task types
  4. Verifies all 8 pickle files were created
  5. Checks API keys are set
  6. Runs a 1-sample smoke test with the real ER model
  7. Tells you what to do next

Usage:
  python setup_teammate.py
"""

import os
import sys
import subprocess
import pickle
import json
import argparse
from pathlib import Path

# ============================================================
# Config — paths are relative to the SHREC/ directory
# ============================================================

SHREC_DIR = Path(__file__).parent.resolve()
PARENT_DIR = SHREC_DIR.parent

CSV_PATH = PARENT_DIR / 'shrec_empathic.csv'
FRAMES_DIR = PARENT_DIR / 'shrec_empathic'
OUTPUT_DATASETS = SHREC_DIR / 'output_datasets'

TASKS = [
    'detection',
    'detection_error_only',
    'attribute',
    'attribute_agreed_multiple_subj',
    'rationale',
    'correction',
    'pre',
    'post',
]

EXPECTED_COUNTS = {
    'detection': 1016,
    'detection_error_only': 2794,
    'attribute': 1389,
    'attribute_agreed_multiple_subj': 1389,
    'rationale': 2637,
    'correction': 1320,
    'pre': 310,
    'post': 310,
}

REQUIRED_PACKAGES = [
    'google-genai',
    'google-generativeai',
    'openai',
    'opencv-python',
    'pandas',
    'Pillow',
    'tqdm',
    'pydantic',
    'scikit-learn',
    'natsort',
    'requests',
]


def header(msg):
    print('\n' + '=' * 60)
    print(msg)
    print('=' * 60)


def step(num, msg):
    print('\n[Step {}] {}'.format(num, msg))
    print('-' * 40)


def fail(msg):
    print('\n  [FAILED] {}'.format(msg))
    return False


def ok(msg=''):
    print('  [OK] {}'.format(msg) if msg else '  [OK]')
    return True


def check_command(cmd):
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# ============================================================
# Step 1: Check Python version
# ============================================================

def check_python():
    version = sys.version_info
    print('  Python {}.{}.{}'.format(version.major, version.minor, version.micro))
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return fail('Python 3.8+ required. You have {}.{}'.format(version.major, version.minor))
    return ok('Python version OK')


# ============================================================
# Step 2: Install dependencies
# ============================================================

def install_deps():
    print('  Checking and installing required packages...')
    for pkg in REQUIRED_PACKAGES:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', pkg, '--quiet'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print('  [WARN] Failed to install {}: {}'.format(pkg, result.stderr.strip()[:100]))
        else:
            print('  [OK] {}'.format(pkg))

    # Also install from requirements.txt if it exists
    req_file = SHREC_DIR / 'requirements.txt'
    if req_file.exists():
        print('  Installing from requirements.txt (may take a few minutes)...')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', str(req_file), '--quiet'],
            capture_output=True, text=True
        )
    return ok('Dependencies installed')


# ============================================================
# Step 3: Check data files exist
# ============================================================

def check_data():
    all_good = True

    # Check CSV
    if CSV_PATH.exists():
        size_mb = CSV_PATH.stat().st_size / (1024 * 1024)
        ok('shrec_empathic.csv found ({:.1f} MB)'.format(size_mb))
    else:
        fail('shrec_empathic.csv NOT FOUND at {}'.format(CSV_PATH))
        print('         Expected location: {}'.format(CSV_PATH))
        print('         This file must be in the PARENT directory of SHREC/')
        all_good = False

    # Check frames directory
    if FRAMES_DIR.exists():
        video_dirs = [d for d in FRAMES_DIR.iterdir()
                      if d.is_dir() and not d.name.startswith('.') and d.name != 'data']
        print('  [INFO] Found {} video subdirectories in shrec_empathic/'.format(len(video_dirs)))
        if len(video_dirs) >= 100:
            ok('Frame data found ({} video dirs)'.format(len(video_dirs)))
        else:
            fail('Only {} video dirs found (expected ~143)'.format(len(video_dirs)))
            all_good = False

        # Check one video dir has PNG frames
        if video_dirs:
            sample_dir = video_dirs[0]
            pngs = list(sample_dir.glob('*.png'))
            if len(pngs) > 0:
                ok('Sample dir {} has {} PNG frames'.format(sample_dir.name, len(pngs)))
            else:
                fail('No PNG frames found in {}'.format(sample_dir))
                all_good = False
    else:
        fail('shrec_empathic/ frames directory NOT FOUND at {}'.format(FRAMES_DIR))
        print('         Expected location: {}'.format(FRAMES_DIR))
        print('         This directory must contain 143 video subdirectories with PNG frames')
        all_good = False

    if not all_good:
        print('\n' + '!' * 60)
        print('DATA FILES MISSING!')
        print('!' * 60)
        print("""
You need two data files that are NOT in the git repo:

1. shrec_empathic.csv  (~8 MB)
   Location: {} 

2. shrec_empathic/  (143 video subdirectories, each with ~5-10 PNG frames)
   Location: {}

These are derived from the raw SHREC dataset. Ask your teammate
for a copy of these files, or extract them from the raw SHREC
video data using the frame extraction script.

Once you have these files, re-run this script:
  python setup_teammate.py
""".format(CSV_PATH, FRAMES_DIR))

    return all_good


# ============================================================
# Step 4: Run preprocessing for all 8 task types
# ============================================================

def run_preprocessing():
    os.makedirs(OUTPUT_DATASETS, exist_ok=True)

    for task in TASKS:
        pickle_path = OUTPUT_DATASETS / 'shrec_empathic_{}.pickle'.format(task)

        # Check if already exists with correct count
        if pickle_path.exists():
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                count = len(data)
                expected = EXPECTED_COUNTS.get(task, 0)
                if count >= expected * 0.9:  # Allow 10% variance
                    ok('{} already preprocessed ({} samples)'.format(task, count))
                    continue
                else:
                    print('  [WARN] {} has {} samples (expected ~{}), re-running...'.format(
                        task, count, expected))
            except Exception:
                print('  [WARN] {} exists but corrupted, re-running...'.format(task))

        # Run preprocessing
        print('  Preprocessing {}...'.format(task))
        cmd = [
            sys.executable, 'main_vlm_get_data.py',
            '--data_path', str(CSV_PATH),
            '--data_name', 'shrec_empathic',
            '--task_type', task,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SHREC_DIR))

        if result.returncode != 0:
            fail('Preprocessing {} failed: {}'.format(task, result.stderr[-200:]))
            continue

        # Verify
        if pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            ok('{} -> {} samples'.format(task, len(data)))
        else:
            fail('Pickle not created for {}'.format(task))

    return True


# ============================================================
# Step 5: Verify all 8 pickles
# ============================================================

def verify_pickles():
    all_good = True
    print('  {:40s} {:>10s} {:>10s}'.format('Task', 'Samples', 'Expected'))
    print('  ' + '-' * 62)

    for task in TASKS:
        pickle_path = OUTPUT_DATASETS / 'shrec_empathic_{}.pickle'.format(task)
        if not pickle_path.exists():
            print('  {:40s} {:>10s} {:>10s}'.format(task, 'MISSING', str(EXPECTED_COUNTS.get(task, '?'))))
            all_good = False
            continue

        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            count = len(data)
            expected = EXPECTED_COUNTS.get(task, 0)
            status = 'OK' if count >= expected * 0.9 else 'LOW'
            print('  {:40s} {:>10d} {:>10d}  {}'.format(task, count, expected, status))
            if status == 'LOW':
                all_good = False
        except Exception as e:
            print('  {:40s} ERROR: {}'.format(task, str(e)[:30]))
            all_good = False

    return all_good


# ============================================================
# Step 6: Check API keys
# ============================================================

def check_api_keys():
    all_good = True

    google_key = os.environ.get('GOOGLE_GENAI_API_KEY', '')
    if google_key and google_key != 'dummy':
        ok('GOOGLE_GENAI_API_KEY is set')
    else:
        fail('GOOGLE_GENAI_API_KEY is NOT set')
        print('         This is required for Gemini Robotics-ER API calls.')
        print('         Set it as an environment variable:')
        print('           setx GOOGLE_GENAI_API_KEY "your-key-here"')
        print('         Then restart your terminal and re-run this script.')
        all_good = False

    openai_key = os.environ.get('OPENAI_API_KEY', '')
    if openai_key and openai_key != 'dummy':
        ok('OPENAI_API_KEY is set (for eval_pydantic.py gpt-4o-mini parser)')
    else:
        fail('OPENAI_API_KEY is NOT set')
        print('         This is required for the pydantic answer parser (gpt-4o-mini).')
        print('         Set it as an environment variable:')
        print('           setx OPENAI_API_KEY "your-key-here"')
        print('         Then restart your terminal and re-run this script.')
        all_good = False

    return all_good


# ============================================================
# Step 7: Smoke test (1 sample, real ER API call)
# ============================================================

def smoke_test():
    print('  Running 1-sample detection test with GeminiRoboticsER...')

    cmd = [
        sys.executable, 'main_vlm_exp.py',
        '--context_window', '15',
        '--model', 'GeminiRoboticsER',
        '--data_path', str(OUTPUT_DATASETS),
        '--task_type', 'shrec_empathic_detection.pickle',
        '--csv_path', str(CSV_PATH),
        '--images_dir', str(FRAMES_DIR),
        '--seed', '0',
        '--run_name', 'smoke_test',
        '--sample_size', '1',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SHREC_DIR), timeout=120)

    if result.returncode != 0:
        fail('Smoke test failed!')
        print('  stderr (last 500 chars):')
        print('  ' + result.stderr[-500:])
        return False

    # Check output
    output_file = SHREC_DIR / 'output' / 'GeminiRoboticsER_smoke_test_shrec_empathic_detection.pickle.json'
    if not output_file.exists():
        fail('Output file not created')
        return False

    with open(output_file, 'r') as f:
        data = json.load(f)

    if len(data) == 0:
        fail('No samples in output')
        return False

    response = data[0].get('response', '')
    if not response:
        fail('Empty response from ER model')
        return False

    ok('ER model responded: "{}"'.format(response[:80]))

    # Clean up smoke test files
    try:
        output_file.unlink()
        pydantic_file = SHREC_DIR / 'output_pydantic' / 'GeminiRoboticsER_smoke_test_shrec_empathic_detection.pickle.json'
        if pydantic_file.exists():
            pydantic_file.unlink()
    except:
        pass

    return True


# ============================================================
# Step 8: Test Condition B (video mode)
# ============================================================

def smoke_test_video():
    print('  Running 1-sample video mode test (Condition B)...')

    cmd = [
        sys.executable, 'main_vlm_exp.py',
        '--context_window', '15',
        '--model', 'GeminiRoboticsER',
        '--data_path', str(OUTPUT_DATASETS),
        '--task_type', 'shrec_empathic_detection.pickle',
        '--csv_path', str(CSV_PATH),
        '--images_dir', str(FRAMES_DIR),
        '--seed', '0',
        '--run_name', 'smoke_test_video',
        '--sample_size', '1',
        '--video',
        '--er_video_mode', 'video',
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SHREC_DIR), timeout=120)

    if result.returncode != 0:
        fail('Video smoke test failed!')
        print('  stderr (last 500 chars):')
        print('  ' + result.stderr[-500:])
        return False

    output_file = SHREC_DIR / 'output' / 'GeminiRoboticsER_smoke_test_video_shrec_empathic_detection.pickle.json'
    if not output_file.exists():
        fail('Output file not created')
        return False

    with open(output_file, 'r') as f:
        data = json.load(f)

    if len(data) == 0 or not data[0].get('response'):
        fail('Empty response from ER model')
        return False

    response = data[0]['response']
    ok('ER video mode responded: "{}"'.format(response[:80]))

    # Clean up
    try:
        output_file.unlink()
        pydantic_file = SHREC_DIR / 'output_pydantic' / 'GeminiRoboticsER_smoke_test_video_shrec_empathic_detection.pickle.json'
        if pydantic_file.exists():
            pydantic_file.unlink()
    except:
        pass

    return True


# ============================================================
# Main
# ============================================================

def main():
    header('SHREC Experiment Setup — Gemini Robotics-ER')
    print('  SHREC dir:   {}'.format(SHREC_DIR))
    print('  Parent dir:  {}'.format(PARENT_DIR))
    print('  CSV path:    {}'.format(CSV_PATH))
    print('  Frames dir:  {}'.format(FRAMES_DIR))

    results = {}

    # Step 1: Python
    step(1, 'Checking Python version')
    results['python'] = check_python()

    if not results['python']:
        print('\nFix Python version and re-run.')
        return

    # Step 2: Dependencies
    step(2, 'Installing dependencies')
    results['deps'] = install_deps()

    # Step 3: Data files
    step(3, 'Checking data files (CSV + frames)')
    results['data'] = check_data()

    if not results['data']:
        print('\nFix missing data files and re-run: python setup_teammate.py')
        return

    # Step 4: Preprocessing
    step(4, 'Running preprocessing for all 8 task types')
    results['preprocess'] = run_preprocessing()

    # Step 5: Verify pickles
    step(5, 'Verifying all 8 pickle files')
    results['pickles'] = verify_pickles()

    # Step 6: API keys
    step(6, 'Checking API keys')
    results['api_keys'] = check_api_keys()

    if not results['api_keys']:
        print('\nSet the API keys and re-run: python setup_teammate.py')
        return

    # Step 7: Smoke test (Condition A)
    step(7, 'Smoke test — Condition A (language only, 1 sample)')
    results['smoke_a'] = smoke_test()

    # Step 8: Smoke test (Condition B)
    step(8, 'Smoke test — Condition B (video mode, 1 sample)')
    results['smoke_b'] = smoke_test_video()

    # Summary
    header('SETUP SUMMARY')
    for name, passed in results.items():
        status = '[OK]' if passed else '[!!]'
        print('  {} {}'.format(status, name))

    all_passed = all(results.values())
    if all_passed:
        print('\n' + '=' * 60)
        print('ALL CHECKS PASSED!')
        print('=' * 60)
        print("""
You are ready to run the full experiment:

  python run_experiments.py

This will run 48 configurations (8 tasks x 2 conditions x 3 seeds).
It takes ~3 days. The script resumes automatically if interrupted.

To run a quick test first (2 samples per config):
  python run_experiments.py --sample-override 2

To run one specific task:
  python run_experiments.py --task detection

To dry-run (just print commands):
  python run_experiments.py --dry-run
""")
    else:
        print('\n' + '!' * 60)
        print('SOME CHECKS FAILED — fix the issues above and re-run')
        print('!' * 60)
        print('  python setup_teammate.py')


if __name__ == '__main__':
    main()
