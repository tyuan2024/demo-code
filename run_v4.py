"""v4 pipeline runner, 支持断点续跑"""
import subprocess
import sys
import os
import time
import json

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), 'output_v4', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

steps = [
    ('Step 1: Preprocessing', 'step1_preprocess_v4.py'),
    ('Step 2: Peritumoral Rings', 'step2_peritumoral_v4.py'),
    ('Step 3: Habitat Clustering', 'step3_habitat_v4.py'),
    ('Step 3b: Peri Habitat', 'step3b_peri_habitat_v4.py'),
    ('Step 4: Feature Extraction', 'step4_features_v4.py'),
    ('Step 4b: ICC Filter', 'step4b_icc_filter_v4_parallel.py'),
    ('Step 4c: DL Features', 'step4c_dl_features_v4.py'),
    ('Step 5: Feature Selection', 'step5_v4.py'),
    ('Step 6: Classification', 'step6_v4.py'),
    ('Step 7: Survival', 'step7_v4.py'),
    ('Step 8: Figures', 'step8_v4.py'),
]

checkpoint_file = os.path.join(LOG_DIR, 'checkpoint_v4.json')


def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            return json.load(f)
    return {'completed': [], 'timings': {}}


def save_checkpoint(cp):
    with open(checkpoint_file, 'w') as f:
        json.dump(cp, f, indent=2)


def main():
    cp = load_checkpoint()

    for name, script in steps:
        if script in cp['completed']:
            print(f'SKIP (already done): {name}')
            continue

        print(f'\n{"=" * 60}')
        print(f'  {name}')
        print(f'{"=" * 60}\n')
        sys.stdout.flush()

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS_DIR, script)],
            cwd=SCRIPTS_DIR)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f'\nFAILED: {name} (exit code {result.returncode})')
            print(f'Elapsed: {elapsed:.0f}s')
            sys.exit(1)

        cp['completed'].append(script)
        cp['timings'][script] = round(elapsed, 1)
        save_checkpoint(cp)
        print(f'\nDone: {name} ({elapsed:.0f}s)')
        sys.stdout.flush()

    print(f'\n{"=" * 60}')
    print('  ALL v4 STEPS COMPLETE')
    print(f'{"=" * 60}')
    total = sum(cp['timings'].values())
    print(f'Total time: {total:.0f}s ({total/3600:.1f}h)')
    for script, t in cp['timings'].items():
        print(f'  {script:40s} {t:8.1f}s')


if __name__ == '__main__':
    main()
