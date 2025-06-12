import os
import sys

sys.path.append(os.getcwd() + '/src')

import argparse
import random
import subprocess
from functools import partial
from multiprocessing.pool import Pool

from PyExpUtils.runner.utils import gather_missing_indices

import experiment.ExperimentModel as Experiment

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, required=True)
parser.add_argument('-e', type=str, nargs='+', required=True)
parser.add_argument('--cpus', type=int, default=8)
parser.add_argument("--gpu", action="store_true", default=False)
parser.add_argument("--record", action="store_true", default=False)
parser.add_argument('--entry', type=str, default='src/main.py')
parser.add_argument('--results', type=str, default='./')
parser.add_argument("--xla_python_client_mem_fraction", type=float, default=0.3)

def count(pre, it):
    print(pre, 0, end='\r')
    for i, x in enumerate(it):
        print(pre, i + 1, end='\r')
        yield x

    print()

if __name__ == "__main__":
    cmdline = parser.parse_args()

    pool = Pool(cmdline.cpus)

    gpu = "--gpu" if cmdline.gpu else ""
    record = "--record" if cmdline.record else ""

    cmds = []
    e_to_missing = gather_missing_indices(cmdline.e, cmdline.runs, loader=Experiment.load)
    for path in cmdline.e:
        exp = Experiment.load(path)

        indices = count(path, e_to_missing[path])
        for idx in indices:
            exe = f"python {cmdline.entry} {gpu} {record} --silent -e {path} -i {idx}"
            cmds.append(exe)

    print(len(cmds))
    random.shuffle(cmds)
    n = len(cmds)

    max_workers = os.cpu_count() or 1
    workers = min(max_workers, n, cmdline.cpus)
    env = os.environ.copy()
    if not cmdline.gpu:
        env["JAX_PLATFORMS"] = "cpu"
    else:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(cmdline.xla_python_client_mem_fraction / workers)

    res = pool.imap_unordered(partial(subprocess.run, shell=True, stdout=subprocess.PIPE, env=env), cmds, chunksize=1)
    for i, _ in enumerate(res):
        sys.stderr.write(f'\r{i+1}/{len(cmds)}')
    sys.stderr.write('\n')
