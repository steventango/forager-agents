NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-small/learning_curve.py save pdf &
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-large/learning_curve.py save pdf &
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-small-sweep/learning_curve.py save pdf &
NUMBA_DISABLE_JIT=1 python experiments/forager-two-biome-large-sweep/learning_curve.py save pdf &
NUMBA_DISABLE_JIT=1 python experiments/forager-extra-large-sweep/learning_curve.py save pdf &
NUMBA_DISABLE_JIT=1 python experiments/jellybean-world-extra-large-sweep/learning_curve.py save pdf &
wait
