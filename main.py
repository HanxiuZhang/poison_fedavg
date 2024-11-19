from flips import *
from selection import RandomSelectionStrategy
from server import run_exp
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    START_EXP_IDX = 0
    NUM_EXP = 1
    NUM_POISONED_WORKERS = 5
    REPLACEMENT_METHOD = replace_1_with_9
    KWARGS = {
        "NUM_WORKERS_PER_ROUND" : 10
    }

    for experiment_id in range(START_EXP_IDX, START_EXP_IDX + NUM_EXP):
        run_exp(REPLACEMENT_METHOD, NUM_POISONED_WORKERS, KWARGS, RandomSelectionStrategy(), experiment_id)
