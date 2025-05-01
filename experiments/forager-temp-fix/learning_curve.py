import os
import sys

sys.path.append(os.getcwd() + '/src')

from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import DataDefinition, data_definition, maybe_global
from RlEvaluation.interpolation import Interpolation, compute_step_return
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import TimeSummary, curve_percentile_bootstrap_ci
from RlEvaluation.utils.pandas import split_over_column, subset_df

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
setFonts(20)

COLORS = {
    'DQN': 'blue',
    'DQN-11': 'magenta',
    'Random': 'black',
    'Temperature': 'orange',
    'Greedy': 'green',
    'Greedy-hot': 'cyan',
    'DQN-privileged': 'red',
    'Greedy-privileged': 'purple',
}

SKIP = ["Greedy", "Greedy-hot"]
# SKIP = COLORS.keys() - ["DQN-11"]
METRIC = "reward"
# keep 1 in every SUBSAMPLE measurements
POINTS = 100
PLOT_THE_FIRST = 1.0
post_fix = "" if PLOT_THE_FIRST == 1.0 else f"_{PLOT_THE_FIRST}"

BASE = 'Greedy-privileged'
BASE_COLOR = 'grey'
BASE_LINESTYLE = '--'
base_post_fix = "" if BASE is None else "_based"

PLOT_REWARD = False
reward_post_fix = "_reward" if PLOT_REWARD else ""

if PLOT_REWARD:
    SKIP = COLORS.keys() - ["Temperature"]
else:
    SKIP.append("Temperature")

def extract_learning_curves(
    df: pd.DataFrame,
    hyper_vals: Tuple[Any, ...],
    metric: str,
    data_definition: DataDefinition | None = None,
    interpolation: Interpolation | None = None,
):
    dd = maybe_global(data_definition)
    cols = set(dd.hyper_cols).intersection(df.columns)
    sub = subset_df(df, list(cols), hyper_vals)

    groups = sub.groupby(dd.seed_col)

    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for _, group in groups:
        non_na = group[group[metric].notna()]
        x = non_na[dd.time_col].to_numpy().astype(np.int64)
        y = non_na[metric].to_numpy().astype(np.float64)

        print('source:', x.shape, y.shape)
        # if x is not strictly increasing, there are duplicates,
        # we just take the second half for now
        if not np.all(x[1:] > x[:-1]):
            x = x[len(x) // 2:]
            y = y[len(y) // 2:]
            print('processed:', x.shape, y.shape)

        if interpolation is not None:
            x, y = interpolation(x, y)

        xs.append(x)
        ys.append(y)

    return xs, ys


if __name__ == "__main__":
    path, should_save, save_type = parseCmdLineArgs()

    results = ResultCollection.fromExperiments(Model=ExperimentModel, metrics=[METRIC])

    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col='environment',
        algorithm_col='algorithm',

        # makes this data definition globally accessible
        # so we don't need to supply it to all API calls
        make_global=True,
    )

    df = results.combine(
        # converts path like "experiments/example/MountainCar"
        # into a new column "environment" with value "MountainCar"
        # None means to ignore a path part
        folder_columns=(None, None, 'environment'),

        # and creates a new column named "algorithm"
        # whose value is the name of an experiment file, minus extension.
        # For instance, ESARSA.json becomes ESARSA
        file_col='algorithm',
    )

    assert df is not None

    exp = results.get_any_exp()

    f, ax = plt.subplots()
    for alg, sub_df in sorted(split_over_column(df, col='algorithm'), key=lambda x: x[0] if x[0] != BASE else '0'):
        if len(sub_df) == 0: continue
        if alg in SKIP: continue

        report = Hypers.select_best_hypers(
            sub_df,
            metric=METRIC,
            prefer=Hypers.Preference.high,
            time_summary=TimeSummary.sum,
            statistic=Statistic.mean,
        )

        print('-' * 25)
        print(alg)
        Hypers.pretty_print(report)

        xs, ys = extract_learning_curves(
            sub_df,
            report.best_configuration,
            metric=METRIC,
            interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
        )

        if PLOT_THE_FIRST < 1.0:
            n = int(len(xs[0]) * PLOT_THE_FIRST)
            xs = [x[:n] for x in xs]
            ys = [y[:n] for y in ys]

        subsample = len(xs[0]) // POINTS
        xs = np.asarray(xs)[:, ::subsample]
        ys = np.asarray(ys)[:, ::subsample]
        print(xs.shape, ys.shape)

        if alg == BASE:
            base_ys = ys
        if BASE is not None and not PLOT_REWARD:
            ys = ys / base_ys

        # make sure all of the x values are the same for each curve
        assert np.all(np.isclose(xs[0], xs))

        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=ys,
            statistic=Statistic.mean,
        )

        label = alg if alg != "Temperature" else "Reward"
        sample_stat = res.sample_stat
        if alg == "Temperature":
            # get absolute reward
            sample_stat = np.abs(sample_stat)
            ys = np.abs(ys)
        if alg == BASE:
            ax.plot(xs[0], sample_stat, label=alg, color=BASE_COLOR, linestyle=BASE_LINESTYLE, linewidth=0.5)
        else:
            ax.plot(xs[0], sample_stat, label=label, color=COLORS[alg], linewidth=0.5)
        if len(ys) <= 5 and False:
            for x, y in zip(xs, ys):
                ax.plot(x, y, color=COLORS[alg], linewidth=0.5, alpha=0.2)
        elif not PLOT_REWARD:
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
        ax.set_xlabel("Time steps")
        if PLOT_REWARD:
            ax.set_ylabel('Reward')
        else:
            if BASE is not None:
                ax.set_ylabel("Relative Average Reward")
            else:
                ax.set_ylabel("Average Reward")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    # put legend outside of plot
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box
    #                  .height])
    if "Reward" not in ax.get_legend_handles_labels()[1]:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if should_save:
        save(
            save_path=f"{path}/plots",
            plot_name=f"learning_curve{reward_post_fix}{post_fix}{base_post_fix}",
            save_type=save_type,
        )
        save(
            save_path=f"{path}/plots",
            plot_name=f"learning_curve{reward_post_fix}{post_fix}{base_post_fix}",
            save_type=save_type,
        )
        plt.clf()
    else:
        plt.show()
        exit()
