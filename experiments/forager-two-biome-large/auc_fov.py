import os
import sys

sys.path.append(os.getcwd() + '/src')

import matplotlib.pyplot as plt
import numpy as np
import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics
from PyExpPlotting.matplot import save, setDefaultConference, setFonts
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.interpolation import compute_step_return
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import (
    TimeSummary,
    curve_percentile_bootstrap_ci,
    extract_learning_curves,
)
from RlEvaluation.utils.pandas import split_over_column

# from analysis.confidence_intervals import bootstrapCI
from utils.plotting import GDMColor
from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
setFonts(20)

METRIC = "reward"
LAST_PERCENT = 0.1
ORDER = {
    "Random": 0,
    "Greedy": 2,
    "Greedy-122":1,
}

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
    apertures = []
    auc = []
    auc_ci_low = []
    auc_ci_high = []
    special = {}
    for env, env_df in sorted(split_over_column(df, col='environment.aperture'), key=lambda x: x[0]):
        for alg, sub_df in sorted(split_over_column(env_df, col='algorithm'), key=lambda x: x[0]):
            if len(sub_df) == 0: continue

            report = Hypers.select_best_hypers(
                sub_df,
                metric=METRIC,
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print('-' * 25)
            print(env, alg)
            Hypers.pretty_print(report)

            xs, ys = extract_learning_curves(
                sub_df,
                report.best_configuration,
                metric=METRIC,
                interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            xs = np.asarray(xs)
            ys = np.asarray(ys)

            # make sure all of the x values are the same for each curve
            assert np.all(np.isclose(xs[0], xs))

            print(xs.shape, ys.shape)
            last_idx = int((1 - LAST_PERCENT) * xs.shape[1])
            xs = xs[:, last_idx:]
            ys = ys[:, last_idx:]
            print(xs.shape, ys.shape)
            xs = xs.mean(axis=1, keepdims=True)
            ys = ys.mean(axis=1, keepdims=True)
            print(xs.shape, ys.shape)

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
            )

            if alg.startswith("DQN"):
                apertures.append(env)
                auc.append(res.sample_stat)
                auc_ci_low.extend(res.ci[0])
                auc_ci_high.extend(res.ci[1])
            else:
                special[alg] = res



    ax.plot(apertures, auc, label='DQN', color=GDMColor.BLUE, linewidth=1)
    ax.fill_between(apertures, auc_ci_low, auc_ci_high, color=GDMColor.BLUE, alpha=0.2)

    for (alg, report), color in zip(sorted(special.items(), key=lambda x: ORDER[x[0]]), [ GDMColor.BLACK, GDMColor.RED, GDMColor.GREEN,]):
        if alg == "Greedy":
            alg = "Search Oracle"
        if alg == "Greedy-122":
            alg = "Search Nearest"
        ax.plot(apertures, [report.sample_stat] * len(apertures), label=alg, color=color, linewidth=1)
        ax.fill_between(apertures, report.ci[0], report.ci[1], color=color, alpha=0.4)

    ax.set_xlabel('Field of View')
    ax.set_ylabel('Last 10% Average Reward AUC')
    ax.set_xticks(apertures)
    ax.set_xticklabels([str(int(x)) for x in apertures])

    # ax.legend(ncol=2, frameon=False)
    ax.text(3, 1.3, "Search Oracle", color=GDMColor.GREEN)
    # right side
    ax.text(15, 1.2, "DQN", color=GDMColor.BLUE, ha='right')
    ax.text(15, 0.95, "Search Nearest", color=GDMColor.RED, ha='right')
    ax.text(15, 0.4, "Random", color=GDMColor.BLACK, ha='right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if should_save:
        save(
            save_path=f'{path}/plots',
            plot_name=f'auc_fov',
            save_type=save_type,
        )
        plt.clf()
    else:
        plt.show()
        exit()
