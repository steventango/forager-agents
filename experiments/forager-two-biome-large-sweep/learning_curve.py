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
from experiment.ExperimentModel import ExperimentModel
from experiment.hypers import update_best_config
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
setFonts(20)

COLORS = {
    'DQN-3': '#00ffff',
    'DQN-5': '#3ddcff',
    'DQN-7': '#57abff',
    'DQN-9': '#8b8cff',
    'DQN-11': '#b260ff',
    'DQN-13': '#d72dff',
    'DQN-15': '#ff00ff',
    'Random': '#000000',
}

METRIC = "reward"
# keep 1 in every SUBSAMPLE measurements
POINTS = 500

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
    for alg, sub_df in sorted(split_over_column(df, col='algorithm'), key=lambda x: int(x[0].split('-')[1])):
        if len(sub_df) == 0: continue

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

        update_best_config(alg, report, __file__)

        xs, ys = extract_learning_curves(
            sub_df,
            report.best_configuration,
            metric=METRIC,
            interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
        )

        subsample = len(xs[0]) // POINTS
        xs = np.asarray(xs)[:, ::subsample]
        ys = np.asarray(ys)[:, ::subsample]
        print(xs.shape, ys.shape)

        # make sure all of the x values are the same for each curve
        assert np.all(np.isclose(xs[0], xs))

        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=ys,
            statistic=Statistic.mean,
        )

        ax.plot(xs[0], res.sample_stat, label=alg, color=COLORS[alg], linewidth=0.5)
        if len(ys) <= 5:
            for x, y in zip(xs, ys):
                ax.plot(x, y, color=COLORS[alg], linewidth=0.5, alpha=0.2)
        else:
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Average Reward')
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if should_save:
        save(
            save_path=f'{path}/plots',
            plot_name=f'learning_curve',
            save_type=save_type,
        )
        plt.clf()
    else:
        plt.show()
        exit()
