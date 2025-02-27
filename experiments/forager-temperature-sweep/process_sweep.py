import os
import sys

sys.path.append(os.getcwd() + '/src')
import matplotlib.pyplot as plt
import RlEvaluation.hypers as Hypers
from PyExpPlotting.matplot import setDefaultConference, setFonts
from PyExpUtils.results.Collection import ResultCollection
from RlEvaluation.config import data_definition
from RlEvaluation.statistics import Statistic
from RlEvaluation.temporal import TimeSummary
from RlEvaluation.utils.pandas import split_over_column

# from analysis.confidence_intervals import bootstrapCI
from experiment.ExperimentModel import ExperimentModel
from experiment.hypers import generate_hyper_sweep_table, update_best_config
from experiment.tools import parseCmdLineArgs

# makes sure figures are right size for the paper/column widths
# also sets fonts to be right size when saving
setDefaultConference('jmlr')
setFonts(20)

METRIC = "reward"

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

    algs = []
    reports = []

    f, ax = plt.subplots()
    for alg, sub_df in sorted(split_over_column(df, col='algorithm'), key=lambda x: x[0]):
        if len(sub_df) == 0: continue

        report = Hypers.select_best_hypers(
            sub_df,
            metric=METRIC,
            prefer=Hypers.Preference.high,
            time_summary=TimeSummary.sum,
            statistic=Statistic.mean,
        )
        algs.append(alg)
        reports.append(report)

        print('-' * 25)
        print(alg)
        Hypers.pretty_print(report)

        update_best_config(alg, report, __file__)

    generate_hyper_sweep_table(algs, reports, __file__)
