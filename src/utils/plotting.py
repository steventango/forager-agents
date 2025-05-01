from enum import StrEnum


def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)


def plot(ax, data, label=None):
    mean, ste, runs = data
    (base,) = ax.plot(mean, label=label, linewidth=2)
    (low_ci, high_ci) = confidenceInterval(mean, ste)
    ax.fill_between(range(mean.shape[0]), low_ci, high_ci, color=base.get_color(), alpha=0.4)


class GDMColor(StrEnum):
    BLUE = "#4285f4"
    ORANGE = "#ff902a"
    GREEN = "#34a853"
    RED = "#ea4335"
    YELLOW = "#fbbc04"
    TEAL = "#2daeb8"
    BLACK = "#000000"
