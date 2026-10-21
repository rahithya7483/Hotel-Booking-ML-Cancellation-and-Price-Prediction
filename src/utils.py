import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")


def savefig(name, tight=True):
    from config import FIGS
    if tight:
        plt.tight_layout()
    plt.savefig(FIGS / name, dpi=130, bbox_inches="tight")
    plt.close()
