import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
import numpy as np
from tqdm import tqdm
import torch

from typing import List, Dict, Tuple, Optional
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator


class WandbRun:
    def __init__(
        self,
        wandb_api: wandb.Api,
        run_path: str,
        model_type: str = "transformer",
        prefix: str = "final",
    ):
        run = wandb_api.run(run_path)

        self.summary = dict(run.summary)
        config = dict(run.config)
        self.config = dict()
        for k, v in config.items():
            if isinstance(v, Dict):
                self.config = {**self.config, **v}
            else:
                self.config[k] = v

        if model_type == "transformer":
            if self.config["working_memory"]:
                self.model_type = "MapWM"
            elif self.config["rope"]:
                self.model_type = "RoPE"
            elif self.config["cope"]:
                self.model_type = "CoPE"
        else:
            self.model_type = "MapEM"

        self.key_metrics = dict()
        self.key_metrics["n_layer"] = self.config["n_layer"]
        self.key_metrics["MLP"] = self.config["use_mlp"]
        if self.model_type == "RoPE":
            
            self.model_type += f"-{self.config['n_layer']}L"
        else:
            self.model_type += f"-R{self.config['dt_rank']}"

        if self.key_metrics["MLP"]:
            self.model_type += f"-MLP"

        self.prefix = prefix
        columns = [
            c.split("_")[1:]
            for c in self.summary.keys()
            if f"{prefix}_seq" in c and "f1" in c
        ]
        # columns = [c.split("_")[1:] for c in self.history_df.columns if f"{prefix}_seq" in c and "f1" in c]
        self.val_indexes = dict()
        for v in columns[0][:-1:2]:
            values = list()
            for c in columns:
                i = c.index(v)
                values.append(int(c[i + 1]))
            self.val_indexes[v] = np.sort(np.unique(values)).tolist()

        self._get_validation()

    def _get_validation(self):
        self.validations = np.zeros(
            (len(self.val_indexes["seq"]), len(self.val_indexes["depth"]))
        )
        for i, n in enumerate(self.val_indexes["seq"]):
            for j, d in enumerate(self.val_indexes["depth"]):
                if "symb" in self.val_indexes:
                    col_name = f"{self.prefix}_seq_{n}_depth_{d}_symb_{self.val_indexes['symb'][0]}_f1"
                else:
                    col_name = f"{self.prefix}_seq_{n}_depth_{d}_f1"
                self.validations[i, j] = self.summary[col_name]
                # self.validations[i, j] = self.history_df[
                #     self.history_df[col_name].notna()
                # ][col_name].values.tolist()[-1]

        if "symb" in self.val_indexes:
            self.sym_vals = np.zeros((len(self.val_indexes["symb"]),))
            for i, s in enumerate(self.val_indexes["symb"]):
                col_name = f"{self.prefix}_seq_1_depth_0_symb_{s}_f1"
                self.sym_vals[i] = self.summary[col_name]
                # self.sym_vals[i] = self.history_df[self.history_df[col_name].notna()][
                #     col_name
                # ].values.tolist()[-1]


def plot_sns_figure(
    model_names: List[str],
    key_metrics: Dict,
    x_values: List[float],
    x_name: str,
    y_values: List[str],
    y_name: str,
    fig_title: str,
    key_metric: str = "Model Type",
    save_fig: bool = False,
    plt_params: Dict = {},
    loc_legend: str = "lower right",
    figsize: Tuple[float] = (8.0, 5.0),
) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=2)
    plt.rcParams.update(
        {
            #     'font.size': 12,
            #     'axes.labelsize': 12,
            #     'axes.titlesize': 14,
            #     'legend.fontsize': 10,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "lines.linewidth": 3.0,
            "figure.figsize": figsize,  # Standard single-column figure size
            "figure.dpi": 300,
            "figure.autolayout": True,
            #     'grid.linestyle': '',
            #     'grid.alpha': 0.7,
            "axes.grid": False,  # Ensure grid is visible,
            **plt_params,
        }
    )
    data = list()
    for i, model in enumerate(model_names):
        print(len(y_values[i]))
        df_run = pd.DataFrame(
            {
                x_name: x_values[i],
                y_name: y_values[i],
                "Model": model,
                key_metric: key_metrics[i]
    
            }
        )
        data.append(df_run)
    df_history = pd.concat(data)
    fig, ax = plt.subplots()

    # Use hue for Model, and style for Metric Type
    sns.lineplot(
        data=df_history,
        x=x_name,
        y=y_name,
        hue="Model",
        color="color",
        style=key_metric,  # Use dashed/solid lines for train vs val
        # style=style_key,  # Use dashed/solid lines for train vs val
        #     errorbar='sd',
        ax=ax,
        markers=False,
        dashes=False,
        linewidth=2,
        markersize=10,
    )

    # x_min = int(df_history[x_name].min())
    # x_max = int(df_history[x_name].max())
    # n_ticks = 5 
    # tick_positions = np.linspace(0, x_max-x_min, n_ticks, dtype=np.int32)

    # # 3. Force the axis to use these positions
    # ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.2f}'))
    # ax.set_xticks(tick_positions)

    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1)) # Forces scientific notation for numbers outside [0.1, 10]
    ax.xaxis.set_major_formatter(formatter)

    # 2. Set the number of ticks if it's still too crowded
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",  # Draw ticks inward
        top=True,  # Show ticks on the top axis
        right=True,  # Show ticks on the right axis
        bottom=True,  # Show ticks on the top axis
        left=True,  # Show ticks on the right axis
        length=4,
        width=1,
    )
    ax.ticklabel_format()

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")

    # ax.tick_params(labeltop=False, labelright=False)
    # ax.set_xticks(x_values)
    # ax.set_xticklabels(x_values)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    handles, labels = ax.get_legend_handles_labels()
    l = len(np.unique(key_metrics))
    handles = handles[1 : l+1]
    labels = labels[1 : l+1]
    # labels = labels[1 : 1 + len(key_metrics)]
    # plt.legend(labels, title="", loc=loc_legend, frameon=True)
    plt.legend(handles, labels, title="", loc=loc_legend, frameon=True)
    ax.set_title(
        fig_title,
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    if save_fig:
        fig.savefig(fig_title, bbox_inches="tight", dpi=300, format="pdf")

    return