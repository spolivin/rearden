import os
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_model_comparison(
    results: Sequence[Tuple[str, float]],
    metric_name: str = "metric_name",
    title_name: str = "title_name",
    dot_size: int = 150,
    figure_dims: Tuple[int] = (15, 7),
    xticks_fontsize: int = 15,
    yticks_fontsize: int = 12,
    title_fontsize: int = 20,
    ylabel_fontsize: int = 15,
    save_fig: bool = False,
) -> Any:
    """Provides models performance vizualization.

    Generates a scatterplot with model names and their
    respective metric values for comparison.

    Args:
        results (Sequence[Tuple[str, float]]): Sequence of Tuples with
            model names and metric values.
        metric_name (str, optional): Name of the metric. Defaults to
            "metric_name".
        title_name (str, optional): Title of the plot. Defaults to
            "title_name".
        dot_size (int, optional): Size of scatterplot dots.
            Defaults to 150.
        figure_dims (Tuple[int], optional): Dimensions of the figure.
            Defaults to (15, 7).
        xticks_fontsize (int, optional): Size of xticks on the plot.
            Defaults to 15.
        yticks_fontsize (int, optional): Size of yticks on the plot.
            Defaults to 12.
        title_fontsize (int, optional): Size of the title of the plot.
            Defaults to 20.
        ylabel_fontsize (int, optional): Size of the Y-label.
            Defaults to 15.
        save_fig (bool, optional): Boolean indicating saving the plot
            in a separate directory. Defaults to False.
    """
    # Separating scores from a sequence of tuples passed
    _, scores = zip(*results)
    # Joining model names with scores
    names_with_scores = ["%s\n%.4f" % (name, loss) for name, loss in results]

    # Making a plot
    plt.figure(figsize=figure_dims)

    plt.scatter(range(len(results)), scores, s=dot_size)

    plt.xticks(
        range(len(results)), names_with_scores, fontsize=xticks_fontsize
    )
    plt.yticks(fontsize=yticks_fontsize)

    plt.title(title_name, fontsize=title_fontsize)
    plt.ylabel(metric_name, fontsize=ylabel_fontsize)

    plt.tight_layout()

    if save_fig:
        dir_name = "images/"
        if os.path.isdir(dir_name) is False:
            os.makedirs(dir_name)
        plt.savefig(dir_name + "model_comparison.png")

    plt.show()


def plot_corr_heatmap(
    data: pd.DataFrame,
    target_var: Optional[pd.Series] = None,
    annotation: bool = False,
    annot_num_size: Optional[int] = None,
    heatmap_coloring: Optional[Any] = None,
    upper_triangle: bool = False,
    lower_triangle: bool = False,
    save_fig: bool = False,
) -> Any:
    """Plots a heatmap for the correlation matrix.

    Args:
        data (pd.DataFrame): DataFrame for which a heatmap needs
            to be plotted.
        target_var (Optional[pd.Series], optional): Vector with a
            target variable if needed to include it on the heatmap.
            Defaults to None.
        annotation (bool, optional): Boolean indicator of displaying
            numbers in the heatmap. Defaults to False.
        annot_num_size (Optional[int], optional): Size of the
            figures inside the plot. Defaults to None.
        heatmap_coloring (Optional[Any], optional): Heatmap colormap.
            Defaults to None.
        upper_triangle (bool, optional): Boolean indicator of displaying
            only the upper triangle of the matrix. Defaults to False.
        lower_triangle (bool, optional): Boolean indicator of displaying
            only the lower triangle of the matrix. Defaults to False.
        save_fig (bool, optional): Boolean indicating saving the figure
            in a separate directory. Defaults to False.
    """
    if target_var is not None:
        data = pd.concat([data, target_var], axis=1)

    corr_matrix = data.corr()

    # Showing upper/lower triangle of a matrix
    if upper_triangle:
        mask = np.zeros_like(corr_matrix)
        mask[np.tril_indices_from(mask)] = True
    elif lower_triangle:
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None

    # Plotting a matrix
    corr_heatmap = sns.heatmap(
        corr_matrix,
        annot=annotation,
        mask=mask,
        annot_kws={"size": annot_num_size},
        cmap=heatmap_coloring,
    )
    corr_heatmap.xaxis.tick_bottom()
    corr_heatmap.yaxis.tick_left()
    corr_heatmap.set(title="Correlation matrix heatmap")

    if save_fig:
        dir_name = "images/"
        if os.path.isdir(dir_name) is False:
            os.makedirs(dir_name)
        plt.savefig(dir_name + "corr_mat_heatmap.png")

    plt.show()


def plot_class_structure(
    target_var: pd.Series,
    xlabel_name: str = "xlabel_name",
    ylabel_name: str = "ylabel_name",
    title_name: str = "title_name",
    save_fig: bool = False,
) -> Any:
    """Plots the structure of the target variable.

    Args:
        target_var (pd.Series): Target vector.
        xlabel_name (str, optional): Name of the xlabel on the plot.
            Defaults to "xlabel_name".
        ylabel_name (str, optional): Name of the ylabel on the plot.
            Defaults to "ylabel_name".
        title_name (str, optional): Title of the plot.
            Defaults to "title_name".
        save_fig (bool, optional): Boolean indicating saving the figure
            in a separate directory. Defaults to False.
    """
    target_var.value_counts(normalize=True).plot(
        kind="bar",
        xlabel=xlabel_name,
        ylabel=ylabel_name,
        title=title_name,
    )
    plt.xticks(rotation=0)

    if save_fig:
        dir_name = "images/"
        if os.path.isdir(dir_name) is False:
            os.makedirs(dir_name)
        plt.savefig(dir_name + "class_structure.png")

    plt.show()
