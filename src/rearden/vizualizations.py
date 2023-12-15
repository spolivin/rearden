"""Visualization tools."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import os
from collections.abc import Sequence
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Type aliases for type signatures
NameScore = tuple[str, float]


def save_plot_in_dir(
    file_name: str,
    dir_name: str = "images",
) -> None:
    """Saves a plot in a directory.

    Function automatically creates a separate directory and
    saves a plot there. Additionally, it verifies the correctness
    of a file format and by default saves a plot as png-file if
    `file_name` is defined without a format.

    Function is to be called before `plt.show()`.

    Args:
        file_name (str): Name of a file. Can be specified with
            a file format or without it (converting to png-format
            by default).
        dir_name (str, optional): Name of a directory where plot
            is to be saved. Defaults to "images".

    Raises:
        OSError: Exception raised if the format of a file is incorrect.
    """
    # Creating a separate directory if absent
    if os.path.isdir(dir_name) is False:
        os.makedirs(dir_name)
    # Checking a file name
    formats_allowed = (".png", ".pdf", ".svg")
    if not file_name.endswith(formats_allowed):
        file_name_splitted = file_name.split(".")
        # Exception raised for unknown format
        if len(file_name_splitted) != 1:
            wrong_format = file_name_splitted[-1]
            raise OSError(f"Format '{wrong_format}' not recognized.")
        # If file name without format, add ".png" to file name
        file_name += formats_allowed[0]
    # Saving a plot
    plot_path = os.path.join(dir_name, file_name)
    plt.savefig(plot_path)


def plot_model_comparison(
    results: Sequence[NameScore],
    metric_name: str = "metric_name",
    title_name: str = "title_name",
    dot_size: int = 150,
    figure_dims: tuple[int] = (15, 7),
    xticks_fontsize: int = 15,
    yticks_fontsize: int = 12,
    title_fontsize: int = 20,
    ylabel_fontsize: int = 15,
    save_fig: bool = False,
) -> Any:
    """Provides models performance visualization.

    Generates a scatterplot with model names and their
    respective metric values for comparison.

    Args:
        results (Sequence[NameScore]): Sequence of tuples with
            model names and metric values.
        metric_name (str, optional): Name of the metric. Defaults to
            "metric_name".
        title_name (str, optional): Title of the plot. Defaults to
            "title_name".
        dot_size (int, optional): Size of scatterplot dots.
            Defaults to 150.
        figure_dims (tuple[int], optional): Dimensions of the figure.
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
    names_with_scores = [f"{name}\n{loss:.4f}" for name, loss in results]

    # Making a plot
    plt.figure(figsize=figure_dims)

    plt.scatter(range(len(results)), scores, s=dot_size)

    plt.xticks(
        range(len(results)), names_with_scores, fontsize=xticks_fontsize
    )
    plt.yticks(fontsize=yticks_fontsize)
    plt.ylabel(metric_name, fontsize=ylabel_fontsize)
    plt.title(title_name, fontsize=title_fontsize)

    plt.tight_layout()

    # Saving the figure in the directory
    if save_fig:
        save_plot_in_dir(file_name="model_comparison.png")

    plt.show()


def plot_corr_heatmap(
    data: pd.DataFrame,
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
    # Computing the correlation matrix
    corr_matrix = data.corr()

    # Showing upper triangle of a matrix
    if upper_triangle:
        mask = np.zeros_like(corr_matrix)
        mask[np.tril_indices_from(mask)] = True
    # Showing lower triangle of a matrix
    elif lower_triangle:
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True
    # Showing the entire matrix
    else:
        mask = None

    # Plotting a heatmap of the matrix
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

    plt.tight_layout()

    # Saving the figure in the directory
    if save_fig:
        save_plot_in_dir(file_name="corr_heatmap.png")

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
    # Plotting the shares of classes in the target variable
    target_var.value_counts(normalize=True).plot(
        kind="bar",
        xlabel=xlabel_name,
        ylabel=ylabel_name,
        title=title_name,
    )
    plt.xticks(rotation=0)

    plt.tight_layout()

    # Saving the figure in a directory
    if save_fig:
        save_plot_in_dir(file_name="class_structure.png")

    plt.show()
