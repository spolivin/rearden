"""Additional metrics."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import numpy as np
from numpy.typing import ArrayLike


def smape_score(
    y_true: ArrayLike, y_pred: ArrayLike, percent: bool = False
) -> float:
    """Calculates the value of sMAPE metric.

    Args:
        y_true (ArrayLike): Real target data.
        y_pred (ArrayLike): Predicted values of target.
        percent (bool, optional): Boolean indicator of returning
            the value of metric in percent. Defaults to False.

    Returns:
        float: Floating point number being the value
        of sMAPE metric.
    """
    # Computing the score in relative terms
    smape = np.mean(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )

    # If needed, transforming the result into percent
    if percent:
        smape *= 100

    return smape
