"""Grid search wrappers."""

# Author: Sergey Polivin <s.polivin@gmail.com>
# License: MIT License

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import RandomizedSearchCV

from .decorators import check_est_fit_by_attr, exec_timer
from .preprocessings import FeaturesTarget, RandomStateInstance
from .vizualizations import save_plot_in_dir

HyperGrid = Union[Sequence[Mapping], Mapping]
SklearnMetric = Callable[..., float]


class OutputFixer:
    """Class for methods used for adjusting grid search object output."""

    @property
    def _model_name(self):
        """Retrieves last estimator name."""
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            model_name = type(self.estimator).__name__
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            last_estimator = self.estimator[last_estimator_name]
            model_name = type(last_estimator).__name__

        return model_name

    @property
    def _relevant_columns(self):
        """Chooses only relevant columns."""
        # Creating the DataFrame from `cv_results_` attribute
        df = pd.DataFrame(self.cv_results_)

        # Choosing specific columns
        columns_to_display = df.columns.str.startswith(("param_", "mean_t"))
        relevant_columns = df.columns[columns_to_display]

        return relevant_columns

    def _fix_col_names(self, df):
        """Fixes columns names in the DataFrame."""
        try:
            pipeline_steps = self.estimator.named_steps
        except AttributeError:
            df.columns = df.columns.str.replace("param_", "")
        else:
            pipeline_steps_names = list(pipeline_steps.keys())
            last_estimator_name = pipeline_steps_names[-1]
            col_name = "param_" + last_estimator_name + "__"
            df.columns = df.columns.str.replace(col_name, "")
        finally:
            df = df.round(4)

        return df


class GridSearchLauncher:
    """Class containing grid search features."""

    @exec_timer
    def train_crossvalidate(self) -> Any:
        """Launches grid search algorithm."""
        self.fit(self.train_features_, self.train_target_)

        model_name = self._model_name

        print(f"Grid search for {model_name} completed.")

    @check_est_fit_by_attr(attr="cv_results_", func_hint="train_crossvalidate")
    def display_tuning_process(self) -> pd.DataFrame:
        """Displays the results of grid search as a DataFrame."""
        # Creating the initial DataFrame from `cv_results_` attribute
        cv_results_df = pd.DataFrame(self.cv_results_)

        # Selecting the relevant columns
        relevant_columns = self._relevant_columns
        cv_results_df = cv_results_df[relevant_columns]

        # Fixing column names in the DataFrame
        cv_results_df = self._fix_col_names(cv_results_df)

        return cv_results_df

    @property
    def best_iter_(self):
        """Displays the best iteration as a Series."""
        # Computing the table with results
        df = self.display_tuning_process()

        # Selecting the best index
        best_iteration = self.best_index_

        # Choosing the best row
        cv_results_best_iter = df.iloc[best_iteration]

        # Retrieving model name (last estimator name)
        model_name = self._model_name

        # Renaming the pd.Series
        cv_results_best_iter = cv_results_best_iter.rename(model_name)

        return cv_results_best_iter


class GridSearchMixin(OutputFixer, GridSearchLauncher):
    """Mixin class for combining methods from parent classes."""


class RandomizedHyperoptClassification(RandomizedSearchCV, GridSearchMixin):
    """Wrapper for RandomizedSearchCV for classification."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: HyperGrid,
        train_dataset: FeaturesTarget,
        eval_dataset: FeaturesTarget,
        cv: Union[Iterable, int] = 2,
        random_state: Union[RandomStateInstance, int, None] = None,
        scoring: str = "f1",
        n_iter: int = 5,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> None:
        """Initializes an instance."""
        super().__init__(
            estimator,
            param_distributions,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_features_, self.train_target_ = self.train_dataset
        self.eval_features_, self.eval_target_ = self.eval_dataset
        self.eval_predictions_ = None

    @property
    @check_est_fit_by_attr(attr="cv_results_", func_hint="train_crossvalidate")
    def classif_stats_(self):
        """Performs computation of classification metrics."""
        # Computing predictions on the test set
        self.eval_predictions_ = self.predict(self.eval_features_)
        # Print out a summary of all classification metrics if metric=None
        classification_stats = classification_report(
            y_true=self.eval_target_,
            y_pred=self.eval_predictions_,
        )
        print(classification_stats)

    @check_est_fit_by_attr(attr="cv_results_", func_hint="train_crossvalidate")
    def plot_confusion_matrix(
        self, labels: tuple = None, save_fig: bool = False
    ) -> Any:
        """Plots a confusion matrix."""
        # Computing predictions on the test set
        self.eval_predictions_ = self.predict(self.eval_features_)

        # Plotting confusion matrix
        sns.reset_defaults()
        cm = confusion_matrix(
            y_true=self.eval_target_, y_pred=self.eval_predictions_
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels
        )
        disp.plot(cmap=plt.cm.Blues)

        # Selecting last estimator name
        model_name = self._model_name

        plt.title(f"Confusion matrix ({model_name})")
        plt.tight_layout()

        if save_fig:
            save_plot_in_dir(file_name="confusion_matrix.png")

        plt.show()


class RandomizedHyperoptRegression(RandomizedSearchCV, GridSearchMixin):
    """Wrapper for RandomizedSearchCV for regressions."""

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: Union[Sequence[Mapping], Mapping],
        train_dataset: FeaturesTarget,
        eval_dataset: FeaturesTarget,
        cv: Union[Iterable, int] = 2,
        random_state: Union[RandomStateInstance, int, None] = None,
        scoring: str = "neg_root_mean_squared_error",
        n_iter: int = 5,
        n_jobs: Optional[int] = None,
        return_train_score: bool = True,
    ) -> None:
        """Initializes an instance."""
        super().__init__(
            estimator,
            param_distributions,
            cv=cv,
            random_state=random_state,
            scoring=scoring,
            n_iter=n_iter,
            n_jobs=n_jobs,
            return_train_score=return_train_score,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_features_, self.train_target_ = self.train_dataset
        self.eval_features_, self.eval_target_ = self.eval_dataset
        self.eval_predictions_ = None

    def display_tuning_process(self) -> pd.DataFrame:
        """Displays the results of grid search."""
        cv_results_df = super().display_tuning_process()

        # Adjusting the sign in case of regression task
        cv_results_df["mean_test_score"] = -cv_results_df["mean_test_score"]

        if self.return_train_score:
            cv_results_df["mean_train_score"] = -cv_results_df[
                "mean_train_score"
            ]

        return cv_results_df

    @check_est_fit_by_attr(attr="cv_results_", func_hint="train_crossvalidate")
    def compute_regression_stats(
        self, metric: Literal["rmse", "mse", "mae"]
    ) -> float:
        """Computes key regression metrics."""
        # Proceed if exception has not been raised
        self.eval_predictions_ = self.predict(self.eval_features_)
        if metric == "rmse":
            rmse = mean_squared_error(
                y_true=self.eval_target_,
                y_pred=self.eval_predictions_,
                squared=False,
            )
            return rmse
        if metric == "mse":
            mse = mean_squared_error(
                y_true=self.eval_target_,
                y_pred=self.eval_predictions_,
                squared=True,
            )
            return mse
        if metric == "mae":
            mae = mean_absolute_error(
                y_true=self.eval_target_, y_pred=self.eval_predictions_
            )
            return mae
        raise ValueError("Incorrect metric name specified")


class SimpleHyperparamsOptimizer:
    """Class for implementing a simple grid search algorithm."""

    def __init__(
        self,
        model: BaseEstimator,
        param_grid: HyperGrid,
        train_dataset: FeaturesTarget,
        eval_dataset: FeaturesTarget,
        scoring_function: SklearnMetric,
        display_search: bool = False,
        higher_better: bool = True,
    ) -> None:
        """Constructor for SimpleHyperparamsOptimizer class."""
        self.model = model
        self.param_grid = param_grid
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.scoring_function = scoring_function
        self.display_search = display_search
        self.higher_better = higher_better

        self.train_features_, self.train_target_ = self.train_dataset
        self.eval_features_, self.eval_target_ = self.eval_dataset

    def _retrieve_combinations(self) -> list[dict[str, Any]]:
        """Retrieves all possible hyperparameter values combinations."""
        keys, values = zip(*self.param_grid.items())
        config_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

        return config_list

    def _log_progress(
        self,
        params: dict[str, Any],
        metric_train: float,
        metric_valid: float,
    ) -> None:
        """Logs the hyperparameter search process."""
        # Retrieving the name of the scoring function
        score_func_name = self.scoring_function.__name__
        # Printing the grid search step
        print(
            f"{params}: {score_func_name}=(train={metric_train:.4f}, valid={metric_valid:.4f})"
        )

    def _evaluate(self, model: BaseEstimator) -> tuple[float, float]:
        """Computes training and validation scores."""
        # Computing predictions for training and validation sets
        predictions_train = model.predict(self.train_features_)
        predictions_valid = model.predict(self.eval_features_)
        # Computing metric for training and validation sets
        metric_score_train = self.scoring_function(
            self.train_target_, predictions_train
        )
        metric_score_valid = self.scoring_function(
            self.eval_target_, predictions_valid
        )

        return metric_score_train, metric_score_valid

    def train(self) -> None:
        """Launches the grid search algorithm."""
        # Setting initial values
        best_model = None
        best_result = 0 if self.higher_better else np.Inf
        # Retrieving a list of all possible hyperparameter combinations
        config_list = self._retrieve_combinations()

        # Launching grid search
        for config in config_list:
            # Reinitializing the model
            model = clone(self.model)
            model.set_params(**config)
            # Training the model
            model.fit(self.train_features_, self.train_target_)
            # Computing training and validation scores
            metric_score_train, metric_score_valid = self._evaluate(
                model=model
            )
            # Logging the grid search process
            if self.display_search:
                self._log_progress(
                    params=config,
                    metric_train=metric_score_train,
                    metric_valid=metric_score_valid,
                )
            # Making a decision about the best model
            if self.higher_better:
                if metric_score_valid > best_result:
                    best_model = model
                    best_config = config
                    best_result = metric_score_valid
            else:
                if metric_score_valid < best_result:
                    best_model = model
                    best_config = config
                    best_result = metric_score_valid
        # Saving the information about the best model in attributes
        self.best_model_ = best_model
        self.best_config_ = best_config
        self.best_result_ = best_result
