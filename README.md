# Rearden

**Rearden** is a Python package that provides a faster way of carrying out data science and running machine learning algorithms in a more convenient way. Making use of the functionality of the most common libraries for data analysis (`pandas`, `numpy`, `statsmodels`), data vizualization (`matplotlib`, `seaborn`) and grid search (`scikit-learn`), it enables reaching the conclusions in a more convenient and faster way.

----

## Modules
The package (as of *v0.0.1*) is dedicated to the following areas of Data Science/Machine Learning:

* Data preprocessing
* Data vizualization
* Time-series analysis
* Grid search

Hence, the data structures which make up the **Rearden** package have been logically divided into Python modules based off of the above respective parts:

* `preprocessings.py`
* `vizualizations.py`
* `time_series.py`
* `grid_search.py`

## Installation

### Package dependencies

**Rearden** library requires the following dependencies:

| Package | Version |
| :---------------------- | :---------------------- |
| Matplotlib | >= 3.3.4|
| Pandas | >= 1.2.4|
| NumPy| >= 1.24.3|
| Scikit-learn| >= 1.1.3|
| Seaborn| >= 0.11.1|
| Statsmodels| >= 0.13.2|

> **_NOTE:_**  The package currently requires Python 3.7 or higher.

### Installation using `pip`

The package is available on [PyPI Index](https://pypi.org/project/rearden/) and can be easily installed using `pip`:

```
pip install rearden
```

The dependencies are automatically downloaded when executing the above command or can be installed manually using:

```
pip install -r requirements.txt
```

## Building the package

Thank to the build system requirements and other metadata specified in `pyproject.toml` it is easy to build and install the package. Firstly, clone the repository:

```
git clone https://github.com/spolivin/rearden.git

cd rearden
```

Then, one can simply run the following:

```
pip install -e .
```

## Automatic code style checks

### Installation of `pre-commit`
Before pushing the changed code to the remote Github repository, the code undergoes numerous checks conducted with the help of *pre-commit hooks* specified in `.pre-commit-config.yaml`. Before making use of this feature, it is important to first download `pre-commit` package to the system:

```
pip install pre-commit
```

or if `rearden` package has already been installed:

```
pip install rearden[precommit]
```

Afterwards, in the git-repository run the following command for installation:

```
pre-commit install
```

Now, the *pre-commit hooks* can be easily used for verifying the code style.

### Pre-commit hooks

After running `git commit -m "<Commit message>"` in the terminal, the file to be committed goes through a few checks before being enabled to be committed. As specified in `.pre-commit-config.yaml`, the following hooks are used:

| Hooks | Version |
| :---------------------- | :---------------------- |
| Pre-commit-hooks | 4.3.0 |
| Autoflake | 2.1.1 |
| Isort | 5.12.0 |
| Black | 23.3.0 |
| Flake8 | 5.0.0|

> **_NOTE:_** Check `.pre-commit-config.yaml` for more information about the repos and hooks used.

It is also possible to download the required dependencies for pre-commit hooks:

```
pip install -r requirements-dev.txt
```

or:

```
pip install rearden[formatters]

pip install rearden[linters]
```
