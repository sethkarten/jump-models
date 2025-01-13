# merge.py
import os

def concat_python_files():
    output_file = "concatenated_output.py"
    
    with open(output_file, "w") as outfile:
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as infile:
                        outfile.write(f"# {file}\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")

    print(f"All Python files have been concatenated into {output_file}")

# Call the function
concat_python_files()


# concatenated_output.py


# utils_dir.py
"""
Helpers for working with file directories.

Useful for all scripts/notebooks in this folder. 
Please ensure the file structure under `example/Nasdaq` is preserved 
in its original form for everything to function properly.
"""

import sys, os

def get_curr_dir():
    """
    Return the current directory of this `get_data.py` file.
    """
    return os.path.dirname(os.path.abspath(__file__))

def include_home_dir():
    """
    Add the project's home directory to `sys.path`.

    This function ensures that the home directory of the project is included in 
    `sys.path` to allow imports from other parts of the project. For this to work 
    correctly, the script must be placed in the `example/Nasdaq/` folder.
    """
    curr_dir = get_curr_dir()
    home_dir = os.path.dirname(os.path.dirname(curr_dir))
    sys.path.append(home_dir)
    return

# feature.py
"""
Helpers for engineering the features to be input to JMs.
"""

from utils_dir import *
include_home_dir()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from jumpmodels.utils import *

############################################
## Feature Engineering
############################################

# reviewed
def compute_ewm_DD(ret_ser: pd.Series, hl: float) -> pd.Series:
    """
    Compute the exponentially weighted moving downside deviation (DD) for a return series.

    The downside deviation is calculated as the square root of the exponentially 
    weighted second moment of negative returns.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series.

    hl : float
        The halflife parameter for the exponentially weighted moving average.

    Returns
    -------
    pd.Series
        The exponentially weighted moving downside deviation for the return series.
    """
    ret_ser_neg: pd.Series = np.minimum(ret_ser, 0.)
    sq_mean = ret_ser_neg.pow(2).ewm(halflife=hl).mean()
    return np.sqrt(sq_mean)

# reviewed
def feature_engineer(ret_ser: pd.Series, ver: str = "v0") -> pd.DataFrame:
    """
    Engineer a set of features based on a return series.

    This function customizes the feature set according to the specified version string.

    Parameters
    ----------
    ret_ser : pd.Series
        The input return series for feature engineering.

    ver : str
        The version of feature engineering to apply. Only supports "v0".
    
    Returns
    -------
    pd.DataFrame
        The engineered feature set.
    """
    if ver == "v0":
        feat_dict = {}
        hls = [5, 20, 60]
        for hl in hls:
            # Feature 1: EWM-ret
            feat_dict[f"ret_{hl}"] = ret_ser.ewm(halflife=hl).mean()
            # Feature 2: log(EWM-DD)
            DD = compute_ewm_DD(ret_ser, hl)
            feat_dict[f"DD-log_{hl}"] = np.log(DD)
            # Feature 3: EWM-Sortino-ratio = EWM-ret/EWM-DD 
            feat_dict[f"sortino_{hl}"] = feat_dict[f"ret_{hl}"].div(DD)
        return pd.DataFrame(feat_dict)

    # try out your favorite feature sets
    else:
        raise NotImplementedError()

############################################
## DataLoader Class
############################################

class DataLoader(BaseEstimator):
    """
    Class for loading the feature matrix.

    This class loads raw return data, computes features, and filters the data by date.
    
    Parameters
    ----------
    ticker : str
        The ticker symbol for which to load data. Only supports "NDX".

    ver : str
        The version of the feature set to apply. Only supports "v0".

    Attributes
    ----------
    X : pd.DataFrame
        The feature matrix.
    
    ret_ser : pd.Series
        The return series.
    """
    def __init__(self, ticker: str = "NDX", ver: str = "v0"):
        self.ticker = ticker
        self.ver = ver
    
    # reviewed
    def load(self, start_date: DATE_TYPE = None, end_date: DATE_TYPE = None):
        """
        Load the raw return data, compute features, and filter by date range.

        Parameters
        ----------
        start_date : DATE_TYPE, optional
            The start date for filtering the data. If None, no start filtering is applied.

        end_date : DATE_TYPE, optional
            The end date for filtering the data. If None, no end filtering is applied.

        Returns
        -------
        self
            The DataLoader instance with the feature matrix and return series stored in attributes.
        """
        # load raw data
        curr_dir = get_curr_dir()
        ret_ser_raw = pd.read_pickle(f"{curr_dir}/data/{self.ticker}.pkl").ret.dropna()
        ret_ser_raw.name = self.ticker
        # features
        df_features_all = feature_engineer(ret_ser_raw, self.ver)
        
        # filter date
        X = filter_date_range(df_features_all, start_date, end_date)
        valid_no_nan(X)
        # save attributes
        self.X = X
        self.ret_ser = filter_date_range(ret_ser_raw, start_date, end_date)
        # save more useful attributes if needed
        return self


# get_data.py
"""
This script retrieves the daily closing price data for 
the Nasdaq-100 index from Yahoo Finance via its Python API.

Users do not need to run this script manually, as the return data 
is already saved in `example/Nasdaq/data/`. 
"""

from utils_dir import *
include_home_dir()

import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf

from jumpmodels.utils import check_dir_exist

TICKER = "NVDA"   # Nasdaq-100 Index

def get_data():
    # download closing prices
    # close: pd.Series = yf.download("^"+TICKER, start="1985-10-01", end="2024-12-26")['Close']
    close: pd.Series = yf.download(TICKER, start="1985-10-01", end="2025-01-10")['Close']
    # convert to ret
    ret = close.pct_change()
    # concat as df
    try:
        df = pd.DataFrame({"open": close.values.reshape(-1), "ret": ret.values.reshape(-1)}, index=close.index.date)
    except ValueError as e:
        print(f"Error creating DataFrame: {e}")
        print(f"Close shape: {close.shape}, Return shape: {ret.shape}")

    df.index.name = "date"

    # save
    curr_dir = get_curr_dir()
    data_dir = f"{curr_dir}/data/"; check_dir_exist(data_dir)
    pd.to_pickle(df, f"{data_dir}{TICKER}.pkl")
    np.round(df, 6).to_csv(f"{data_dir}{TICKER}.csv")
    print("Successfully downloaded data for ticker:", TICKER)
    return 

def pickle():
    # Define the file paths
    curr_dir = get_curr_dir()
    data_dir = Path(f"{curr_dir}/data/")
    csv_file_path = data_dir / f"{TICKER}.csv"
    pickle_file_path = data_dir / f"{TICKER}.pkl"

    # Read the CSV file
    df = pd.read_csv(csv_file_path, index_col="date", parse_dates=True)

    # Ensure the index is of type date
    df.index = pd.to_datetime(df.index).date

    # Calculate returns if not already present
    if "ret" not in df.columns:
        df["ret"] = df["open"].pct_change()

    # Round the values to 6 decimal places
    df = np.round(df, 6)

    # Save as pickle file
    df.to_pickle(pickle_file_path)

    print(f"Successfully saved {TICKER} data as pickle file.")

if __name__ == "__main__":
    # get_data()
    pickle()

# example.py
#!/usr/bin/env python
# coding: utf-8

# In[1]:

from utils_dir import get_curr_dir, include_home_dir
include_home_dir()

import pandas as pd

from jumpmodels.utils import filter_date_range        # useful helpers
from jumpmodels.jump import JumpModel                 # class of JM & CJM
from jumpmodels.sparse_jump import SparseJumpModel    # class of Sparse JM


# # Load Data & Features
# 
# This example demonstrates the class of *statistical jump models* (JMs) and various helper functions for regime analysis provided by our package `jumpmodels`, using an application on the Nasdaq-100 Index. 
# The core classes, `JumpModel` and `SparseJumpModel`, implement the original JM, continuous JM (CJM), and sparse JM (SJM) with feature selection.
# These models follow the API style used in `scikit-learn` for easy integration and efficient usage.
# For detailed mathematical and algorithmic explanations of these models, please refer to the literature cited in the `README`.
# 
# Relevant helper functions will be imported as needed throughout this example.
# If running this notebook in Jupyter Lab/Notebook poses any issues, there is an exported `.py` script available in this folder for convenient execution.
# 

# ## Raw Data
# 
# In this example, we analyze the regimes of the Nasdaq-100 Index.
# The daily index price data is retrieved from [Yahoo Finance](https://finance.yahoo.com/quote/%5ENDX/) under the ticker `NDX`. 
# 
# The data retrieval is handled in the script `get_data.py`, and the dataset is already saved in the `example/Nasdaq/data/` folder in both `csv` and `pkl` formats, so there’s no need to run `get_data.py` manually.
# 
# We work with daily frequency data using `pandas` DataFrames, where the index is of type `datetime.date`. 
# This format is consistent with the convention used in the `CRSP` database. 
# All helper functions in this package are designed to support this type of date index.

# ## Feature Engineering
# 
# Feeding the model a robust feature set is key to the successful application of any learning algorithm. 
# This example uses a simple feature set consisting of nine features: the exponentially weighted moving (EWM) return, downside deviation (in log scale), and Sortino ratio, each computed with three halflife values ranging from one week (5 days) to one quarter (3 months). 
# 
# Users may need to adjust the features or halflives to suit their specific applications. 
# The literature referenced in the `README` offers a solid foundation for further exploration.
# 
# The computation of these features is detailed in `feature.py` in the same folder as this example, and we use the `DataLoader` class to load both the index returns and the engineered features.
# The loaded data covers the period from the start of 2007 to the end of September 2024.

# In[2]:


from feature import DataLoader

data = DataLoader(ticker="NDX", ver="v0").load(start_date="2007-1-1", end_date="2024-09-30")

print("Daily returns stored in `data.ret_ser`:", "-"*50, sep="\n")
print(data.ret_ser, "-"*50, sep="\n")
print("Features stored in `data.X`:", "-"*50, sep="\n")
print(data.X)


# ## Train/Test Split and Preprocessing
# 
# We perform a simple time-based split: data from the beginning of 2007 to the end of 2021, covering a 15-year period, is used as the training set for fitting the JMs.
# The period from 2022 to late 2024 is reserved as the test set, where we apply the trained JMs to perform online regime inference.
# We use the helper function `filter_date_range` to filter the start and end dates of a DataFrame.

# In[3]:


train_start, test_start = "2007-1-1", "2022-1-1"
# filter dates
X_train = filter_date_range(data.X, start_date=train_start, end_date=test_start)
X_test = filter_date_range(data.X, start_date=test_start)
# print time split
train_start, train_end = X_train.index[[0, -1]]
test_start, test_end = X_test.index[[0, -1]]
print("Training starts at:", train_start, "and ends at:", train_end)
print("Testing starts at:", test_start, "and ends at:", test_end)


# The module `jumpmodels.preprocess` provides two classes for preprocessing: one for standardizing and one for clipping the feature data. 
# We first clip the data within three standard deviations for all features and then perform standardization before feeding the data into the JMs. 
# Both classes are first fitted on the training data and subsequently used to transform the test data.
# 
# These classes support both `pandas` DataFrames and `numpy` arrays as direct inputs and outputs. 
# We prefer to retain the DataFrame type whenever possible to preserve the date index and column labels.

# In[4]:


# Preprocessing
from jumpmodels.preprocess import StandardScalerPD, DataClipperStd
clipper = DataClipperStd(mul=3.)
scalar = StandardScalerPD()
# fit on training data
X_train_processed = scalar.fit_transform(clipper.fit_transform(X_train))
# transform the test data
X_test_processed = scalar.transform(clipper.transform(X_test))


# # Original JM

# ## In-Sample Fitting
# 
# We begin by illustrating the in-sample training of the original JM.
# The model parameters are set as follows: the number of components/states/regimes is 2, the jump penalty $\lambda$ is 50.0, and `cont=False`, indicating the original discrete JM that performs hard clustering. 
# It is important to note that the jump penalty $\lambda$ is a crucial hyperparameter that requires tuning, either through statistical criteria or cross-validation (see references for details). 
# 
# The docstring provides comprehensive documentation of all parameters and attributes (thanks to ChatGPT).

# In[5]:


# set the jump penalty
jump_penalty=50.
# initlalize the JM instance
jm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=False, )


# In the `.fit()` call, we pass the return series for each period to be used for sorting the states.
# We specify `sort_by="cumret"`, meaning that the state labels (0 or 1) are determined by the cumulative returns under each state. The state with higher cumulative returns is denoted as $s_t=0$ (bull market), and the state with lower returns is denoted as $s_t=1$ (bear market). 
# 

# In[6]:


# call .fit()
jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")


# The cluster centroids for each state are stored in the `centers_` attribute. 
# While these values are scaled, making direct interpretation hard, the bull market state is clearly characterized by higher returns, lower downside deviation, and a higher Sortino ratio, with a distinct separation between the two regimes.

# In[7]:


print("Scaled Cluster Centroids:", pd.DataFrame(jm.centers_, index=["Bull", "Bear"], columns=X_train.columns), sep="\n" + "-"*50 + "\n")


# ### Visualization
# 
# The `jumpmodels.plot` module provides useful functions for visualizing regime identification. 
# We'll use the `labels_` attribute of the JM instance, which contains integers from 0 to `n_c-1`, representing the in-sample fitted regime assignment for each period.
# 
# From the plot, we observe that the identified regimes for the Nasdaq-100 Index successfully capture several significant market downturns, including the global financial crisis, corrections in 2012, 2015-2016, 2019, and the COVID-19 crash in 2020. 
# These identified regimes correspond well to shifts in market fundamentals, as interpreted in hindsight.
# 

# In[8]:


from jumpmodels.plot import plot_regimes_and_cumret, savefig_plt

ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_train.pdf")


# ### Modifying Parameters via `set_params()`
# 
# Our model inherits from the `BaseEstimator` class provided by `scikit-learn`, enabling a wide range of utility methods.
# Among these, we highlight the `.set_params()` function, which allows users to reset any input parameters without creating a new instance.
# This functionality is particularly useful when the model needs to be refitted multiple times, such as when testing different jump penalties.
# 
# As an example, we reset the jump penalty to zero, effectively reducing the model to a baseline $k$-means clustering algorithm where temporal information is ignored. 
# This comparison illustrates the value of applying a jump penalty to ensure temporal consistency and reduce the occurrence of unrealistic regime shifts.

# In[9]:


# reset jump_penalty to zero
jump_penalty=0.
jm.set_params(jump_penalty=jump_penalty)
print("The jump penalty of the JM instance has been reset to: jm.jump_penalty =", jm.jump_penalty)


# In[10]:


# refit
jm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

# plot
ax, ax2 = plot_regimes_and_cumret(jm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_train.pdf")


# ## Online Inference
# 
# After completing the in-sample training, we apply the trained models for online inference on the test period using the `predict_online()` method. 
# Here, *online inference* means that the regime inference for period $t$ is based solely on the data available up to the end of that period, without using any future data.
# We revert the jump penalty to a reasonable value of 50.0.
# 
# 

# In[11]:


# refit
jump_penalty=50.
jm.set_params(jump_penalty=jump_penalty).fit(X_train_processed, data.ret_ser, sort_by="cumret")
# make online inference 
labels_test_online = jm.predict_online(X_test_processed)


# From the visualization below, we observe that the JM effectively signals the bear market in 2022, driven by aggressive interest rate hikes. 
# This period saw a return of over $-$15% and a significant drawdown.
# However, the brief bear period captured in the second half of 2024 is followed by a strong price reversal.
# This latency issue constitutes a common challenge in real-time applications of regime-switching signals.
# Improving the feature set or fine-tuning the jump penalty may help address this issue.

# In[12]:


# plot and save
ax, ax2 = plot_regimes_and_cumret(labels_test_online, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the JM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/JM_lambd-{jump_penalty}_test_online.pdf")


# In contrast to online inference, the `.predict()` method performs state decoding using all test data (i.e., from 2022 to 2024) at once. 
# While this approach is less realistic for trading applications, we observe that, with access to the full dataset, the model avoids the reversal in late 2024 and exits the bear signal in 2023 slightly earlier than with online inference.
# 
# Though this approach is less applicable for real-world backtesting in financial markets, it holds potential uses in other engineering fields (such as language modeling, where access to an entire sentence is available at once.)

# In[13]:


# make inference using all test data
labels_test = jm.predict(X_test_processed)
# plot
ax, ax2 = plot_regimes_and_cumret(labels_test, data.ret_ser, n_c=2, start_date=test_start, end_date=test_end, )
_ = ax.set(title=f"Out-of-Sample Predicted Regimes by the JM Using All Test Data ($\\lambda$={jump_penalty})")


# # CJM: Continuous Extension of the JM
# 
# With this, we conclude a minimal overview of the core functionality of using JMs to assign regime labels to in-sample training periods and leverage trained models for out-of-sample prediction, either through online inference or by processing all data at once. 
# The methods -- such as `.fit()`, `.set_params()`, and `predict_online()` -- extend seamlessly to the following JM variants: CJM and SJM. 
# Here, we provide brief illustrations of these extensions.

# ## In-Sample Fitting
# 
# The CJM (Continuous Jump Model) uses the same `JumpModel` class as the discrete model, with the parameter `cont=True`. 
# 
# ### Parameters 
# 
# Regarding the jump penalty value, it is typically set to be 10 times larger than the $\lambda$ used in the discrete model to achieve similar fittings, so we choose $\lambda=600.0$ here.
# 
# Additionally, CJM introduces two specialized parameters: `mode_loss` and `grid_size`, which require more nuanced understanding. 
# Generally, the default values are recommended for most cases.
# 

# In[14]:


jump_penalty=600.
cjm = JumpModel(n_components=2, jump_penalty=jump_penalty, cont=True)


# The `proba_` attribute of the CJM instance stores the estimated probability of each period belonging to each state.
# Unlike the discrete model, where the state assignment changes abruptly, CJM offers smooth probability transitions, ranging from 0% to 100%. 
# This probabilistic interpretation has potential applications in many domains, especially where softer regime assignments are beneficial.

# In[15]:


cjm.fit(X_train_processed, data.ret_ser, sort_by="cumret")

# plot
ax, ax2 = plot_regimes_and_cumret(cjm.proba_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax2.set(ylabel="Regime Probability")
ax.set(title=f"In-Sample Fitted Regimes by the CJM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/CJM_lambd-{jump_penalty}_train.pdf")


# ## Online Inference
# 
# The `.predict_proba_online()` method allows CJM to make probabilistic regime inferences online.
# From the plot, we observe that the confidence in the bear market during late 2024 doesn't fully reach 100%, potentially reducing the mislabeling issue discussed earlier. 
# This smoother transition in probabilities may offer better regime detection in uncertain market conditions.

# In[16]:


# online inference
proba_test_online = cjm.predict_proba_online(X_test_processed)

# plot
ax, ax2 = plot_regimes_and_cumret(proba_test_online, data.ret_ser, start_date=test_start, end_date=test_end, )
ax2.set(ylabel="Regime Probability")
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the CJM ($\\lambda$={jump_penalty})")
savefig_plt(f"{get_curr_dir()}/plots/CJM_lambd-{jump_penalty}_test_online.pdf")


# # SJM: Sparse JM with Feature Selection
# 
# Finally, the Sparse Jump Model (SJM) introduces feature weighting on top of the original JM or CJM. 
# Features leading to better in-sample clustering effects, as measured by variance reduction, are assigned higher weights, while a LASSO-like constraint on the weight vector ensures that noisy features receive zero weight.
# 
# ## In-Sample Fitting
# 
# ### Parameters
# 
# SJM is implemented in the class `SparseJumpModel`, with an additional parameter `max_feats`, which controls the number of features included.
# This parameter roughly reflects the effective number of features. (In the notation of Nystrup et al. (2021), `max_feats` corresponds to $\kappa^2$.)
# 
# The jump penalty value is of a similar magnitude to the non-sparse model. In this case, we try `max_feats=3.` and `jump_penalty=50.`

# In[17]:


max_feats=3.
jump_penalty=50.
# init sjm instance
sjm = SparseJumpModel(n_components=2, max_feats=max_feats, jump_penalty=jump_penalty, )
# fit
sjm.fit(X_train_processed, ret_ser=data.ret_ser, sort_by="cumret")


# The feature weights are stored in the attribute `feature_weights`. 
# Generally, we observe that features with longer halflives receive higher weights, indicating that less smoothed features are noisier and are excluded from the model, thanks to the feature weighting mechanism.

# In[18]:


print("SJM Feature Weights:", "-"*50, sjm.feat_weights, sep="\n")


# A comparison of the SJM-identified regimes with those identified by JM reveals that the GFC is consolidated into a single bear regime, demonstrating that short-term noise has been effectively mitigated.

# In[19]:


# plot
ax, ax2 = plot_regimes_and_cumret(sjm.labels_, data.ret_ser, n_c=2, start_date=train_start, end_date=train_end, )
ax.set(title=f"In-Sample Fitted Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_train.pdf")


# ## Online Inference
# 
# As before, the `.predict_online()` method handles online inference. 
# Notably, through feature selection, the previously problematic bear market signal in late 2024 is absent in the SJM's online inference, highlighting the potential benefits of feature selection.
# 
# 

# In[20]:


# online inference
labels_test_online_sjm = sjm.predict_online(X_test_processed)

# plot
ax, ax2 = plot_regimes_and_cumret(labels_test_online_sjm, data.ret_ser, start_date=test_start, end_date=test_end, )
ax.set(title=f"Out-of-Sample Online Inferred Regimes by the SJM ($\\lambda$={jump_penalty}, $\\kappa^2$={max_feats})")
savefig_plt(f"{get_curr_dir()}/plots/SJM_lambd-{jump_penalty}_max-feats-{max_feats}_test_online.pdf")


# # Conclusion
# 
# This concludes the introduction to the functionalities of our `jumpmodels` library. 
# The field of statistical jump models is still actively evolving, with ongoing research exploring new avenues.
# We hope that the models and helper functions provided in this package will be useful in your own work. 
# Citations and credits are always appreciated.
# 
# We welcome pull requests and open issues, and I’m happy to discuss any related questions.

# In[ ]:






# plot.py
"""
Module for plotting functions, especially for visualizing regime identification.

Depends on:
-----------
utils/ : Modules
"""

from .utils import *

import matplotlib.pyplot as plt

ALPHA_LINE = .8
ALPHA_FILL = .3
AXES_TYPE = Optional[plt.Axes]

############################
## matplotlib setting
############################

# reviewed
def matplotlib_setting():
    """
    Set global rcParams for matplotlib to produce nice and large publication-quality figures.
    """
    plt.rcParams['figure.figsize'] = (24, 12)
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
    plt.rcParams['legend.fontsize'] = 30
    plt.rcParams['font.size'] = 26
    plt.rcParams['font.family'] = 'cmr10'
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"
    return 

# Set global matplotlib params
matplotlib_setting()

########################################################
## File I/O + Function I/O
########################################################

# reviewed
def savefig_plt(filepath, close=False):
    """
    Save the current figure to a specified path. Automatically creates the folder if it doesn't exist.

    Parameters
    ----------
    filepath : str
        The path where the figure should be saved.

    close : bool, optional (default=False)
        Whether to close the figure after saving.
    """
    check_dir_exist(filepath)
    print(filepath)
    plt.savefig(filepath)
    if close: plt.close()
    return 

# reviewed
def check_axes(ax: AXES_TYPE = None, nrows=1, ncols=1, figsize_single=(24, 12), **kwargs) -> Union[plt.Axes, np.ndarray]:
    """
    Create a new axes if `ax` is None; otherwise return the existing axes.

    Parameters
    ----------
    ax : plt.Axes, optional
        An existing matplotlib Axes object. If None, a new one is created.

    nrows : int, optional (default=1)
        Number of rows for the subplots.

    ncols : int, optional (default=1)
        Number of columns for the subplots.

    figsize_single : tuple, optional (default=(24, 12))
        The size of a single subplot.

    Returns
    -------
    plt.Axes or np.ndarray
        The axes object(s) for plotting.
    """
    if ax is None:
        w, h = figsize_single
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*w, nrows*h), **kwargs)
    return ax

########################################################
## Plotting Cumulative Returns
########################################################

# Convert y-axis to percent format
from matplotlib.ticker import FuncFormatter

def convert_yaxis_to_percent(ax: plt.Axes) -> None:
    """
    Convert the ticks on the y-axis to percent without decimals (e.g., 4.0 becomes 400%).
    """
    def to_percent(x, position): 
        pos_flag = x >= 0 
        string = f"{abs(x) * 100:.0f}\\%"
        if pos_flag: return string
        return "$-$" + string
    ax.yaxis.set_major_formatter(FuncFormatter(to_percent))
    return 

# reviewed
def plot_cumret(ret_df: Union[PD_TYPE, dict], 
                start_date: DATE_TYPE = None, 
                end_date: DATE_TYPE = None, 
                ax: AXES_TYPE = None, 
                ylabel_ret="Cumulative Returns",
                ) -> plt.Axes:
    """
    Plot the cumulative returns from a return DataFrame or dictionary.

    Parameters
    ----------
    ret_df : DataFrame or dict
        The input return data for computing cumulative returns.

    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.

    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.

    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.

    ylabel_ret : str, optional (default="Cumulative Returns")
        The label for the y-axis.

    Returns
    -------
    plt.Axes
        The axes object with the plotted cumulative returns.
    """
    ax = check_axes(ax)
    # Process and filter the return data
    ret_df = filter_date_range(pd.DataFrame(ret_df), start_date, end_date)
    ret_df.index.name = None
    # plot cumret
    ret_df.cumsum(axis=0).plot(ax=ax)
    # set ax attrs
    ax.set(ylabel=ylabel_ret)
    convert_yaxis_to_percent(ax)
    return ax

############################
## plot regimes
############################

# reviewed
def fill_between(ser: pd.Series,
                 start_date: DATE_TYPE = None, 
                 end_date: DATE_TYPE = None, 
                 ax: AXES_TYPE = None, 
                 color: Optional[str] = None, 
                 fill_between_label: Optional[str] = None) -> plt.Axes:
    """
    Fill the area between a curve and the x-axis with a specified color and label.

    Parameters
    ----------
    ser : pd.Series
        The data series to plot.

    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.

    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.

    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.

    color : str, optional
        The fill color. Defaults to None.

    fill_between_label : str, optional
        The label for the filled area. Defaults to None.

    Returns
    -------
    plt.Axes
        The axes object with the filled area plot.
    """
    ax = check_axes(ax)
    # filter dates
    ser = filter_date_range(ser, start_date, end_date)
    # plot
    ax.fill_between(ser.index, ser, step="pre", alpha=ALPHA_FILL, color=color, label=fill_between_label)
    ax.legend()
    return ax

# reviewed
def plot_regimes(regimes: PD_TYPE, 
                 n_c: int = 2, 
                 start_date: DATE_TYPE = None, 
                 end_date: DATE_TYPE = None, 
                 ax: AXES_TYPE = None, 
                 colors_regimes: Optional[list] = ['g', 'r'], 
                 labels_regimes: Optional[list] = ['Bull', 'Bear'],
                 ) -> plt.Axes:
    """
    Plot regime identification based on a 1D label series or 2D probability matrix.

    Parameters
    ----------
    regimes : DataFrame or Series
        The regime data to plot. A integer sequence from {0, 1, ..., n_c-1} if 1D input, 
        or a probability matrix of shape (n_s, n_c)

    n_c : int, optional (default=2)
        The number of components (regimes) to plot.

    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.

    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.

    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.

    colors_regimes : list, optional
        The colors for the regimes. Defaults to ['g', 'r'] (`n_c = 2`).
        if `None`, colors will be automatically generated.

    labels_regimes : list, optional
        The labels for the regimes. Defaults to ['Bull', 'Bear'] (`n_c = 2`).
        if `None`, labels will be automatically generated.

    Returns
    -------
    plt.Axes
        The axes object with the regime plot.
    """
    regimes = filter_date_range(regimes, start_date, end_date)
    if is_ser(regimes):
        regimes = pd.DataFrame(raise_labels_into_proba(regimes.to_numpy(), n_c=n_c), index=regimes.index)
    assert regimes.shape[1]==n_c, "Mismatch between number of components and regime data shape."
    ax = check_axes(ax)
    # color list
    if colors_regimes is None:  # generate color list
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors_regimes = [color_cycle[i % len(color_cycle)] for i in range(n_c)]
    else:
        assert len(colors_regimes) == n_c, "Mismatch between length of color list and number of components. You can input `colors_regimes = None` for colors to be generated authomatically."
    # labels
    if labels_regimes is None:
        labels_regimes = [f"Regime {i}" for i in range(1, n_c+1)]
    else:
        assert len(labels_regimes) == n_c, "Mismatch between length of label list and number of components. You can input `labels_regimes = None` for labels to be generated authomatically."
    # plot
    for i in range(n_c):
        fill_between(regimes.iloc[:, i], ax=ax, color=colors_regimes[i], fill_between_label=labels_regimes[i])
    return ax

# reviewed
def plot_regimes_and_cumret(regimes: PD_TYPE, 
                            ret_df: Union[PD_TYPE, dict], 
                            n_c: int = 2, 
                            start_date: DATE_TYPE = None, 
                            end_date: DATE_TYPE = None, 
                            ax: AXES_TYPE = None, 
                            colors_regimes: Optional[list] = ['g', 'r'], 
                            labels_regimes: Optional[list] = ['Bull', 'Bear'],
                            ylabel_ret="Cumulative Returns",
                            legend_loc="upper left"
                            ) -> tuple[plt.Axes, plt.Axes]:
    """
    Plot cumulative returns alongside regime identification in a single figure.

    Parameters
    ----------
    regimes : DataFrame or Series
        The regime data to plot. A integer sequence from {0, 1, ..., n_c-1} if 1D input, 
        or a probability matrix of shape (n_s, n_c)

    ret_df : DataFrame or dict
        The return data to plot.

    n_c : int, optional (default=2)
        The number of regimes/components.

    start_date : str or datetime.date, optional
        The start date for the plot. Defaults to None.

    end_date : str or datetime.date, optional
        The end date for the plot. Defaults to None.

    ax : plt.Axes, optional
        The axes on which to plot. If None, a new one is created.

    colors_regimes : list, optional
        The colors for the regimes. Defaults to ['g', 'r'] (`n_c = 2`).
        if `None`, colors will be automatically generated.

    labels_regimes : list, optional
        The labels for the regimes. Defaults to ['Bull', 'Bear'] (`n_c = 2`).
        if `None`, labels will be automatically generated.

    ylabel_ret : str, optional
        The label for the cumulative return y-axis.

    legend_loc : str, optional
        The location of the legend.

    Returns
    -------
    tuple
        The axes objects for cumulative returns and regimes.
    """
    # plot cumret
    ax = plot_cumret(ret_df, start_date=start_date, end_date=end_date, ax=ax, ylabel_ret=ylabel_ret)
    # plot regimes
    ax2 = ax.twinx()
    ax2.set(ylabel="Regime")
    plot_regimes(regimes, n_c, start_date=start_date, end_date=end_date, ax=ax2, colors_regimes=colors_regimes, labels_regimes=labels_regimes)
    # merge legneds
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc=legend_loc)
    return (ax, ax2)


# sparse_jump.py
"""
Module for Sparse Jump Models (SJMs).

This module provides an implementation of sparse jump models, extending the jump model 
with additional support for feature selection through Lasso-like optimization.

Depends on
----------
utils/ : Modules
    Utility functions for validation and clustering operations.
jump : Module
    Discrete and continuous jump models.
"""

from .utils import *
from .jump import *

from numpy.linalg import norm

########################################################
## Lasso Problem for Feature Weights
########################################################

# reviewed
def binary_search_decrease(func, 
                           left: float, 
                           right: float, 
                           value: float, 
                           *args, 
                           tol_x: float = 1e-8, 
                           tol_y: float = 0., 
                           max_iter: int = 100, 
                           verbose: int = 0,
                           **kwargs) -> float:
    """
    Binary search for a decreasing function.

    This method performs binary search to find the point where the function `func` 
    decreases to a specified value within given tolerances.

    Parameters
    ----------
    func : callable
        The function to be minimized.
    
    left : float
        The left bound for the search.

    right : float
        The right bound for the search.

    value : float
        The target value to find.

    tol_x : float, optional (default=1e-8)
        The tolerance for the search along the x-axis.

    tol_y : float, optional (default=0.)
        The tolerance for the search along the y-axis (function value).

    max_iter : int, optional (default=100)
        Maximum number of iterations.

    verbose : int, optional (default=0)
        Verbosity level. If greater than 0, prints progress information.

    Returns
    -------
    float
        The optimal point where the function reaches the target value.
    """
    if value >= func(left): return left
    if value <= func(right): return right
    # 
    gap = right-left
    num_iter = 0
    while (gap > tol_x and num_iter < max_iter):
        # print(f"{left}, {right}")
        num_iter += 1
        middle = (right + left) / 2
        func_call = func(middle, *args, **kwargs)
        if verbose: print("x value", middle, "y value", func_call)
        if func_call < value-tol_y/2:
            right = middle
        elif func_call > value+tol_y/2:
            left = middle
        else:
            return middle
        gap /= 2
    if num_iter < max_iter:
        return middle
    raise Exception("Non-convergence: Possible mathematical error.")
    
# reviewed
def soft_thres_l2_normalized(x: SER_ARR_TYPE, thres: float = 0.) -> SER_ARR_TYPE:
    """
    Soft thresholding for a non-negative vector `x`, followed by L2 normalization.

    Parameters
    ----------
    x : Series or ndarray
        The input vector to be thresholded and normalized.

    thres : float, optional (default=0.)
        The threshold for soft thresholding.

    Returns
    -------
    Series or ndarray
        The thresholded and L2-normalized vector.
    """
    y = np.maximum(0, x-thres)
    y_norm = norm(y)
    assert y_norm > 0
    return y / y_norm

# reviewed
def solve_lasso(a: SER_ARR_TYPE, 
                norm_ub: float, 
                tol: float = 1e-8) -> SER_ARR_TYPE:
    """
    Solve the Lasso problem for feature weights.

    This function finds the optimal feature weights subject to the constraint that the 
    L1-norm of the weights is bounded by `norm_ub`.

    Parameters
    ----------
    a : Series or ndarray
        The input vector for the Lasso problem.

    norm_ub : float
        The upper bound for the L1-norm of the feature weights. 
        Equals to `kappa` in the published articles.

    tol : float, optional (default=1e-8)
        The tolerance for the binary search.

    Returns
    -------
    Series or ndarray
        The optimized feature weights.
    """
    assert norm_ub >= 1.
    a_arr = check_1d_array(a)
    left, right = 0., np.unique(a_arr)[-2]  # right is the second largest element of `a`
    if right < tol: thres_sol = 0.
    else:
        func = lambda thres: soft_thres_l2_normalized(a_arr, thres).sum()
        thres_sol = binary_search_decrease(func, left, right, norm_ub, tol_x=tol)
    # return thres_sol
    w = soft_thres_l2_normalized(a_arr, thres_sol)
    return raise_arr_to_pd_obj(w, a)

# reviewed
def compute_BCSS(X: DF_ARR_TYPE, 
                 proba_: DF_ARR_TYPE, 
                 centers_: Optional[np.ndarray] = None,
                 tol: float = 1e-6) -> SER_ARR_TYPE:
    """
    Compute the Between Cluster Sum of Squares (BCSS).

    The BCSS is computed based on the cluster centers and probabilities. If no centers are provided, 
    they will be computed from probabilities. Any BCSS values below the tolerance are set to zero.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    proba_ : DataFrame or ndarray
        The cluster assignment probabilities.

    centers_ : ndarray, optional
        The cluster centers. NA values are acceptable.
        If not provided, they are estimated from the data.

    tol : float, optional (default=1e-6)
        The tolerance for setting BCSS values to zero.

    Returns
    -------
    Series or ndarray
        The BCSS values for each feature.
    """
    X_arr, proba_arr = check_2d_array(X), check_2d_array(proba_)
    if centers_ is None: centers_ = weighted_mean_cluster(X_arr, proba_arr)
    # replace NAs in centers with 0. won't affect computation
    centers_ = np.nan_to_num(centers_, nan=0.)
    # assert not np.isnan(centers_).any()
    Ns = proba_arr.sum(axis=0)
    BCSS = Ns @ ((centers_ - X_arr.mean(axis=0))**2)
    BCSS = set_zero_arr(BCSS, tol=tol)
    assert not np.isnan(BCSS).any()
    return raise_arr_to_pd_obj(BCSS, X, index_key="columns")

############################
## SJM
############################

class SparseJumpModel(BaseEstimator):
    """
    Sparse Jump Model (SJM) with feature selection.

    This model extends the standard jump model by incorporating a Lasso-like feature 
    selection process, where the number of selected features is controlled by `max_feats`.

    Parameters
    ----------
    n_components : int, default=2
        Number of components (clusters).

    max_feats : float, default=100.
        Controls the number of features included. This is the square of `kappa`, and 
        represents the effective number of features.

    jump_penalty : float, default=0.
        The jump penalty. In SJM, this penalty is scaled by 
        `1 / sqrt(n_features)` since features are weighted.

    cont : bool, default=False
        If `True`, the continuous jump model is used. Otherwise, the discrete model is applied.

    grid_size : float, default=0.05
        The grid size for discretizing the probability simplex (only used for continuous models).

    mode_loss : bool, default=True
        Whether to apply the mode loss penalty (only relevant for continuous models).

    random_state : int or RandomState, optional
        Random number generator seed for reproducibility.

    max_iter : int, default=30
        Maximum number of iterations for the coordinate descent algorithm in feature selection.

    tol_w : float, default=1e-4
        Tolerance for stopping the optimization of feature weights.

    max_iter_jm : int, default=1000
        Maximum number of iterations for the jump model fitting process.

    tol_jm : float, default=1e-8
        Stopping tolerance for the jump model fitting.

    n_init_jm : int, default=10
        Number of initializations for the jump model.

    verbose : int, default=0
        Controls the verbosity of the output.

    Attributes
    ----------
    jm_ins : JumpModel
        The fitted jump model instance, with feature weighting.

    feat_weights : ndarray
        The optimal feature weights.
        Square root of the `w` vector in the oroginal SJM formulation.

    labels_ : Series or ndarray
        In-sample optimal state assignments.

    proba_ : DataFrame or ndarray
        In-sample optimal probability matrix.

    ret_, vol_ : Series or ndarray
        Average return (`ret_`) and volatility (`vol_`) for each state, if `ret_ser` is provided.

    centers_ : ndarray
        The weighted cluster centers.
    """
    # reviewed
    def __init__(self,
                 n_components: int = 2, 
                 max_feats: float = 100.,
                 jump_penalty: float = 0., 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = RANDOM_STATE, 
                 max_iter: int = 30, 
                 tol_w: float = 1e-4, 
                 max_iter_jm: int = 1000,
                 tol_jm: float = 1e-8,
                 n_init_jm: int = 10,
                 verbose: int = 0):
        self.n_components = int(n_components)
        self.max_feats = max_feats
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol_w = tol_w
        self.max_iter_jm = max_iter_jm
        self.tol_jm = tol_jm
        self.n_init_jm = n_init_jm
        self.verbose = verbose

    # reviewed
    def init_jm(self):
        """
        Initialize the jump model instance with scaled jump penalty.
        """
        jump_penalty = self.jump_penalty / np.sqrt(self.n_features_all)
        jm = JumpModel(n_components=self.n_components,
                       jump_penalty=jump_penalty,
                       cont=self.cont,
                       grid_size=self.grid_size,
                       mode_loss=self.mode_loss,
                       random_state=self.random_state,
                       max_iter=self.max_iter_jm,
                       tol=self.tol_jm,
                       n_init=self.n_init_jm,
                       verbose=decre_verbose(self.verbose))
        self.jm_ins = jm
        return jm
    
    # reviewed
    def print_log(self, n_iter, BCSS, w):
        """
        Print fitting logs if verbosity is enabled.
        """
        if self.verbose:
            print("Iter:", n_iter)
            print("BCSS:\n", BCSS)     #, "sum:", BCSS.sum()
            print("w:\n", w, "\n")
        return 

    # reviewed
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: Optional[SER_ARR_TYPE] = None,
            sort_by: Optional[str] = "cumret"):
        """
        Fit the sparse jump model using coordinate descent.

        This method iteratively optimizes the feature weights and fits the jump model 
        on the weighted data.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        ret_ser : Series or ndarray, optional
            A return series used for sorting states.

        sort_by : ["cumret", "vol", "freq", "ret"], optional (default="cumret")
            Criterion for sorting states.

        Returns
        -------
        SparseJumpModel
            The fitted sparse jump model.
        """
        #
        X_arr = check_2d_array(X)
        self.n_features_all = X_arr.shape[1]
        # jm ins
        jm = self.init_jm()
        # get attrs
        max_iter = self.max_iter
        tol_w = self.tol_w
        norm_ub = np.sqrt(self.max_feats)
        # 
        w_old = np.ones(self.n_features_all)*2  # not a valid weight, only used for entering the 1st iter
        w = np.ones(self.n_features_all) / np.sqrt(self.n_features_all)  # initial weight   #  np.repeat(1/np.sqrt(self.n_features_all), self.n_features_all)  
        n_iter = 0
        while (n_iter < max_iter and norm(w-w_old, 1) / norm(w_old, 1) > tol_w):
            # 
            n_iter += 1
            w_old = w
            # Step 1: fix w, fit JM
            feat_weights = np.sqrt(w)
            # use the previous optimal center, weighted by the most recent w, as an initialization
            if n_iter > 1: jm.centers_ = centers_unweighted * feat_weights    
            # fit JM on weighted data
            jm.fit(X, ret_ser=ret_ser, feat_weights=feat_weights, sort_by=sort_by)
            # Step 2: optimize w
            # update (unweighted) centers
            centers_unweighted = weighted_mean_cluster(X_arr, jm.proba_)
            # compute BCSS on the original data
            BCSS = compute_BCSS(X_arr, jm.proba_, centers_unweighted)
            if (BCSS <= 0).all(): # all in one cluster
                self.print_log(n_iter, BCSS, w)
                break
            w = solve_lasso(BCSS/BCSS.max(), norm_ub)
            self.print_log(n_iter, BCSS, w)
        # best res
        self.w = raise_arr_to_pd_obj(w, X, index_key="columns")
        self.feat_weights = raise_arr_to_pd_obj(jm.feat_weights, X, index_key="columns")
        self.centers_ = jm.centers_ # weighted centers
        # self.centers_ = weighted_mean_cluster(X_arr, jm.proba_, )
        self.labels_ = jm.labels_
        self.proba_ = jm.proba_
        if ret_ser is not None:
            self.ret_ = jm.ret_
            self.vol_ = jm.vol_
        return self
    
    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict state probabilities in an online fashion.
        """
        return self.jm_ins.predict_proba_online(X)
    
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict states in an online fashion.
        """
        return self.jm_ins.predict_online(X)
    
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict state probabilities using all available data.
        """
        return self.jm_ins.predict_proba(X)

    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict states using all available data.
        """
        return self.jm_ins.predict(X)

# jump.py
"""
Module for statistical jump models (JMs) and continuous jump models (CJMs).

This module provides utilities and helper functions for implementing and working 
with jump models and their continuous variants.

Depends on
----------
utils/ : Modules
    Utility functions for validation and clustering operations.
base : Module
    Base class for clustering-like algorithms.
"""

from itertools import product
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from . import RANDOM_STATE
from .utils import *
from .base import *

#################################
## model helpers
#################################

# reviewed
def jump_penalty_to_mx(jump_penalty: float, n_c: int) -> np.ndarray:
    """
    Convert a scalar jump penalty into a penalty matrix.

    Parameters
    ----------
    jump_penalty : float
        The scalar value representing the jump penalty.

    n_c : int
        The number of clusters or components.

    Returns
    -------
    np.ndarray
        A matrix of shape (n_c, n_c) where off-diagonal elements are the penalty values 
        and diagonal elements are zero.
    """
    # assert is_numbers(jump_penalty)
    return jump_penalty * (np.ones((n_c, n_c)) - np.eye(n_c))   # default dtype is float

# reviewed
def discretize_prob_simplex(n_c: int, grid_size: float) -> np.ndarray:
    """
    Sample grid points on a probability simplex. This function generates all possible 
    combinations of probabilities that sum to 1, given the grid size.
    NB: this operation is of combinatorial complexity.

    Parameters
    ----------
    n_c : int
        The number of components or clusters.

    grid_size : float
        The step size for discretization of the simplex.

    Returns
    -------
    np.ndarray
        An array of shape (n_candidates, n_c), where each row represents a point on the 
        simplex. The number of candidates depends on the grid size.
    """
    N = int(1/grid_size)
    tuples = filter(lambda x: sum(x)==N, product(range(N+1), repeat = n_c))
    lst = np.array(list(tuples)[::-1], dtype=float)/N   # (n_candidates, n_c)
    return lst

#################################
## DP algo & E step
#################################

# reviewed
def dp(loss_mx: np.ndarray, 
       penalty_mx: np.ndarray, 
       return_value_mx: bool = False) -> Union[tuple[np.ndarray, float], np.ndarray]:
    r"""
    Solve the optimization problem involved in the E-step calculation (state assignment), 
    using a dynamic programming (DP) algorithm.

    The objective is to minimize:

    $$\min \sum_{t=0}^{T-1} L(t, s_t) + \sum_{t=1}^{T-1} \Lambda(s_{t-1}, s_t).$$

    If some columns of `loss_mx` contain `NaN` values, they are replaced with `inf`, 
    making those clusters unreachable.

    Note: The DP algorithm cannot be easily sped up using Numba due to issues with 
    `.min(axis=0)` in Numba.

    Parameters
    ----------
    loss_mx : ndarray of shape (n_s, n_c)
        The loss matrix, where `L(t, k)` represents the loss for time `t` and state `k`.

    penalty_mx : ndarray of shape (n_c, n_c)
        The jump penalty matrix between states.

    return_value_mx : bool, optional (default=False)
        If `True`, compute and return the value matrix from the DP algorithm. The value at 
        each time step `t` is based on all information up to that point, making it suitable 
        for online inference.

    Returns
    -------
    tuple[np.ndarray, float] or np.ndarray
        If `return_value_mx` is `False`, returns a tuple containing:
        - The optimal state assignments.
        - The optimal loss function value.
        
        If `return_value_mx` is `True`, returns the value matrix.
    """
    # valid shape
    n_s, n_c = loss_mx.shape
    assert penalty_mx.shape == (n_c, n_c)
    # replace nan by inf
    loss_mx = replace_nan_by_inf(loss_mx)
    # DP algo
    values, assign = np.empty((n_s, n_c)), np.empty(n_s, dtype=int)
    # initial
    values[0] = loss_mx[0]
    # DP iteration
    for t in range(1, n_s):
        values[t] = loss_mx[t] + (values[t-1][:, np.newaxis] + penalty_mx).min(axis=0) # values[t-1][:, np.newaxis] turns the (t-1)-th row into a column
    # 
    if return_value_mx:
        return values
    # find optimal path backwards
    assign[-1] = values[-1].argmin()
    value_opt = values[-1, assign[-1]]
    # traceback
    for t in range(n_s - 1, 0, -1):
        assign[t-1] = (values[t-1] + penalty_mx[:, assign[t]]).argmin()
    return assign, value_opt

# reviewed
def raise_JM_labels_to_proba(labels_: np.ndarray, n_c: int, prob_vecs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert JM labels into a probability matrix. If `prob_vecs` is provided, 
    the probability matrix is constructed using the probability vectors corresponding to each label. 
    Otherwise, a hard-clustering probability matrix is created from the labels.
    """
    return prob_vecs[labels_] if prob_vecs is not None else raise_labels_into_proba(labels_, n_c)

# reviewed
def raise_JM_proba_to_df(proba_: np.ndarray, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
    """
    Convert a probability matrix into a pandas DataFrame, aligning with the index of the input 
    data matrix `X`.
    """
    return raise_arr_to_pd_obj(proba_, X, columns_key=None, return_as_ser=False)

LARGE_FLOAT = 1e100

# reviewed
def do_E_step(X: np.ndarray, 
              centers_: np.ndarray, 
              penalty_mx: np.ndarray, 
              prob_vecs: Optional[np.ndarray] = None, 
              return_value_mx: bool = False) -> Union[tuple[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Perform a single E-step: compute the loss matrix and calling the solver.

    This function handles both hard clustering and continuous models. The `centers_` parameter 
    can contain `NaN` values. It returns the probabilities, labels, and optimal value, where 
    `labels_` correspond to the state space.

    Parameters
    ----------
    X : ndarray of shape (n_s, n_f)
        The input data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    centers_ : ndarray of shape (n_c, n_f)
        The cluster centers. Can contain `NaN` values.

    penalty_mx : ndarray of shape (n_c, n_c)
        The penalty matrix representing the transition cost between states.

    prob_vecs : ndarray of shape (N, n_c), optional
        Probability vectors for the continuous model. If provided, this adjusts the loss matrix.

    return_value_mx : bool, optional (default=False)
        If `True`, return the value matrix from the DP algorithm, which can be used for online inference.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float] or np.ndarray
        If `return_value_mx` is `False`, returns a tuple containing:
        - `proba_` : ndarray of shape (n_s, n_c)
            The probability matrix, where each row corresponds to the probabilities for a sample.
        - `labels_` : ndarray of shape (n_s,)
            The state labels assigned to each sample.
        - `val_` : float
            The optimal value of the objective function.

        If `return_value_mx` is `True`, returns the value matrix instead of the tuple.
    """
    n_c = len(centers_)     # (n_c, n_f)
    # compute loss matrix
    loss_mx = .5 * cdist(X, centers_, "sqeuclidean")    # (n_s, n_c)
                                                        # contain `nan` if `centers_` contains `nan`.
    if prob_vecs is not None:    # cont model, (N, n_c)
        # replace the nan in loss_mx by a very large floating number
        loss_mx = np.nan_to_num(loss_mx, nan=LARGE_FLOAT, posinf=LARGE_FLOAT, neginf=LARGE_FLOAT)
        loss_mx = loss_mx @ prob_vecs.T     # each pair of loss between period t and candidate vector, (n_s, N)
    if return_value_mx: return dp(loss_mx, penalty_mx, return_value_mx=True)
    # do a full E step
    labels_, val_ = dp(loss_mx, penalty_mx, return_value_mx=False)     # output labels_ is of type int
    proba_ = raise_JM_labels_to_proba(labels_, n_c, prob_vecs)
    return proba_, labels_, val_    # the returned proba_ must be a valid proba arr

#################################
## feature weights
#################################

# reviewed
def valid_feat_weights(feat_weights: Optional[SER_ARR_TYPE]) -> None:
    """
    Validate the input `feat_weights`, ensuring all weights are non-negative and at least 
    one is positive. This function is called at the beginning of the method to ensure 
    the feature weights are valid.

    Parameters
    ----------
    feat_weights : Series or ndarray, optional
        The array of feature weights to validate. If `None`, no validation is performed.

    Raises
    ------
    AssertionError
        If any feature weights are negative or if no positive weights exist.
    """
    if feat_weights is None: return 
    feat_weights_arr = check_1d_array(feat_weights)
    assert (feat_weights_arr >= 0.).all(), "Feature weights must be non-negative."
    assert (feat_weights_arr > 0.).any(), "At least one feature weight must be positive."
    return 

# reviewed
def _valid_shape_X_feat_weights(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> None:
    """
    Assert that the dimensions of the input data matrix `X` and feature weights match.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no assertion is made.

    Raises
    ------
    AssertionError
        If the dimensions of `X` and `feat_weights` do not match.
    """
    if feat_weights is None: 
        return
    if is_ser_df(X) and is_ser_df(feat_weights): 
        assert (X.columns==feat_weights.index).all(), "Feature mismatch: column names do not match feature weight index."
    else: 
        assert X.shape[1]==len(feat_weights) , "Feature mismatch: number of features does not match feature weights."
    return 

# reviewed
def _weight_X(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> np.ndarray:
    """
    Apply feature weights to the input data matrix `X`. If `feat_weights` is `None`, no 
    weights are applied. It is assumed that dimensions match.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no weighting is applied.

    Returns
    -------
    np.ndarray
        The weighted data matrix, with the same shape as `X`.
    """
    X_arr = check_2d_array(X)
    if feat_weights is None: return X_arr
    # Apply feature weights
    feat_weights_arr = check_1d_array(feat_weights)
    return X_arr * feat_weights_arr
        
# reviewed
def check_X_with_feat_weights(X: DF_ARR_TYPE, feat_weights: Optional[SER_ARR_TYPE]) -> np.ndarray:
    """
    Process the input data matrix `X` and feature weights, returning a weighted version of `X`.

    Parameters
    ----------
    X : DataFrame or ndarray
        The input data matrix.

    feat_weights : Series or ndarray, optional
        The array of feature weights. If `None`, no weighting is applied.

    Returns
    -------
    np.ndarray
        The weighted data matrix.
    """
    # Validate that the dimensions of X and feat_weights match
    _valid_shape_X_feat_weights(X, feat_weights)
    # Apply feature weights to X
    return _weight_X(X, feat_weights)

#################################
## model code
#################################

class JumpModel(BaseClusteringAlgo):
    """
    Statistical jump model estimation, supporting both discrete and continuous models.

    This class provides methods for fitting and predicting with jump models, using coordinate 
    descent for optimization. Both discrete and continuous models are supported, with optional 
    feature weighting and state sorting.

    Parameters
    ----------
    n_components : int, default=2
        The number of components (states) in the model.

    jump_penalty : float, default=0.
        Penalty term (`lambda`) applied to state transitions in both discrete and continuous models.

    cont : bool, default=False
        If `True`, the continuous jump model is used. Otherwise, the discrete model is applied.

    grid_size : float, default=0.05
        The grid size for discretizing the probability simplex. Only relevant for the continuous model.

    mode_loss : bool, default=True
        Whether to apply the mode loss penalty. Only relevant for the continuous model.

    random_state : int or RandomState, optional (default=None)
        Random number seed for reproducibility.

    max_iter : int, default=1000
        Maximum number of iterations for the coordinate descent algorithm during model fitting.

    tol : float, default=1e-8
        Stopping tolerance for the improvement in objective value during optimization.

    n_init : int, default=10
        Number of initializations for the model fitting process.

    verbose : int, default=0
        Controls the verbosity of the output. Higher values indicate more verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_c, n_f)
        The cluster centroids estimated during model fitting.

    labels_ : Series or ndarray
        In-sample fitted optimal label sequence.

    proba_ : DataFrame or ndarray
        In-sample fitted optimal probability matrix.

    ret_, vol_ : Series or ndarray
        The average return (`ret_`) and volatility (`vol_`) for each state. These attributes 
        are available only if `ret_ser` is provided to the `.fit()` method.

    transmat_ : ndarray of shape (n_c, n_c)
        The estimated transition probability matrix between states.

    val_ : float
        The optimal value of the loss function.
    """
    # reviewed
    def __init__(self,
                 n_components: int = 2, 
                 jump_penalty: float = 0., 
                 cont: bool = False, 
                 grid_size: float = 0.05, 
                 mode_loss: bool = True, 
                 random_state = RANDOM_STATE, 
                 max_iter: int = 1000, 
                 tol: float = 1e-8, 
                 n_init: int = 10, 
                 verbose: int = 0):
        super().__init__(int(n_components), n_init, max_iter, tol, random_state, verbose)
        self.jump_penalty = jump_penalty
        self.cont = cont
        self.grid_size = grid_size
        self.mode_loss = mode_loss
        self.alpha = 2  # the power raised to the jump penalty in CJM

    # reviewed           
    def check_jump_penalty_mx(self) -> np.ndarray:
        """
        Initialize the jump penalty matrix for state transitions.

        - For the discrete model, the state space is {0, 1, ..., n_c - 1}, and the scalar 
          `jump_penalty` is converted into a matrix.
        - For the continuous model, `jump_penalty` is multiplied by the pairwise L1 distance 
          between probability vectors. Optionally applies a mode loss penalty.

        Returns
        -------
        np.ndarray
            The jump penalty matrix to be used in the model.
        """
        assert is_numbers(self.jump_penalty)
        if not self.cont:
            self.prob_vecs = None      # useful in the E step to tell whether the model is continuous/discrete.
            jump_penalty_mx = jump_penalty_to_mx(self.jump_penalty, self.n_components) 
        else:    # continuous model
            self.prob_vecs = discretize_prob_simplex(self.n_components, self.grid_size)   # state space. useful for computing L mx in E step
            pairwise_l1_dist = cdist(self.prob_vecs, self.prob_vecs, 'cityblock')/2
            jump_penalty_mx = self.jump_penalty * (pairwise_l1_dist ** self.alpha)
            if self.mode_loss:      # adding mode loss ensures that the penalty mx has correspondence with a TPM. i.e. sum(exp(- )) of every row leads to the same value.
                mode_loss = logsumexp(-jump_penalty_mx, axis=1, keepdims=True)
                mode_loss -= mode_loss[0]     # offset a constant
                jump_penalty_mx += mode_loss
        self.jump_penalty_mx = jump_penalty_mx      # to be used in `.predict()`  & `.predict_proba()`
        return jump_penalty_mx
    
    # reviewed
    def check_X_predict_func(self, X: DF_ARR_TYPE) -> np.ndarray:
        """
        Validate the input data `X` for all prediction methods (but not for fitting), 
        and apply feature weighting if applicable. Assumes that the model has already 
        been fitted.

        This method overrides the superclass method.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        np.ndarray
            The weighted input data matrix, if feature weights are provided.
        """
        self.is_shape_match_X_centers(X)
        feat_weights = getattr_(self, "feat_weights")
        return check_X_with_feat_weights(X, feat_weights)
    
    # reviewed
    def fit(self, 
            X: DF_ARR_TYPE, 
            ret_ser: Optional[SER_ARR_TYPE] = None, 
            feat_weights: Optional[SER_ARR_TYPE] = None,
            sort_by: Optional[str] = "cumret"):
        """
        Fit the jump model using the coordinate descent algorithm.

        The states are sorted by the specified criterion: ["cumret", "vol", "freq", "ret"].
        The Viterbi algorithm is optionally used for state assignment. This choice does 
        not impact the final numerical results but may affect computational speed.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        ret_ser : Series or ndarray, optional
            A return series used for sorting states and calculating state-specific returns 
            and volatilities.

        feat_weights : Series or ndarray, optional
            Feature weights to apply to the input data matrix.

        sort_by : ["cumret", "vol", "freq", "ret"], optional (default="cumret")
            Criterion for sorting the states.
        """
        # valid feat weights
        valid_feat_weights(feat_weights)
        # check X
        X_arr = check_X_with_feat_weights(X, feat_weights)
        # save valid feat weights
        self.feat_weights = feat_weights
        # get attributes
        n_c = self.n_components
        max_iter = self.max_iter
        tol = self.tol
        verbose = self.verbose
        # make sure the state space, and compute the penalty matrix used for the E step
        jump_penalty_mx = self.check_jump_penalty_mx()
        # init centers
        init_centers_values = self.init_centers(X_arr)
        # the best results over all initializations, compare to it in the last part of each iteration
        best_val = np.inf
        best_res = {}   # store: "centers_", "proba_", "labels_".
        best_res['labels_'] = None # "labels_" is not always 0/1, but the labels of the state space (candidate prob vecs)
                                   #  it is only used to compare whether two inits lead to the same estimation. the final `labels_` is based on `proba_.argmax(axis=1)`.
        # iter over all the initializations
        for n_init_, centers_ in enumerate(init_centers_values):
            # initialize the labels and value in the previous iteration.
            labels_pre, val_pre = None, np.inf
            # do one E step
            proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)
            num_iter = 0
            # iterate between M and E steps
            while (num_iter < max_iter and (not is_same_clustering(labels_, labels_pre)) and val_pre - val_ > tol):
                # update
                num_iter += 1
                labels_pre, val_pre = labels_, val_
                # M step: update centers
                centers_ = weighted_mean_cluster(X_arr, proba_) 
                # E step
                proba_, labels_, val_ = do_E_step(X_arr, centers_, jump_penalty_mx, prob_vecs=self.prob_vecs)
            if verbose: print(f"{n_init_}-th init. val: {val_}")
            # compare with previous initializations
            if (not is_same_clustering(best_res['labels_'], labels_)) and val_ < best_val:
                best_idx = n_init_
                best_val = val_
                # save model attributes
                best_res['centers_'] = centers_
                best_res['labels_'] = labels_   # only used to compare with later iters, won't permutate
                best_res['proba_'] = proba_
        self.val_ = best_val
        if verbose: print(f"{best_idx}-th init has the best value: {best_val}.")
        # sort states
        sort_states_from_ret(ret_ser, X, best_res, sort_by=sort_by)
        # save attributes
        if ret_ser is not None:
            self.ret_ = best_res["ret_"]
            self.vol_ = best_res["vol_"]
        self.centers_ = best_res['centers_']        # weighted centers
        self.proba_ = raise_JM_proba_to_df(best_res['proba_'], X)
        self.labels_ = reduce_proba_to_labels(self.proba_)
        self.transmat_ = empirical_trans_mx(self.labels_, n_components=n_c)
        return self
        
    # reviewed
    def predict_proba_online(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict the probability of each state in an online fashion, where the prediction 
        for the i-th row is based only on data prior to that row.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        DataFrame or ndarray
            The predicted probabilities for each state.
        """
        X_arr = self.check_X_predict_func(X)
        value_mx = do_E_step(X_arr, self.centers_, self.jump_penalty_mx, self.prob_vecs, return_value_mx=True)
        labels_ = value_mx.argmin(axis=1)
        proba_ = raise_JM_labels_to_proba(labels_, self.n_components, self.prob_vecs)
        return raise_JM_proba_to_df(proba_, X)
    
    # reviewed
    def predict_online(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict the state in an online fashion, where the prediction for the i-th row 
        is based only on data prior to that row.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        Series or ndarray
            The predicted state labels for each sample.
        """
        return reduce_proba_to_labels(self.predict_proba_online(X))
    
    # reviewed
    def predict_proba(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Predict the probability of each state, using all available data in `X`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        use_viterbi : bool, optional (default=True)
            Whether to use the Viterbi solver.

        Returns
        -------
        DataFrame or ndarray
            The predicted probabilities for each state.
        """
        X_arr = self.check_X_predict_func(X)
        proba_, _, _ = do_E_step(X_arr, self.centers_, self.jump_penalty_mx, self.prob_vecs)
        return raise_JM_proba_to_df(proba_, X)

    # reviewed
    def predict(self, X: DF_ARR_TYPE) -> SER_ARR_TYPE:
        """
        Predict the state for each sample, using all available data in `X`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        use_viterbi : bool, optional (default=True)
            Whether to use the Viterbi solver.

        Returns
        -------
        Series or ndarray
            The predicted state labels for each sample.
        """
        return reduce_proba_to_labels(self.predict_proba(X))


# __init__.py
# global constants
RANDOM_STATE = 0

# base.py
"""
Module for the base class used in clustering-like algorithms.

This module provides helpers for parameter sorting, parameter initialization, and base class 
definitions for clustering-like algorithms.

Depends on
----------
utils/ : Modules
"""

from .utils import *

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.cluster import kmeans_plusplus

##################################
# Sorting
##################################

# reviewed
def sort_param_dict_from_idx(params: dict, idx: np.ndarray) -> None:
    """
    Sort a dictionary of parameters according to a given index array.

    Expected parameter shapes:
    - `ret_` : (n_c,)
    - `vol_` : (n_c,)
    - `means_` : (n_c, n_f)
    - `centers_` : (n_c, n_f)
    - `transmat_` : (n_c, n_c)
    - `startprob_` : (n_c,)
    - `proba_` : (n_s, n_c)
    - `covars_` : (n_c, 1)

    Parameters
    ----------
    params : dict
        A dictionary of parameters, each corresponding to a clustering result.

    idx : ndarray of shape (n_c,)
        The index array to sort the parameters by.
    """
    # permute `axis=0`
    for key in ['ret_', 'vol_', 'means_', 'centers_', 'startprob_', 'covars_']:
        if key in params: params[key] = params[key][idx]
    # transmat, need to permute both `axis=0 & 1`
    if 'transmat_' in params: params['transmat_'] = params['transmat_'][idx][:, idx]
    # proba, need to permute `axis=1`
    if 'proba_' in params: params['proba_'] = params['proba_'][:, idx]
    return 
 
# reviewed
def sort_param_dict(params: dict, sort_by='ret') -> None:
    """
    Sort the states by a given criterion and permute all parameters accordingly.
    Supported sorting criteria are ["cumret", "vol", "freq", "ret"], i.e.
    states sorted by decreasing (cumulative) return, increasing vol, decreasing frequency.

    `nan` values will be (ideally) sorted to the end.

    Parameters
    ----------
    params : dict
        A dictionary of parameters, each corresponding to a clustering result.

    sort_by : str, optional (default='ret')
        The criterion to sort the parameters by. Must be one of ["cumret", "vol", "freq", "ret"].
    """
    if sort_by is None: return
    assert sort_by in ["cumret", "vol", "freq", "ret"]
    if "proba_" in params: freq = params["proba_"].sum(axis=0)
    if sort_by == 'vol':
        assert 'vol_' in params
        criterion = params['vol_']
    elif sort_by == "cumret":
        assert "ret_" in params and "proba_" in params
        criterion = -params["ret_"] * freq   # missing regimes will have a cumret of nan*0 = nan
    elif sort_by == "ret":
        assert "ret_" in params
        criterion = -params['ret_']
    elif sort_by == "freq":
        assert "proba_" in params
        criterion = -freq # decreasing freq
    else:
        raise NotImplementedError()
    criterion = replace_inf_by_nan(criterion)
    idx = np.argsort(criterion)  
    sort_param_dict_from_idx(params, idx)
    return 

# reviewed
def align_and_check_ret_ser(ret_ser: SER_ARR_TYPE, X: DF_ARR_TYPE) -> np.ndarray:
    """
    Align a return series with the input data matrix `X`,
    and convert it to a 1D array.

    Parameters
    ----------
    ret_ser : Series or ndarray
        The return series to validate.

    X : DataFrame or ndarray
        The data matrix to align with.

    Returns
    -------
    ndarray
        The aligned and validated 1D return array.
    """
    ret_ser = align_x_with_y(ret_ser, X)
    return check_1d_array(ret_ser)

# reviewed
def sort_states_from_ret(ret_ser: Optional[SER_ARR_TYPE], 
                         X: DF_ARR_TYPE,
                         best_res: dict, 
                         sort_by: str = "cumret") -> None:
    """
    Sort the states in the fitted parameters stored in a dictionary according to a specified criterion.
    This is intended for financial applications. If not applicable, input `None` for `ret_ser`.

    Parameters
    ----------
    ret_ser : Series or ndarray, optional
        The return series to use for computing average return and volatility within each state.
        If `None`, sorting is attempted by decreasing frequency (given that the `proba_` param is estimated).

    X : DataFrame or ndarray
        The data matrix to use for alignment.

    best_res : dict
        Fitted parameters of the best clustering results to sort.

    sort_by : str, optional (default="cumret")
        The criterion to use for sorting. Must be one of ["cumret", "vol", "freq", "ret"].

        - If `ret_ser` is provided, it is used to compute the mean return (`ret_`) and volatility (`vol_`) 
        within each state. Sorting by decreasing (cumulative) return and increasing volatility is possible. 
        - If `ret_ser` is `None`, sort by frequency if the `proba_` attribute exists, otherwise 
        don't sort anything.
    """
    if ret_ser is not None: 
        # valid inputs
        ret_ser_arr = align_and_check_ret_ser(ret_ser, X)
        # compute mean & vol for each cluster
        best_res['ret_'], best_res['vol_'] = weighted_mean_std_cluster(ret_ser_arr, best_res['proba_'])
        # the best parameters sorted by a criterion
        sort_param_dict(best_res, sort_by=sort_by)
    elif "proba_" in best_res:
        sort_param_dict(best_res, sort_by="freq")
    return 

##################################
# Initialization
##################################

# reviewed
def init_centers_kmeans_plusplus(X: np.ndarray, n_c=2, n_init=10, random_state=None) -> list[np.ndarray]:
    """
    Initialize the cluster centers using the K-Means++ algorithm, repeated `n_init` times.

    Parameters
    ----------
    X : ndarray of shape (n_s, n_f)
        The data matrix.

    n_c : int, optional (default=2)
        The number of clusters.

    n_init : int, optional (default=10)
        The number of initializations to perform.

    random_state : int, RandomState instance, or None, optional (default=None)
        Controls the randomness of the center initialization.

    Returns
    -------
    centers : list of ndarray
        A list of initialized centers for each run.
    """
    random_state = check_random_state(random_state)
    centers = [kmeans_plusplus(X, n_c, random_state=random_state)[0] for _ in range(n_init)]
    return centers   # (n_init, n_c, n_f)

##################################
# Base Class
##################################

class BaseClusteringAlgo(BaseEstimator):
    """
    A base class for all clustering-like algorithms.

    This class provides several common methods but does not include any model fitting logic. 
    It is intended to be inherited with specific implementations.

    Parameters
    ----------
    n_components : int
        The number of components (clusters).

    n_init : int
        The number of initializations to perform.

    max_iter : int
        The maximum number of iterations.

    tol : float
        The tolerance for convergence.

    random_state : int, RandomState instance, or None
        Controls the randomness.

    verbose : int
        Controls the verbosity of the output.
    """
    # reviewed
    def __init__(self,
                 n_components,
                 n_init,
                 max_iter,
                 tol,
                 random_state,
                 verbose
                 ) -> None:
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    # reviewed
    def is_shape_match_X_centers(self, X: DF_ARR_TYPE) -> bool:
        """
        Check whether the shape of `X` and `centers_` matches. Useful for `predict` methods.
        `self` must already has the attribute `centers_`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        bool
            True if the shapes match, False otherwise.
        """
        n_f = X.shape[1]
        return self.centers_.shape == (self.n_components, n_f)
    
    # reviewed
    def init_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Initialize the centers using k-Means++ for multiple initializations. 
        If attribute `centers_` exists and matches the shape of `X`, it will also 
        be included as an initial value.

        Parameters
        ----------
        X : ndarray of shape (n_s, n_f)
            The input data matrix.

        Returns
        -------
        centers : ndarray
            The initialized centers for each run.
        """
        centers = init_centers_kmeans_plusplus(X, self.n_components, self.n_init, self.random_state)
        if hasattr(self, "centers_") and self.is_shape_match_X_centers(X): 
            centers.append(self.centers_)  # use previously fitted value as one initial center value
        return np.array(centers)
    
    # reviewed
    def check_X_predict_func(self, X: DF_ARR_TYPE) -> np.ndarray:
        """
        Check the input data matrix for `.predict` methods, ensuring it is a 2D array and 
        matches the shape of `centers_`.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data matrix.

        Returns
        -------
        ndarray
            The validated 2D data array.
        """
        X_arr = check_2d_array(X)
        assert self.is_shape_match_X_centers(X_arr)
        return X_arr
 

# preprocess.py
"""
Module for data preprocessing.

This module contains classes for scaling and clipping data, with a focus on 
handling pandas DataFrame input/output.

Depends on
----------
utils/ : Modules
"""

from .utils import *

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

############################################
## Scaler
############################################

# reviewed
class StandardScalerPD(BaseEstimator):
    """
    Provides support for pandas DataFrame input/output with the `StandardScaler()` class.
    
    This class extends the functionality of the standard `StandardScaler` by ensuring that
    the input and output are handled as pandas DataFrames, preserving index and column labels.
    """
    def init_scaler(self):
        """
        Initialize and return the standard `StandardScaler` instance.
        """
        return StandardScaler()
    
    def fit_transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Fit the scaler to the DataFrame and transform it in one step.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be scaled.
            
        Returns
        -------
        DataFrame or ndarray
            The scaled DataFrame.
        """
        return self.fit(X).transform(X)
    
    def fit(self, X: DF_ARR_TYPE):
        """
        Fit the scaler to the input DataFrame.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be used for fitting.
        
        Returns
        -------
        self
        """
        self.scaler = self.init_scaler().fit(X)
        return self

    def transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Transform the input DataFrame using the fitted scaler.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input DataFrame to be transformed.
        
        Returns
        -------
        DataFrame or ndarray
            The transformed (scaled) DataFrame.
        """
        return raise_arr_to_pd_obj(self.scaler.transform(X), X, return_as_ser=False)

############################################
## Clipper
############################################

# reviewed
class BaseDataClipper(BaseEstimator):
    """
    Base class for data clippers. 

    This class implements the `.transform()` and `.fit_transform()` methods, but leaves the `.fit()` 
    method to be implemented in subclasses. It is designed to clip data values within a specified range.
    
    Should be inherited by other classes that define the clipping bounds.
    """
    def __init__(self) -> None:
        self.lb = None  # Lower bound, initialized as None. Must be a numpy array.
        self.ub = None  # Upper bound, initialized as None. Must be a numpy array.

    def fit(self, X: DF_ARR_TYPE):
        raise NotImplementedError()

    def fit_transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Fit the clipper and transform the input data in one step.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to be clipped.

        Returns
        -------
        DataFrame or ndarray
            The clipped data.
        """
        return self.fit(X).transform(X)
    
    def transform(self, X: DF_ARR_TYPE) -> DF_ARR_TYPE:
        """
        Clip the input data using the fitted lower (`lb`) and upper (`ub`) bounds.
        
        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to be clipped.

        Returns
        -------
        DataFrame or ndarray
            The clipped data.
        """
        if self.ub is None and self.lb is None: return X
        return np.clip(X, self.lb, self.ub)

# reviewed
class DataClipperStd(BaseDataClipper):
    """
    Data clipper based on feature standard deviation.

    This class performs winsorization of the data, clipping it within a specified multiple of the 
    feature's standard deviation. The clipping bounds are defined as:
    
    lower bound = mean - (mul * std)
    upper bound = mean + (mul * std)

    Parameters
    ----------
    mul : float, default=3.
        The multiple of the feature's standard deviation used for clipping.

    Attributes
    ----------
    lb : ndarray
        The lower bound for each feature, calculated as mean - (mul * std).
    
    ub : ndarray
        The upper bound for each feature, calculated as mean + (mul * std).
    """
    def __init__(self, mul: float = 3.) -> None:
        super().__init__()
        self.mul = mul

    def fit(self, X: DF_ARR_TYPE):
        """
        Fit the clipper to the data by calculating the clipping bounds based on 
        the mean and standard deviation of each feature.

        Parameters
        ----------
        X : DataFrame or ndarray
            The input data to fit the clipper.

        Returns
        -------
        DataClipperStd
            The fitted clipper instance.
        """
        mul = self.mul
        assert mul > 0, "The multiplier `mul` must be positive."

        mean, std = X.mean(axis=0), X.std(axis=0, ddof=0)
        if is_df(X):
            mean = mean.to_numpy()
            std = std.to_numpy()
        self.lb = mean - mul * std; assert isinstance(self.lb, np.ndarray)
        self.ub = mean + mul * std; assert isinstance(self.ub, np.ndarray)
        return self


# cluster.py
"""
Helpers for numerical calculations in clustering analysis.

This module provides functions to handle clustering-related tasks such as label validation, 
probability conversion, and transition matrix computation.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# reviewed
def is_valid_labels(labels_: SER_ARR_TYPE, n_c: int = 2) -> bool:
    """
    Check whether a label array/series is a valid label sequence. The values of `labels_` must 
    lie in the set {0, 1, ..., n_c-1}.

    Parameters
    ----------
    labels_ : ndarray or Series
        The array or series of labels to check.

    n_c : int, optional (default=2)
        The number of clusters. Labels must lie in {0, 1, ..., n_c-1}.

    Returns
    -------
    bool
        True if the labels are valid, False otherwise.
    """
    labels_arr = check_1d_array(labels_)   # check whether it is intrinsically 1-d
    return set(labels_arr).issubset(set(range(n_c)))

# reviewed
def is_valid_proba(proba_: DF_ARR_TYPE) -> bool:
    """
    Check whether a probability array/series is valid, meaning all values are non-negative 
    and all rows sum to 1.

    Parameters
    ----------
    proba_ : ndarray or DataFrame
        The probability matrix to check.

    Returns
    -------
    bool
        True if the probability matrix is valid, False otherwise.
    """
    proba_arr = check_2d_array(proba_)
    return (proba_arr>=0).all() and np.isclose(proba_arr.sum(axis=1), 1.).all()

# reviewed
def raise_labels_into_proba(labels_: np.ndarray, n_c: int) -> np.ndarray:
    """
    Convert a discrete label array into a probability matrix. The resulting matrix corresponds 
    to hard clustering, with 0./1. values.

    Parameters
    ----------
    labels_ : ndarray of shape (n_s,)
        The array of integer labels.

    n_c : int
        The number of clusters.

    Returns
    -------
    proba_ : ndarray of shape (n_s, n_c)
        The probability assignment array.
    """
    # labels_ must be ints, and smaller than n_c
    # don't verify inputs, for performance consideration
    n_s = len(labels_)
    proba_ = np.zeros((n_s, n_c)) 
    proba_[range(n_s), labels_] = 1.
    # assert is_valid_proba(proba_)
    return proba_

# reviewed
def reduce_proba_to_labels(proba_: DF_ARR_TYPE) -> SER_ARR_TYPE:
    """
    Convert a probability matrix into a label series by taking the argmax of each row.

    Parameters
    ----------
    proba_ : ndarray or DataFrame
        The probability matrix to convert.

    Returns
    -------
    labels_ : ndarray or Series
        The label series obtained by taking the argmax of each row.
    """
    if is_df(proba_): return proba_.idxmax(axis=1)
    # arr
    return proba_.argmax(axis=1)

# reviewed
def is_map_from_left_to_right(labels_left: Optional[SER_ARR_TYPE], labels_right: Optional[SER_ARR_TYPE]) -> bool:
    """
    Check whether the map from `labels_left` to `labels_right` is valid, meaning elements with the same label 
    in `labels_left` must have the same label in `labels_right`. If either label array is `None`, return `False`.

    Parameters
    ----------
    labels_left : ndarray or Series, optional
        The left-side label array.

    labels_right : ndarray or Series, optional
        The right-side label array.

    Returns
    -------
    bool
        True if the mapping is valid, False otherwise.
    """
    if labels_left is None or labels_right is None:
        return False
    assert len(labels_left) == len(labels_right)
    for label in np.unique(labels_left):
        if len(np.unique(labels_right[labels_left==label])) != 1:
            return False
    return True

# reviewed
def is_same_clustering(labels1: Optional[SER_ARR_TYPE], labels2: Optional[SER_ARR_TYPE]) -> bool:
    """
    Check whether two clustering results are the same, under permutation. If either input is `None`, return `False`.

    Parameters
    ----------
    labels1 : ndarray or Series, optional
        The first label array.

    labels2 : ndarray or Series, optional
        The second label array.

    Returns
    -------
    bool
        True if the two clustering results are the same, False otherwise.
    """
    return is_map_from_left_to_right(labels1, labels2) and is_map_from_left_to_right(labels2, labels1)

# reviewed
def empirical_trans_mx(labels_: SER_ARR_TYPE, n_components=2, return_counts=False) -> np.ndarray:
    """
    Compute the empirical transition count or probability matrix from a label array/series. 
    Probability values will be `nan` if no transition from a state is observed.

    Parameters
    ----------
    labels_ : ndarray or Series
        The label array/series with values in {0, 1, ..., n_components - 1}, of both float/int dtype.

    n_components : int, optional (default=2)
        The number of unique labels.

    return_counts : bool, optional (default=False)
        If True, return the transition counts instead of probabilities.

    Returns
    -------
    ndarray
        The transition count or probability matrix.
    """
    assert is_valid_labels(labels_, n_c=n_components)
    labels_ = check_1d_array(labels_, dtype=int)    # labels must be int type, as it will be used as arr index.
    # count transitions
    count_mx = np.zeros((n_components, n_components), dtype=int)
    for i in range(n_components):
        # the next states after label==i
        labels_next = labels_[1:][labels_[:-1]==i]  # shift label by 1
        # count next states
        states, counts = np.unique(labels_next, return_counts=True)     # states must be ints.
        count_mx[i, states] = counts
    if return_counts: return count_mx
    # return probability
    return (1.*count_mx) / count_mx.sum(axis=1, keepdims=True)

# reviewed
def compute_num_shifts(labels_: SER_ARR_TYPE) -> int:
    """
    Count the number of regime shifts in a (int) label array/series.
    """
    labels_arr = check_1d_array(labels_)
    return (labels_arr[:-1]!=labels_arr[1:]).sum()


# calculation.py
"""
Helpers for basic numerical calculations.

This module focuses on numerical calculations with special attention to `numpy` behaviors 
involving NaN and infinity:

- 0. / 0. = np.nan
- 0. * np.inf = np.nan
- 0. * np.nan = np.nan
- 1. / 0. = np.inf
- -1. / 0. = -np.inf

Typically, it is rare for a statement to directly yield `np.inf`; the first two examples 
are the most common cases.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# will not raise warnings if: divide by zero, take sqrt of nega values
np.seterr(divide="ignore", invalid="ignore")

# reviewed
def set_zero_arr(x: np.ndarray, tol=1e-6) -> np.ndarray:
    """
    Set elements of a numpy array that are close to zero to exactly zero.

    Parameters
    ----------
    x : ndarray
        The input numpy array.

    tol : float, optional (default=1e-6)
        The tolerance value. Elements with absolute values smaller than `tol` 
        are set to zero.

    Returns
    -------
    ndarray
        A numpy array with near-zero values replaced by exact zeros.
    """
    return np.where(np.abs(x) < tol, 0., x)

# reviewed
def replace_inf_by_nan(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Replace both positive and negative infinity values with NaN in a float or numpy array.

    Parameters
    ----------
    x : float or ndarray
        The input float or numpy array.

    Returns
    -------
    float or ndarray
        A float or numpy array with infinities replaced by NaN.
    """
    return np.where(np.isinf(x), np.nan, x)

# reviewed
def replace_nan_by_inf(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Replace all NaN values with positive infinity in a float or numpy array.

    Parameters
    ----------
    x : float or ndarray
        The input float or numpy array.

    Returns
    -------
    float or ndarray
        A float or numpy array with NaN values replaced by infinity.
    """
    return np.where(np.isnan(x), np.inf, x)

# reviewed
def decre_verbose(verbose: int) -> int:
    """
    Decrement a non-negative integer by 1, ensuring the result is non-negative.

    Parameters
    ----------
    verbose : int
        A non-negative integer to decrement.

    Returns
    -------
    int
        The decremented value, ensuring it is non-negative.
    """
    return max(0, verbose-1)

#################################
## weighted ave
#################################

# reviewed
def weighted_mean_cluster(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the weighted sample average for each cluster. `X` can be a 1D or 2D array.
    If the total weights sum to zero (indicating no observation), return `np.nan`.
    No `np.inf` will appear in the result.

    Parameters
    ----------
    X : ndarray of shape (n_s,) or (n_s, n_f)
        The data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    weights : ndarray of shape (n_s, n_c)
        The weight array for each sample and cluster. Must be all non-negative. Support for 
        `weights` of shape (n_s,) can be added later if needed.

    Returns
    -------
    ndarray of shape (n_c,) or (n_c, n_f)
        The weighted mean for each cluster.
    """
    # valid X
    assert X.ndim in [1, 2]   # (n_s,) or (n_s, n_f)
    X_2d = check_2d_array(X, assert_na=False)   # (n_s, n_f)
    # valid weights
    weights = check_2d_array(weights, assert_na=False)   # (n_s, n_c)
    assert len(X_2d) == len(weights)
    assert (weights >= 0).all()
    # 
    weighted_sum = weights.T @ X_2d        # (n_c, n_f)
    Ns = weights.sum(axis=0, keepdims=True).T   # (n_c, 1)
    means_ = weighted_sum / Ns   # (n_c, n_f)
    if X.ndim == 1: means_ = means_.squeeze()
    return means_        # (n_c,) or (n_c, n_f)

# reviewed
def weighted_mean_std_cluster(X: np.ndarray, weights: np.ndarray, bias=False) -> np.ndarray:
    """
    Compute the weighted means and standard deviations for each cluster.

    In extreme cases leading to NaNs (otherwise, all values are normal):
    - No observation: both `var_` and `factor` will be NaNs, and standard deviation will also be NaN.
    - Only one observation: `var_` will be zero, while `factor` will be `np.inf`. When considering the debiasing 
      factor, this results in NaN standard deviations.

    Parameters
    ----------
    X : ndarray of shape (n_s,) or (n_s, n_f)
        The data matrix, where `n_s` is the number of samples and `n_f` is the number of features.

    weights : ndarray of shape (n_s, n_c)
        The weight array for each sample and cluster. Must be all non-negative.

    bias : bool, optional (default=False)
        If False, apply a debiasing factor to the variance calculation.

    Returns
    -------
    means_ : ndarray of shape (n_c,) or (n_c, n_f)
        The weighted mean for each cluster.

    stds_ : ndarray of shape (n_c,) or (n_c, n_f)
        The weighted standard deviation for each cluster.
    """
    X_2d = check_2d_array(X, assert_na=False)    # (n_s, n_f)
    means_ = weighted_mean_cluster(X_2d, weights)   # (n_c, n_f)
    sq_means_ = weighted_mean_cluster(X_2d ** 2, weights)   # (n_c, n_f)
    var_ = sq_means_ - means_ ** 2  # (n_c, n_f)
    if not bias:    # debiase factor, see: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        V1 = weights.sum(axis=0, keepdims=True)     # (1, n_c)
        V2 = (weights**2).sum(axis=0, keepdims=True)   # (1, n_c)
        factor = 1. / (1. - V2/V1**2)   # (1, n_c)
        factor = factor.T   # (n_c, 1)
        var_ *= factor  # (n_c, n_f)
    stds_ = np.sqrt(var_)  # (n_c, n_f)
    if X.ndim == 1:
        return means_.squeeze(), stds_.squeeze()
    return means_, stds_


# index.py
"""
Helpers for working with the index of pandas objects, typically of type `datetime.date`.

This module provides functions to filter and align the index of pandas Series 
and DataFrames. The functionality ensures proper handling of date-based indices and 
alignment of pandas objects.

Depends on
----------
utils.validation : Module
"""

from .validation import *

# reviewed
def filter_date_range(obj: PD_TYPE, start_date: DATE_TYPE = None, end_date: DATE_TYPE = None) -> PD_TYPE:
    """
    Filter a pandas Series or DataFrame with a `datetime.date` index by a specified date range.
    Returns a copy of the filtered object for data safety.

    Parameters
    ----------
    obj : Series or DataFrame
        The pandas object to filter, which must have an index of dtype `datetime.date`.

    start_date : str, datetime.date, or None, optional
        The start date of the range. If `None`, no start date filter is applied.

    end_date : str, datetime.date, or None, optional
        The end date of the range. If `None`, no end date filter is applied.

    Returns
    -------
    Series or DataFrame
        A copy of the filtered pandas object.
    """
    assert is_ser_df(obj)
    start_date, end_date =  check_datetime_date(start_date), check_datetime_date(end_date)
    if start_date is not None: obj = obj.loc[start_date:]
    if end_date is not None: obj = obj.loc[:end_date]
    return obj.copy()

# reviewed
def align_index(x: PD_TYPE, y: PD_TYPE) -> PD_TYPE:
    """
    Return a subset of `x` so that its index aligns with the index of `y`. 
    Returns a copy of the subset for data safety.

    Parameters
    ----------
    x : Series or DataFrame
        The pandas object whose index is to be aligned with `y`.

    y : Series or DataFrame
        The pandas object whose index is used for alignment.

    Returns
    -------
    Series or DataFrame
        A copy of `x` with its index aligned to `y`.
    """
    return x.loc[y.index].copy()    # throw error if the index is not contained

# reviewed
def align_x_with_y(x: NUMERICAL_OBJ_TYPE, y: NUMERICAL_OBJ_TYPE) -> NUMERICAL_OBJ_TYPE:
    """
    Align `x` with `y`. If both `x` and `y` are pandas objects, align their indices using 
    `align_index`. If they are not both pandas objects, assert that their lengths match.
    Returns a copy for data safety.

    Parameters
    ----------
    x : ndarray, Series, or DataFrame
        The first numerical object to align.

    y : ndarray, Series, or DataFrame
        The second numerical object to align.

    Returns
    -------
    ndarray, Series, or DataFrame
        A copy of `x`, aligned with `y`.
    """
    if is_ser_df(x) and is_ser_df(y): return align_index(x, y)
    # not all pd objects, assert that lens match
    assert is_same_len(x, y), "the two input arrays should be of the same length"
    return x.copy()


# validation.py
"""
Module of functions to validate input/output and parameters in functions or estimators.

This module provides general validation functions and does not depend on any custom modules.
"""

import numpy as np
import pandas as pd
import numbers
from typing import Union, Optional, Dict
import datetime

# custom data types
PD_TYPE = Union[pd.Series, pd.DataFrame]
NUMERICAL_OBJ_TYPE = Union[np.ndarray, PD_TYPE]
SER_ARR_TYPE = Union[np.ndarray, pd.Series]
DF_ARR_TYPE = Union[np.ndarray, pd.DataFrame]
DATE_TYPE = Optional[Union[str, datetime.date]]

pd.set_option('display.width', 300)

###############################
## convert input types 
###############################

# reviewed
def is_no_nan(obj: NUMERICAL_OBJ_TYPE) -> bool:
    """
    Check whether an object does not contain any NaN or None values.

    Parameters
    ----------
    obj : Array/Series/DataFrame
        The input numerical object to check. It can be a numpy array, pandas Series, 
        or pandas DataFrame.

    Returns
    -------
    bool
        `True` if the object does not contain any NaN or None values, `False` otherwise.
    """
    return not pd.isna(np.asarray(obj)).any()

# reviewed
def valid_no_nan(obj: NUMERICAL_OBJ_TYPE):
    """
    Assert that an object does not contain any NaN or None values.

    Parameters
    ----------
    obj : Array/Series/DataFrame
        The input numerical object to check. It can be a numpy array, pandas Series, 
        or pandas DataFrame.

    Raises
    ------
    AssertionError
        If the object contains NaN or None values.
    """
    assert is_no_nan(obj), f"input numerical object contains NaNs."
    return 

# reviewed
def check_2d_array(X: NUMERICAL_OBJ_TYPE, single_col=False, dtype=None, assert_na=True) -> np.ndarray:
    """
    Convert an array-like object into a 2D array. If the input is 1D, a new axis will be appended.
    Only accepts 1D and 2D inputs. If `single_col` is True, the function will assert that 
    `X.shape[1] == 1`. The function returns a copy for data safety.

    Parameters
    ----------
    X : Array/Series/DataFrame
        Array-like object (numpy array, pandas Series, or pandas DataFrame). Raises an exception if 
        the dimensionality is not 1 or 2.
    
    single_col : bool, optional (default=False)
        If True, assert that `X.shape[1] == 1`, ensuring that the input contains only one column.

    dtype : data-type, optional
        Desired numpy data type for the returned array.

    assert_na : bool, optional (default=True)
        Whether to assert that the input `X` does not contain any NA values.

    Returns
    -------
    np.ndarray
        A 2D numpy array.
    """
    X = np.array(X, dtype=dtype)
    if X.ndim == 1: X = X[:, np.newaxis]    # append new axis
    assert X.ndim == 2
    if single_col: assert X.shape[1] == 1
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def check_1d_array(X: NUMERICAL_OBJ_TYPE, dtype=None, assert_na=True) -> np.ndarray:
    """
    Convert an array-like object into a 1D array. The function returns a copy for data safety.

    Parameters
    ----------
    X : Array/Series/DataFrame
        Array-like object (numpy array, pandas Series, or pandas DataFrame). Raises an exception if 
        the dimensionality after calling `.squeeze()` is not 1.

    dtype : data-type, optional
        Desired numpy data type for the returned array.

    assert_na : bool, optional (default=True)
        Whether to assert that the input `X` does not contain any NA values.

    Returns
    -------
    np.ndarray
        A 1D numpy array.
    """
    X = np.array(X, dtype=dtype).squeeze()
    assert X.ndim == 1
    if assert_na: valid_no_nan(X)
    return X

# reviewed
def check_datetime_date(date: DATE_TYPE) -> Optional[datetime.date]:
    """
    Convert a date-like object into a `datetime.date` object. If the input is `None`, 
    return `None`.

    Parameters
    ----------
    date : str, datetime.date, or None
        The input date-like object to be converted. Can be a string, a datetime object, 
        or `None`.

    Returns
    -------
    datetime.date or None
        A `datetime.date` object if the input is a valid date-like object, otherwise `None`.
    """
    if date is None: return None
    return pd.Timestamp(date).date()

###############################
## binary checks
###############################

# reviewed
def is_ser(obj) -> bool:
    """
    Check whether the input object is a Series.
    """
    return isinstance(obj, pd.Series)

# reviewed
def is_df(obj) -> bool:
    """
    Check whether the input object is a DataFrame.
    """
    return isinstance(obj, pd.DataFrame)

# reviewed
def is_ser_df(obj) -> bool:
    """
    Check whether the input object is a Series/DataFrame.
    """
    return isinstance(obj, PD_TYPE) 

# reviewed
def is_numbers(x) -> bool:
    """
    Check whether the input is a scalar number.
    """
    return isinstance(x, numbers.Number)

# reviewed
def is_same_len(*args) -> bool:
    """
    Check whether all input arguments have the same length.

    Parameters
    ----------
    *args : iterable
        Variable number of input iterables (e.g., lists, arrays, or other iterable objects).

    Returns
    -------
    bool
        `True` if all input arguments have the same length, `False` otherwise.
    """
    return len(set(len(x) for x in args)) == 1

# reviewed
def is_same_index(*args) -> bool:
    """
    Check whether the index of all input pandas Series or DataFrames are exactly the same.
    This function is typically used to verify if the date indices of different Series/DataFrames 
    align with each other.

    Parameters
    ----------
    *args : Series or DataFrame
        Variable number of pandas Series or DataFrame objects whose indices are to be compared.

    Returns
    -------
    bool
        `True` if all input Series/DataFrames have the same index, `False` otherwise.
    """
    assert is_same_len(*args)
    index_this = None
    for item in args:
        # assert is_ser_df(item)
        if index_this is None: # the first item
            index_this = item.index
            continue 
        index_that = item.index
        if not (index_this==index_that).all():
            return False
    return True

###############################
## output cast in pd types 
###############################

# reviewed
def getattr_(obj: object, key: Optional[str]): 
    """
    Retrieve the attribute `key` from the object `obj`. If `key` is `None`, or the object 
    does not have the attribute `key`, return `None`.

    Parameters
    ----------
    obj : object
        The object from which to retrieve the attribute.

    key : str, optional
        The name of the attribute to retrieve. If `None`, the function returns `None`.

    Returns
    -------
    any or None
        The value of the attribute if it exists, otherwise `None`.
    """
    if key is not None and hasattr(obj, key):
        return getattr(obj, key) 
    else:
        return None

# reviewed
def raise_arr_to_pd_obj(arr: np.ndarray, pd_obj: NUMERICAL_OBJ_TYPE, index_key="index", columns_key="columns", return_as_ser=True) -> NUMERICAL_OBJ_TYPE:
    """
    Convert a numpy array into a pandas Series or DataFrame, using the index and columns 
    attributes of `pd_obj` for labeling. If `pd_obj` is not a pandas object, the function 
    returns the array unchanged.

    Parameters
    ----------
    arr : np.ndarray
        The array to be converted into a pandas Series or DataFrame.

    pd_obj : Series, DataFrame, or array-like
        The pandas object from which to extract the index and columns for the new pandas object.

    index_key : str, optional (default="index")
        The attribute name for retrieving the index of the output from `pd_obj`.

    columns_key : str, optional (default="columns")
        The attribute name for retrieving the columns of the output from `pd_obj`.
        Only useful if the parameter `return_as_ser` is set to `False`.

    return_as_ser : bool, optional (default=True)
        If `True`, the function returns a pandas Series using only the index. 
        If `False`, it returns a pandas DataFrame using both the index and columns.

    Returns
    -------
    Series, DataFrame, or np.ndarray
        A pandas Series or DataFrame with index and columns matching those of `pd_obj`, 
        or the original numpy array if `pd_obj` is not a pandas object.
    """
    if not is_ser_df(pd_obj): return arr
    index = getattr_(pd_obj, index_key)
    columns = getattr_(pd_obj, columns_key)
    if return_as_ser: return pd.Series(arr, index=index)
    return pd.DataFrame(arr, index=index, columns=columns)

###############################
## file i/o
###############################

import os

# reviewed
def check_dir_exist(filepath):
    """
    Check whether the directory of the specified file path exists. If it does not exist, 
    create the directory. Handles potential race conditions where multiple processes may 
    attempt to create the directory simultaneously.

    Parameters
    ----------
    filepath : str
        The file path for which the existence of the parent directory is checked.
    """
    dirname = os.path.dirname(filepath)
    if dirname != "":
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname, exist_ok=True)
                print(f"Created folder: {dirname}")
            except FileExistsError:
                # The directory was created by another process between the check and creation
                pass
    return


# __init__.py
# Although this import style is generally discouraged, 
# it works well for our codebase given the simple structure
from .validation import *
from .index import *
from .calculation import *
from .cluster import *

