import os
import gc
import logging
import pickle

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
from .logging_utils import setup_logger


def rand_start_prob(n_states):
    # Generate random probabilities and normalize using numpy for better numerical stability
    start_prob = np.random.dirichlet(np.ones(n_states))
    # Ensure exact sum of 1 while preserving non-negativity
    start_prob = start_prob / np.sum(start_prob)
    return start_prob


def extract_hmm_data(
    df: pd.DataFrame,
    features: List[str]
) -> np.ndarray:
    """
    Extract and stack feature arrays from dataframe for HMM fitting

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing behavioral data
    features : list of str
        Names of features/columns to extract from df
    
    Returns
    -------
    numpy.ndarray
        Column-stacked array of feature data, shape (n_samples, n_features)
    """
    feature_arrays = [np.array(df[feature]) for feature in features]
    X = np.column_stack(feature_arrays)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled


def fit_single_model(
    X: np.ndarray, 
    n_states: int, 
    n_iter: int = 200, 
    covar_type: str = 'diag'
) -> Tuple[Optional[hmm.GaussianHMM], np.ndarray]:
    """
    Fit a single Gaussian Hidden Markov Model with random initialization.
    
    Initializes transition matrix with uniform probabilities plus small random 
    perturbations to avoid local optima. Uses random seed for reproducibility
    within each fit attempt.
    
    Parameters
    ----------
    X : np.ndarray
        Observation sequences of shape (n_samples, n_features)
    n_states : int
        Number of hidden states in the HMM
    n_iter : int, default=200
        Maximum number of EM iterations
    covar_type : str, default='diag'
        Covariance matrix type ('full', 'diag', 'tied', 'spherical')
    
    Returns
    -------
    Tuple[Optional[hmm.GaussianHMM], np.ndarray]
        Fitted HMM model (None if fitting failed) and original data X
    
    Notes
    -----
    Returns None for model if fitting fails due to convergence issues,
    numerical instability, or other exceptions during training.
    """
    try:
        start_prob = rand_start_prob(n_states)
        transmat = np.random.dirichlet(np.ones(n_states), size=n_states)
        transmat = transmat / transmat.sum(axis=1)[:, np.newaxis]

        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=covar_type,
            init_params='mc',
            n_iter=n_iter,
            random_state=np.random.randint(0, 1000)
        )
        model.startprob_ = start_prob
        model.transmat_ = transmat
        model.fit(X)
        return model, X
    except:
        return None, X


def compute_model_metrics(
    model: hmm.GaussianHMM,
    X: np.ndarray
) -> Dict[str, float]:
    return {
        'BIC': model.bic(X),
        'LL': model.score(X)
    }


def select_best_model(
    models_and_metrics: List[Tuple[hmm.GaussianHMM, Dict[str, float]]]
) -> Tuple[hmm.GaussianHMM, Dict[str, float]]:
   """
   Select HMM with lowest BIC from collection of fitted models

   Parameters
   ----------
   models_and_metrics : list of tuples
       Each tuple contains (model, metrics_dict) where:
       - model is a fitted hmmlearn.hmm.GaussianHMM instance
       - metrics_dict contains 'BIC' and 'LL' keys with corresponding float values
   
   Returns
   -------
   tuple
       (best_model, best_metrics) where:
       - best_model is the HMM instance with lowest BIC
       - best_metrics is its corresponding metrics dictionary
   """
   # Find minimum BIC and corresponding index
   bics = [metrics['BIC'] for _, metrics in models_and_metrics]
   best_idx = np.argmin(bics)
   
   # Return best model and its metrics  
   best_model, best_metrics = models_and_metrics[best_idx]
   
   return best_model, best_metrics


def fit_k_models(
    X: np.ndarray, 
    n_states: int, 
    k_fits: int, 
    n_iter: int = 200, 
    covar_type: str = 'diag', 
    verbose: bool = True
) -> Tuple[Optional[hmm.GaussianHMM], Optional[Dict[str, float]], List[Dict[str, float]]]:
    """
    Fit k randomly initialized HMMs with specified number of states
    Parameters
    ----------
    X : numpy.ndarray
       Input data array of shape (n_samples, n_components)
    n_states : int
       Number of HMM states to fit
    k_fits : int
       Number of random initializations to try
    n_iter : int, optional
       Max iterations for EM algorithm
    covar_type : str, optional
       Covariance type for HMM
    verbose : bool, optional
       Whether to print progress

    Returns
    -------
    tuple
       (best_model, best_metrics, all_metrics) where all_metrics is list of metrics
       from all k fits for logging purposes
    """
    models_and_metrics = []
    attempts = 0
    max_attempts = k_fits * 2
    while len(models_and_metrics) < k_fits and attempts < max_attempts:
        if verbose:
            print(f'Fitting model {len(models_and_metrics)+1}/{k_fits} with {n_states} states')

        model, X = fit_single_model(X, n_states, n_iter, covar_type)
        if model is not None:
            metrics = compute_model_metrics(model, X)
            models_and_metrics.append((model, metrics))
        attempts += 1
        gc.collect()
    if len(models_and_metrics) == 0:
        raise RuntimeError(f'Failed to get any successful fits')
    best_model, best_metrics = select_best_model(models_and_metrics)
    all_metrics = [metrics for _, metrics in models_and_metrics]
    # Cleanup unused models
    for model, _ in models_and_metrics:
        if model is not best_model:
            del model
    gc.collect()

    return best_model, best_metrics, all_metrics


def save_model_and_metrics(
    save_dir: str,
    n_states: int,
    best_model: hmm.GaussianHMM,
    best_metrics: Dict[str, float],
    all_metrics: List[Dict[str, float]]
) -> None:
    """
    Save HMM model and associated metrics to disk.
    
    Saves the fitted HMM model as a pickle file and logs both the best
    model metrics and all fitting attempts for model selection transparency.
    
    Parameters
    ----------
    save_dir : str
        Directory path where model and metrics will be saved
    n_states : int
        Number of hidden states in the HMM (used in filename)
    best_model : hmm.GaussianHMM
        Best fitted HMM model (lowest BIC) to save
    best_metrics : Dict[str, float]
        Metrics dictionary for the best model containing 'BIC' and 'LL'
    all_metrics : List[Dict[str, float]]
        List of metrics from all k fitting attempts for logging purposes
        
    Returns
    -------
    None
        Function saves files to disk but returns nothing
    """
    # Create state-specific directory
    state_dir = os.path.join(save_dir, f'{n_states}_state')
    os.makedirs(state_dir, exist_ok=True)
    # Setup logger
    logger = setup_logger(state_dir, f'{n_states}_state_HMM_log')
    # Save best model
    model_path = os.path.join(state_dir, f'{n_states}_state_best_fit.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    # Log metrics
    logger.info(f"{'='*50}")
    logger.info(f"Best Model Metrics:")
    logger.info(f"{'='*50}")
    logger.info(f"BIC: {best_metrics['BIC']:.2f}")
    logger.info(f"Log Likelihood: {best_metrics['LL']:.2f}")
    logger.info(f"\n{'='*50}")
    logger.info(f"All Fitting Attempts:")
    logger.info(f"{'='*50}")
    # Format all metrics
    for i, metrics in enumerate(all_metrics, 1):
        logger.info(f"Attempt {i:2d}:")
        logger.info(f"    BIC: {metrics['BIC']:.2f}")
        logger.info(f"    Log Likelihood: {metrics['LL']:.2f}")


def fit_multiple_state_models(
    data: np.ndarray,
    state_counts: List[int],
    hmm_dir: str,
    k_fits_per_state: int = 20,
    n_iter: int = 200,
) -> Dict[int, Dict[str, float]]:
    """
    Fit HMMs with different state counts and save results.
    
    For each specified state count, fits multiple randomly initialized HMMs,
    selects the best model based on BIC, and saves model and metrics to disk.
    
    Parameters
    ----------
    data : np.ndarray
        Observation data for HMM fitting, shape (n_samples, n_features)
    state_counts : List[int]
        List of state counts to test (e.g., [2, 3, 4, 5])
    k_fits_per_state : int, default=20
        Number of random initializations per state count
    n_iter : int, default=200
        Maximum EM iterations per model fit
    hmm_dir : str
        Base directory for saving models and metrics
        
    Returns
    -------
    Dict[int, Dict[str, float]]
        Mapping from n_states to best model metrics for model comparison
    """
    os.makedirs(hmm_dir, exist_ok=True)
    results = {}

    for n_states in state_counts:
        best_model, best_metrics, all_metrics = fit_k_models(
            X=data,
            k_fits=k_fits_per_state,
            n_states=n_states,
            n_iter=n_iter
        )

        save_model_and_metrics(
            save_dir=hmm_dir,
            n_states=n_states,
            best_model=best_model,
            best_metrics=best_metrics,
            all_metrics=all_metrics
        )

        results[n_states] = best_metrics
        gc.collect()

    return results


def load_best_model(
    hmm_dir: str, 
    n_states: int
) -> hmm.GaussianHMM:
    """
    Load best fitting HMM model for given number of states.
    
    Parameters
    ----------
    hmm_dir : str
        Directory containing saved HMM models
    n_states : int
        Number of states in the model to load
        
    Returns
    -------
    hmm.GaussianHMM
        Loaded HMM model instance
    """
    model_path = os.path.join(hmm_dir, f'{n_states}_state', f'{n_states}_state_best_fit.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_state_features(
    hmm_dir: str, 
    n_states: int, 
    feature_names: List[str], 
    state_order: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Returns absolute standardized state feature means with optional state reordering.
    
    Loads HMM model and returns raw state means (standardized feature values)
    showing each state's position in the standardized feature space.
    
    Parameters
    ----------
    hmm_dir : str
        Directory containing saved HMM models
    n_states : int
        Number of states in the model
    feature_names : List[str]
        Names of behavioral features
    state_order : Optional[List[int]], default=None
        1-based state ordering for output (e.g., [3, 1, 2])
        
    Returns
    -------
    pd.DataFrame
        Absolute standardized feature means with states as rows, features as columns
    """
    model = load_best_model(hmm_dir, n_states)
    state_means = model.means_
    # Create DataFrame with raw standardized means
    df_means = pd.DataFrame(state_means,
                           columns=feature_names,
                           index=[f'State {i+1}' for i in range(n_states)])
    # Reorder if state_order provided
    if state_order is not None:
        zero_based_order = [i-1 for i in state_order]  # Convert to 0-based indexing
        df_means = df_means.iloc[zero_based_order]
        df_means.index = [f'State {i}' for i in state_order]  # Preserve original state numbers
    
    return df_means


def get_demeaned_state_features(
    hmm_dir: str, 
    n_states: int, 
    feature_names: List[str], 
    state_order: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Returns within-state demeaned feature scores with optional state reordering.
    
    Loads HMM model and computes state means relative to each state's average
    across all features, revealing relative feature strengths within each state.
    
    Parameters
    ----------
    hmm_dir : str
        Directory containing saved HMM models
    n_states : int
        Number of states in the model
    feature_names : List[str]
        Names of behavioral features/features
    state_order : Optional[List[int]], default=None
        1-based state ordering for output (e.g., [3, 1, 2])
        
    Returns
    -------
    pd.DataFrame
        Demeaned feature scores with states as rows, features as columns
    """
    model = load_best_model(hmm_dir, n_states)
    state_means = model.means_
    # Calculate within-state demeaned scores
    state_averages = state_means.mean(axis=1, keepdims=True)
    demeaned_within_state = state_means - state_averages
    # Create DataFrame
    df_demeaned = pd.DataFrame(demeaned_within_state,
                              columns=feature_names,
                              index=[f'State {i+1}' for i in range(n_states)])
    # Reorder if state_order provided
    if state_order is not None:
        zero_based_order = [i-1 for i in state_order]  # Convert to 0-based indexing
        df_demeaned = df_demeaned.iloc[zero_based_order]
        df_demeaned.index = [f'State {i}' for i in state_order]  # Preserve original state numbers

    return df_demeaned
