#spks i sa numpy array of size trial* timebins*neuron
from pathlib import Path
from datetime import datetime

import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from scipy.ndimage import gaussian_filter1d

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, permutation_test_score, GridSearchCV, \
    RandomizedSearchCV, cross_val_score

from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score

from sklearn.dummy import DummyClassifier

from umap import UMAP
import vowel_in_noise.electrophysiology.population_analysis as vowel_pop
from vowel_in_noise import plot_utils


def process_window(
        w,
        ref_spks,
        nref_spks,
        window_size,
        y_ref,
        y_test,
        reducer_pipeline,
        classifier,
        classifier_kwargs,
):
    clf = classifier(**classifier_kwargs)

    window_ref = ref_spks[:, w:w + window_size, :].reshape(ref_spks.shape[0], -1)
    window_nref = nref_spks[:, w:w + window_size, :].reshape(nref_spks.shape[0], -1)

    # Fit the reducer on the reference space
    reducer_pipeline.fit(window_ref, y=y_ref)

    # Transform the reference and non-reference space
    window_ref_reduced = reducer_pipeline.transform(window_ref)
    window_nref_reduced = reducer_pipeline.transform(window_nref)

    # Fit the classifier on the reference space
    clf.fit(window_ref_reduced, y_ref)

    # Predict on the non-reference space
    y_pred = clf.predict(window_nref_reduced)

    # Compute the accuracy
    ba_score = balanced_accuracy_score(y_test, y_pred)
    f1_score_val = f1_score(y_test, y_pred, average='weighted')

    results = {
        'ba_score': ba_score,
        'f1_score': f1_score_val,
        'w': w,
    }

    return results


def train_ref_classify_rest(
        spks,
        bhv,
        space_ref,
        classify,
        classifier,
        classifier_kwargs,
        reducer,
        reducer_kwargs,
        window_size,
        n_permutations=100,
        n_jobs=-1,
):
    """
    Analyzes spike data using dimensionality reduction and classification.

    Parameters:
    - spks: The spike data.
    - bhv: Behavioral data containing masks and labels.
    - space_ref: Reference space identifier.
    - classify: Column name in bhv to use for classification labels.
    - classifier: Classifier to use.
    - classifier_kwargs: Keyword arguments for the classifier.
    - reducer: Dimensionality reduction method to use.
    - reducer_kwargs: Keyword arguments for the reducer.
    - window_size: Size of the window to use for analysis.

    Returns:
    - Tuple of arrays (results_bl, results_f1) containing the balanced accuracy and F1 scores.
    """
    ref_spks = spks[bhv.Mask == space_ref, :, :]
    nref_spks = spks[bhv.Mask != space_ref, :, :]

    # Z-score with respect to reference space
    spks_mean = np.nanmean(ref_spks, axis=0)
    spks_std = np.nanstd(ref_spks, axis=0)
    spks_std[spks_std == 0] = np.finfo(float).eps
    ref_spks = (ref_spks - spks_mean) / spks_std
    nref_spks = (nref_spks - spks_mean) / spks_std

    reducer_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', reducer(**reducer_kwargs)),
    ])

    y_ref = bhv[bhv.Mask == space_ref][classify].values
    y_nref = bhv[bhv.Mask != space_ref][classify].values
    results_cv = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(process_window)(w, ref_spks, nref_spks, window_size, y_ref, y_nref, reducer_pipeline, classifier,
                                classifier_kwargs) for w in tqdm(range(ref_spks.shape[1] - window_size)))

    results_perm = []
    if n_permutations > 0:
        for n in tqdm(range(n_permutations)):
            y_ref = np.random.permutation(y_ref)
            y_nref = np.random.permutation(y_nref)
            # results_n = Parallel(n_jobs=n_jobs, verbose=1)(delayed(process_window)(w, ref_spks, nref_spks, window_size, y_ref, y_nref, reducer_pipeline, classifier, classifier_kwargs) for w in tqdm(range(ref_spks.shape[1] - window_size)))
            # results_perm.append(results_n)
            clf = DummyClassifier(strategy='stratified')
            results_perm_n = []
            for w in tqdm(range(ref_spks.shape[1] - window_size)):
                window_ref = ref_spks[:, w:w + window_size, :].reshape(ref_spks.shape[0], -1)
                window_nref = nref_spks[:, w:w + window_size, :].reshape(nref_spks.shape[0], -1)

                # Fit the classifier on the reference space
                clf.fit(window_ref, y_ref)

                # Predict on the non-reference space
                y_pred = clf.predict(window_nref)

                # Compute the accuracy
                ba_score = balanced_accuracy_score(y_nref, y_pred)
                f1_score_val = f1_score(y_nref, y_pred, average='weighted')

                results = {
                    'ba_score': ba_score,
                    'f1_score': f1_score_val,
                    'w': w,
                }

                results_perm_n.append(results)
            results_perm.append(results_perm_n)

    results = {
        'cv': results_cv,
        'perm': results_perm,
    }
    # Unpack results
    # results_bl, results_f1 = zip(*results)

    return results


def train_within(
        spks,
        bhv,
        skf,
        space_ref,
        classify,
        classifier,
        classifier_kwargs,
        reducer,
        reducer_kwargs,
        window_size,
        n_permutations=100,
        n_jobs=-1,
):
    reducer_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reducer', reducer(**reducer_kwargs)),
    ])

    ref_spks = spks[bhv.Mask == space_ref, :, :]
    ref_labels = bhv[bhv.Mask == space_ref][classify].values
    cv_results = []
    for i, (train_idx, test_idx) in enumerate(skf.split(ref_spks, ref_labels)):
        print(f'Fold {i} / {skf.get_n_splits()}')
        X_train = ref_spks[train_idx, :, :]
        X_test = ref_spks[test_idx, :, :]
        y_train = ref_labels[train_idx]
        y_test = ref_labels[test_idx]

        # Z-score to train data
        spks_mean = np.nanmean(X_train, axis=0)
        spks_std = np.nanstd(X_train, axis=0)
        spks_std[spks_std == 0] = np.finfo(float).eps
        X_train = (X_train - spks_mean) / spks_std
        X_test = (X_test - spks_mean) / spks_std

        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(process_window)(w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline, classifier,
                                    classifier_kwargs) for w in tqdm(range(ref_spks.shape[1] - window_size)))

        cv_results.append(results)

    results_perm = []
    if n_permutations > 0:
        for n in range(n_permutations):
            print(f'Permutation {n} / {n_permutations}')
            ref_labels = np.random.permutation(ref_labels)
            for i, (train_idx, test_idx) in enumerate(skf.split(ref_spks, ref_labels)):
                print(f'Fold {i} / {skf.get_n_splits()}')
                X_train = ref_spks[train_idx, :, :]
                X_test = ref_spks[test_idx, :, :]
                y_train = ref_labels[train_idx]
                y_test = ref_labels[test_idx]

                # Z-score to train data
                spks_mean = np.nanmean(X_train, axis=0)
                spks_std = np.nanstd(X_train, axis=0)
                spks_std[spks_std == 0] = np.finfo(float).eps
                X_train = (X_train - spks_mean) / spks_std
                X_test = (X_test - spks_mean) / spks_std

                results = Parallel(n_jobs=-1, verbose=1)(
                    delayed(process_window)(w, X_train, X_test, window_size, y_train, y_test, reducer_pipeline,
                                            classifier, classifier_kwargs) for w in
                    tqdm(range(ref_spks.shape[1] - window_size)))

                results_perm.append(results)

    results = {
        'cv': cv_results,
        'perm': results_perm,
    }
    return results


def main():
    # Load and preprocess data here
    data_path = Path(
        '/home/jules/code/vowel_in_noise/vowel_in_noise/vowel_in_noise/electrophysiology/Data/PSTHs_1msBins')
    save_path = Path('/home/jules/code/vowel_in_noise/vowel_in_noise/saved_vars/cosyne_submission_2024')
    time_window = [-0.2, 0.9]
    bin_width = 0.02
    window_for_decoding = 0.1  # in s
    window_size = int(window_for_decoding / bin_width)  # in bins
    smooth_spikes = True
    t = np.arange(time_window[0], time_window[1], bin_width)
    t = np.round(t, 3)

    n_runs = 5

    classifier = SVC
    classifier_kwargs = {'kernel': 'poly', 'C': 1}

    reducer = UMAP
    reducer_kwargs = {
        'n_components': 2,
        # 'random_state': 42,
        'n_neighbors': 10,
        'min_dist': 0.001,
        'metric': 'cosine',
        'n_jobs': 1,
    }

    space_ref = ['No Noise', 'Noise']
    classify = 'F1'

    skf = StratifiedKFold(n_splits=5, shuffle=True)

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'results_{now}.npy'
    results_between = {}
    results_within = {}
    n_permutations = 0
    for run in range(n_runs):
        bhv, spks = vowel_pop.process_and_load_pseudo_pop(
            data_path,
            time_window=time_window,
            bin_width=bin_width,
            smooth_spikes=smooth_spikes,
        )
        results_between[run] = {}
        results_within[run] = {}
        for space in space_ref:
            results_between[run][space] = train_ref_classify_rest(
                spks,
                bhv,
                space,
                classify,
                classifier,
                classifier_kwargs,
                reducer,
                reducer_kwargs,
                window_size,
                n_permutations=n_permutations,
            )

            results_within[run][space] = train_within(
                spks,
                bhv,
                skf,
                space,
                classify,
                classifier,
                classifier_kwargs,
                reducer,
                reducer_kwargs,
                window_size,
                n_permutations=n_permutations,
            )

            # Save results
            results = {'between': results_between, 'within': results_within}
            save_path.mkdir(exist_ok=True)
            np.save(save_path / filename, results)


if __name__ == '__main__':
    main()