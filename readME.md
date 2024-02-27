# Work in progress repo for running dimensionality reduction on honeyco maze data
overview of functions: 
<br>
1. extractlfpandspikedata.py:
2. run_umap_exploratory.py: (archived) run unsuperivsed UMAP on spike matrix data
3. run_umap_exploratory_with_history.py: (archived) run unsuperivsed UMAP on spike matrix data considering bins both 6 bins and before and 6 bins after the current bin, similar to a rolling window
4. run_umap_with_history.py: decode head angle from spike matrix data using UMAP
5. run_umap_with_history_grid_search.py: perform a computationally expensive grid search to find the best parameters for UMAP decoding! I am running this on the cluster as of 27/02.
