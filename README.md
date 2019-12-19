# Group 49 Miniproject 2 Code

## Data structure
  
  The data structure assumed by these scripts is as follows:
    ```
    ./data/train/<neg/pos>   # dirs containing labeled training txt files
    ./data/test              # dir containing the unlabeld test txt files
    ```

## Files
  - `fig_algorithms_comparison.py`: run to reproduce the data used in fig 1
  - `fig_features_comparison.py`: run to reproduce the data used in fig 2
  - `kaggle_submissions.py`: generates the models and writes the csv files for
    our top two submissions uploaded to Kaggle
  - `PipelineClasses.py`: dependency, the transformer classes used for our
    feature pipelines
  - `processing.py`: dependency, data and feature processing and submission
    writing
  - `naive_bayes/naive_bayes.py`: our naive bayes from scratch implementation
  - `naive_bayes/<other files>`: data files loaded and written by the naive
    bayes. The feature extraction achieved using
    `features_binary()` in processing.py and exporting to the dense matrix to
    csv files. Ids are computed by `data_no_features()`


