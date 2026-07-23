# TV Show / Movie Genre Prediction

Predicts the primary genre of a Netflix or Disney+ title using classic
machine learning models, given metadata such as type, cast, country,
rating, duration, release year, and description.

## Dataset

Source: https://github.com/vinayak-ensemble/Dataset-TV-Shows-OTT
(9,338 rows: 7,926 Netflix + 1,412 Disney+ titles, 13 columns)

A local copy is included at `data/tv-shows.csv`.

## Approach (high level)

**Target.** The raw `listed_in` column is multi-label (e.g. "Dramas,
International Movies"). The target is the *first* genre listed ("primary
genre"), reduced to the 10 most frequent genres with everything else
bucketed into `Other`, which frames this as a single-label, multi-class
classification problem.

One EDA finding worth calling out: Netflix and Disney+ label the *same*
genre differently (e.g. Netflix: "Action & Adventure", Disney+:
"Action-Adventure"; "Comedies" vs "Comedy"; "Documentaries" vs
"Documentary"). These are normalized to one spelling before picking the
top 10, otherwise the same genre would be artificially split into two
separate classes. See `GENRE_SYNONYMS` in `src/preprocess.py`.

**Features used:** type, platform, rating, country (top 10 + Other),
release year, duration (parsed to a number), number of cast members,
whether a director is listed, year added to the platform, and the
description (via TF-IDF). `listed_in` itself is dropped from the
features since the target is derived from it — using it as a feature
would leak the answer.

**Preprocessing.** Categorical columns are one-hot encoded, numeric
columns are median-imputed and standardized, and the description is
turned into a 300-term TF-IDF matrix. A handful of rows had a duration
value mistakenly sitting in the `rating` column (e.g. `rating == "66
min"`); these are detected and fixed automatically.

**Models trained (both traditional ML, no deep learning):**
- Logistic Regression — simple, fast, interpretable via coefficients.
- Random Forest — handles non-linear feature interactions, gives free
  feature-importance scores.

**Explainability.** Feature importance / coefficients are read directly
off the trained scikit-learn model (`feature_importances_` for Random
Forest, `coef_` for Logistic Regression) and plotted with matplotlib.

**Train/test split.** An 80/20 random split, done *before* any
frequency-based decisions (which genres/countries count as "top 10")
are made, so those decisions only ever see training data. The split
seed is saved inside the model file so `evaluate.py` can reconstruct
the exact same, untouched test set later.

## Notebook walkthrough

`Genre_Prediction_Walkthrough.ipynb` runs the whole pipeline top to
bottom in one place — load data, EDA, feature engineering, both models
trained, evaluated, and explained, with every table and plot shown
inline. 
Open it in VS Code (with the Jupyter extension) or `jupyter lab`. The
first code cell sets `DATA_PATH` to where the dataset lives on disk —
update that one line if you move the project or data file, then
"Run All".

## How to run

```bash
pip install -r requirements.txt

# Train (repeat with each model name)
python src/train.py --data_path data/tv-shows.csv --model logistic_regression --seed 42
python src/train.py --data_path data/tv-shows.csv --model random_forest --seed 42

# Evaluate on the held-out test set
python src/evaluate.py --model_path outputs/model_logistic_regression.pkl --data_path data/tv-shows.csv
python src/evaluate.py --model_path outputs/model_random_forest.pkl --data_path data/tv-shows.csv

# Explainability plots
python src/explain.py --model_path outputs/model_random_forest.pkl
python src/explain.py --model_path outputs/model_logistic_regression.pkl
```

All commands are run from the `tv-genre-prediction/` project root.

## Results summary

| Model               | Accuracy | Macro F1 | Weighted F1 |
|---------------------|----------|----------|-------------|
| Logistic Regression | 0.595    | 0.644    | 0.584       |
| Random Forest        | 0.603    | 0.653    | 0.588       |

See `Model_Evaluation_Report.docx` for the full write-up, plots, and
model recommendation.
