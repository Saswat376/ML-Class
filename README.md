# Machine Learning Projects

This repository contains a collection of machine learning projects and scripts developed as part of a machine learning course. The projects cover various topics, including classification, regression, data collection, and data analysis.

## Projects

### Classification Algorithms

#### K-Nearest Neighbors (KNN)
- **Files:** `knn/knn.ipynb`, `knn/knnwl.ipynb`
- **Description:** These notebooks demonstrate the K-Nearest Neighbors algorithm. `knn.ipynb` uses `scikit-learn` to classify the Iris dataset, while `knnwl.ipynb` provides a from-scratch implementation of KNN.

#### Naive Bayes
- **File:** `naive_bayes/naive_bayes.ipynb`
- **Description:** This notebook explains and implements the Gaussian Naive Bayes classifier on the Iris dataset, including performance evaluation and confusion matrix visualization.

#### Decision Tree
- **Files:** `decision tree/dt.ipy`, `decision tree/dtwithgini.ipynb`, `decision tree/cartdt.ipynb`
- **Description:** These files explore the Decision Tree algorithm. `dt.ipy` and `dtwithgini.ipynb` use `scikit-learn` to build and visualize a decision tree for the Iris dataset, with the latter focusing on the Gini impurity criterion. `cartdt.ipynb` is an empty notebook.

### Regression Algorithms

#### Linear Regression
- **Directory:** `linear_regression/`
- **Description:** This directory contains various implementations of linear regression.
  - `linear.py` and `predict.py`: From-scratch linear regression.
  - `linearwl.py` and `predictwl.py`: Linear regression using `scikit-learn`.
  - `gradient_descent.py`: Linear regression using gradient descent.
  - `MultipleLR.py`: Multiple linear regression.

### Model Comparison

- **Files:** `comparision/classifiers.ipynb`, `comparision/regression.ipynb`
- **Description:** These notebooks compare the performance of different classification and regression models.
  - `classifiers.ipynb`: Compares KNN, Naive Bayes, Decision Tree, Logistic Regression, and SGD on the Iris dataset.
  - `regression.ipynb`: Compares Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, KNN, and SGD on the Diabetes dataset.

### Other Topics

#### Normalization
- **File:** `nomalisation/1.ipynb`
- **Description:** This notebook demonstrates different data normalization techniques, including min-max scaling and standardization.

## Data Collection and Generation

- **`1.py`:** Generates a CSV file (`university_records.csv`) with random student data.
- **`sql.ipynb`:** Connects to a MySQL database, retrieves student data, and loads it into a pandas DataFrame.
- **`3.ipynb`:** Reads an `Amazon.csv` file, presumably for data analysis.
- **`Data collection/`:** This directory contains scripts for various data collection methods, including loading from local CSVs, web scraping, and fetching data from APIs.

## Miscellaneous Scripts

- **`1.ipynb`:** Generates and displays a simple plot using `matplotlib`.
- **`2.py`:** Creates a simple plot and saves it as a PNG file.