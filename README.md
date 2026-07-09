# Profitability and Yield Prediction on Agricultural Crops of India

## Overview

This project aims to predict the profitability and crop production yields of various agricultural crops across different states in India. It leverages Machine Learning models on two diverse datasets to derive meaningful insights for agricultural planning and economics.

The project performs the following key tasks:
- **Profitability Prediction:** Classifies whether a specific crop will be profitable or not based on costs of cultivation, production costs, state, and support prices.
- **Yield Prediction:** Predicts crop production amounts given parameters like area, season, and crop year.
- **Clustering:** Segments crops/data points using K-Means clustering.
- **Data Visualization:** Provides pie charts and bar plots showing crop distributions and yields across states.
- **Text Mining:** Computes similarity between documents using TF-IDF and Count Vectorizer (as an additional utility/exercise).

## Features

- **Data Preprocessing:** Uses `LabelEncoder` and `OneHotEncoder` from `scikit-learn` to handle categorical data (States, Districts, Crops, Seasons) and applies `StandardScaler` for feature scaling.
- **Profitability Classification Models:**
  - Random Forest Classifier
  - Decision Tree Classifier
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
- **Yield / Production Regression Models:**
  - Random Forest Regressor
  - Decision Tree Regressor
  - K-Nearest Neighbors (KNN) Regressor
- **Unsupervised Learning:**
  - K-Means Clustering to group data and find the optimal number of clusters using the Elbow Method.
- **Exploratory Data Analysis (EDA):** Visualizing state-wise and crop-wise distributions and total yield via Matplotlib.

## Datasets

The project utilizes two primary datasets:
1. `datafile.csv`: Contains information about crop costs, cultivation costs (`A2+FL` and `C2`), production costs, yield, and support prices. This is mainly used for predicting whether a crop is profitable.
2. `crop_production.csv`: Contains agricultural records including State, District, Crop Year, Season, Crop, Area, and Production. This is mainly used for predicting total production (yield regression).

## Project Structure

- `Data plots.py`: Script to visualize crop distributions, state-wise breakdowns, and yield per crop.
- `Preprocessing.py`: Central script that contains the preprocessing steps and applies multiple models (Decision Tree, Logistic Regression, KNN, and K-Means Clustering).
- `Clustering.py`: Focuses on K-Means clustering and visualizing the Elbow curve to find optimal clusters.
- `Random Forest.py`: Implementation of Random Forest Classifier for predicting profitability.
- `Decision Tree.py`: Implementation of Decision Tree Classifier for predicting profitability.
- `Logistic Regression.py`: Implementation of Logistic Regression for profitability classification.
- `KNN.py` & `KNN_final.py`: Implementations of K-Nearest Neighbors for both regression (finding optimal `k` using RMSE) and classification.
- `crop_production_dataset.py`: Regression modeling using Random Forest Regressor to predict production based on area, season, etc.
- `Text mining and finding correlation.py`: A separate text mining script to compute Document Term Matrices and Cosine Similarity using Count Vectorizer and TF-IDF.
- `algos_on_crop_production.ipynb`: Jupyter notebook version containing algorithms, exploratory data analysis, and documentation (designed for Google Colab/Jupyter).

## Prerequisites

To run this project, you need Python installed on your system along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## How to Run

1. Clone or download this repository to your local machine.
2. Ensure you have the datasets (`datafile.csv` and `crop_production.csv`) in the root directory.
3. Run the individual Python scripts based on the model or analysis you want to perform. For example:
   ```bash
   python "Random Forest.py"
   ```
   ```bash
   python "Data plots.py"
   ```
4. Alternatively, you can open `algos_on_crop_production.ipynb` in Jupyter Notebook or Google Colab to run the cells interactively.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Data Visualization:** Matplotlib
