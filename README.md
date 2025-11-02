
# Feature Engineering for Time Series

This project is a submission for an assignment on feature engineering. The goal is to load a raw time-series dataset, perform a comprehensive set of preprocessing and feature engineering steps, and prepare it for machine learning.

## Dataset
The dataset used is `MLTempDataset1.csv`, which contains hourly temperature readings.

## Feature Engineering Steps Taken

1.  **Load and Explore:** The dataset was loaded into a pandas DataFrame. An initial analysis was performed to check for data types, missing values, and statistical properties.
2.  **Datetime Conversion:** The `Datetime` column was converted from a string `object` to a `datetime` object and set as the DataFrame's index.
3.  **Handle Missing Data:** The dataset was checked for `NaN` values. No missing data was found, so no imputation was required.
4.  **Feature Extraction:** New features were engineered from the `DatetimeIndex` to provide context for the model:
    * **Date-part features:** `hour`, `dayofweek`, `month`
    * **Lag features:** `temp_lag_1hr`, `temp_lag_24hr`
    * **Rolling window features:** `rolling_mean_24hr`
5.  **Encode Categorical Variables:** The new date-part features (`hour`, `dayofweek`, `month`) were **One-Hot Encoded** using `pd.get_dummies()` to prevent the model from assuming a false numerical order.
6.  **Feature Scaling:** All 30+ features were scaled using `StandardScaler` to have a mean of 0 and a standard deviation of 1. This is crucial for distance-based algorithms and PCA.
7.  **Dimensionality Reduction:** **PCA** was applied to demonstrate how to reduce the 30+ features into 2 principal components.
8.  **Feature Selection:** **Correlation Analysis** was performed to find the features most strongly correlated with the target variable (`Hourly_Temp`). The engineered features (`temp_lag_1hr`, `rolling_mean_24hr`) were found to be the most predictive.
9.  **Ethical Analysis:** The project also includes a written discussion on the ethical implications of using protected characteristics (like Gender or Marital Status) in a hypothetical attrition model.

## How to Run
1.  Upload `MLTempDataset1.csv` to your environment (e.g., Google Colab).
2.  Run the `.ipynb` notebook cells from top to bottom.
3.  All necessary libraries are listed in the notebook (pandas, numpy, sklearn, matplotlib, seaborn).
