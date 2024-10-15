## Data Collection

I collected this dataset from the [Kaggle website](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). The dataset is related to red variants of the Portuguese "Vinho Verde" wine and consists of 1,599 rows, 11 input features, and 1 target column.

### Input Variables (Features):

1. fixed acidity  
2. volatile acidity  
3. citric acid  
4. residual sugar  
5. chlorides  
6. free sulfur dioxide  
7. total sulfur dioxide  
8. density  
9. pH  
10. sulphates  
11. alcohol  

### Output Variable (Target):

12. quality (A score ranging from 0 to 10, where 0 indicates the worst quality and 10 indicates the best quality.)


## Exploratory Data Analysis and Data Preprocessing

### Data Formatting
Firstly, I checked the data types of all columns to ensure they were in the correct format. After confirming that the data was correctly formatted, I obtained a basic mathematical description of the dataset using the `.describe()` function.

### Handling Missing Values
Next, I checked for and handled any missing values. I found no missing data in the entire dataset, so there was no need to drop or impute any NaN values.

### Handling Duplicates
I then checked for the presence of duplicate entries in the dataset. I discovered 240 duplicated rows and dropped these duplicates, retaining only the first occurrence of each duplicate.

### Handling Outliers
Outliers were addressed using the Interquartile Range (IQR) method. I created two functions:
- **outlier_finder**: Detects and prints the outliers present in each column.
- **outlier_remover**: Removes the detected outliers.

I applied these functions to identify and remove outliers from each relevant column.


## Train Test Split

I transformed the target variable into binary values, where a quality score of 7 or above is classified as good wine (1), while a score below 7 is considered bad wine (0). After dividing the dataset into features and the target variable, I performed a train-test split, allocating 20% of the data for testing and 80% for training the model.


## Feature Selection

To identify the most relevant features for the model, I first computed the **correlation matrix** to examine the Pearson correlation between the features. I then visualized the correlations using a **heatmap** and set a cut-off Pearson correlation value of 0.6 for selecting features.

For any pair of features with a correlation higher than 0.6, I kept the one with the strongest correlation to the target variable and dropped the other. Based on this approach, I dropped the following four features:
- `pH`
- `fixed acidity`
- `citric acid`
- `free sulphur dioxide`


## Balancing the Dataset

I found that the dataset was imbalanced, with 93 instances of good wine and 695 instances of bad wine. This imbalance can cause machine learning models to become biased toward the majority class (bad wine), leading to poor performance when predicting the minority class (good wine).

To address this, I applied the **SMOTE (Synthetic Minority Over-sampling Technique)**, which oversamples the minority class (good wine) by generating synthetic data points. This technique helps balance the dataset, ensuring the model does not become biased and improves its ability to generalize for both classes.



## Feature Scaling

After analyzing the distribution of the features in the dataset, I decided to scale the data due to the varying ranges of the feature values. Features like `residual sugar`, `chlorides`, and `alcohol` exhibited different magnitudes, which could potentially affect the performance of certain machine learning algorithms.

To address this issue, I applied the **MinMax Scaler**, which transformed the feature values to a range between 0 and 1. This normalization ensures that features with larger values do not disproportionately influence the model compared to features with smaller ranges.



## Model Selection

For the model selection process, I considered five classifiers: 
- **Random Forest**
- **XGBoost**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **SGDClassifier**

I evaluated the performance of each classifier using accuracy as the primary performance metric. The accuracy scores for each model were as follows:

- **SGDClassifier**: 0.827411
- **DecisionTreeClassifier**: 0.832487
- **KNeighborsClassifier (n_neighbors=1)**: 0.842640
- **XGBClassifier**: 0.893401
- **RandomForestClassifier**: 0.903553

After analyzing the results, I selected the top three performing models for further evaluation. I conducted cross-validation with 10 folds to assess their accuracy more rigorously. The XGBoost Classifier emerged as the best-performing model, achieving a mean cross-validation score of **0.93**. Therefore, I chose **XGBoost** as the final model for the wine quality prediction task.


## Hyperparameter Tuning

To optimize the XGBoost Classifier, I employed **Grid Search Cross-Validation** (CV) to tune four key hyperparameters:

- **learning_rate**: Controls the step size during optimization.
- **max_depth**: Limits the maximum depth of a tree to prevent overfitting.
- **min_child_weight**: Sets the minimum sum of instance weights required in a child node.
- **gamma**: Defines the minimum loss reduction needed to further split a leaf node.

This process aimed to identify the best hyperparameter combinations for enhancing model performance.


## Model Evaluation

To assess the model's performance, I considered three key metrics: **precision**, **recall**, and **F1 score**. Upon evaluating the model on the test set, I achieved an **accuracy of 88%** and an **ROC-AUC score of 81%**. These metrics provide insight into the model's effectiveness in classifying wine quality as good or bad.

The confusion matrix below provides a detailed breakdown of the model's predictions:

|                | Predicted Bad | Predicted Good |
|----------------|---------------|----------------|
| **Actual Bad** | 153 (TN)      | 12 (FP)        |
| **Actual Good**| 10 (FN)       | 22 (TP)        |

- **True Negatives (TN)**: 153 (Correctly predicted bad wines)
- **False Positives (FP)**: 12 (Incorrectly predicted good wines)
- **False Negatives (FN)**: 10 (Incorrectly predicted bad wines)
- **True Positives (TP)**: 22 (Correctly predicted good wines)

This matrix illustrates that the model performs well, with a high number of true positives and true negatives, although there are some misclassifications that can be addressed in future iterations.
