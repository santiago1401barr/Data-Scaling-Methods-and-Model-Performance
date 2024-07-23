# Data-Scaling-Methods-and-Model-Performance


In this project we will be doing a Reimplementation of the paper "Effect of Data Scaling Methods on Machine Learning Algorithms and Model Performance" which aimed to evaluate eleven Machine Learning algorithms:

1.  Logistic Regression (LS) 
1.  Linear Discriminant Analysis (LDA)
1.  K-Nearest Neighbors (KNN)
1.  Naïve Bayes (NB)
1.  Support Vector Machine (SVM)
1.  XGBoost Algorithm (XGB)
1.  Decision Tree Classifier (DT)
1.  Random Forest Classifier (RF)
1.  Gradient Boost (GB)
1.  AdaBoost (AB)
1.  Extra Tree classifier (ET)

Those ML algorithms going to be evaluated using six different data scaling methods:
1.  Standscale (SS).
1.  MinMax (MM).
1.  MaxAbs (MA).
1.  Robust Scaler (RS).
1. Quantile Transformer (QS).
1.  Without Standarization (WS).

The evaluations are going to be made with the next four metrics:
1. Accuracy
1. Precision
1.  Recall
1.  F1-score

In the paper, the dataset used is from http://archive.ics.uci.edu/ml/datasets/Heart+Disease (UCI DATASET), which has no null data and possibly no outlier data and 303 instances. In contrast, in this project the dataset will come from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset, which has null data, outlier data and 920 instances, same features as the other one.

The reason why in this project is using a slightly different dataset is in order to show my data cleaning skilss. As the purpose of Paper Reimplementation is to recreate the results of the paper as much as possible, we are going to focus to find the same best algorithms as in the paper.


The data set  used in the project contains the following features:

| Attribute | Description                                     | Values                                                           |
|-----------|-------------------------------------------------|------------------------------------------------------------------|
| age       | Age in years                                    | Continuous                                                       |
| sex       | Male/female                                     | 1 = male, 0 = female                                             |
| cp        | Chest pain type                                 | 1 = typical type 1, 2 = typical type angina, 3 = nonangina pain, 4 = asymptomatic |
| thestbps  | Resting blood pressure                          | Continuous value in mm hg                                        |
| chol      | Serum Cholesterol                               | Continuous value in mm/dl                                        |
| Restecg   | Resting electrographic results                  | 0 = normal, 1 = having ST_T wave abnormal, 2 = left ventricular hypertrophy |
| fbs       | Fasting blood sugar                             | 1 ≥ 120 mg/dl, 0 ≤ 120 mg/dl                                     |
| thalach   | Maximum heart rate achieved                     | Continuous value                                                 |
| exang     | Exercise induced angina                         | 0 = no, 1 = yes                                                  |
| oldpeak   | ST depression induced by exercise relative to rest | Continuous value                                              |
| slope     | Slope of the peak exercise ST segment           | 1 = unsloping, 2 = flat, 3 = downsloping                         |
| ca        | Number of major vessels colored by flourosopy   | 0-3 value                                                        |
| thal      | Defect type                                     | 3 = normal, 6 = fixed, 7 = reversible defect                     |


Results:

######  Accuracy

|     | WS   | SS   | MM   | MA   | RS   | QT   |
|-----|------|------|------|------|------|------|
| LR  | 0.54 | 0.55 | 0.55 | 0.56 | 0.57 | 0.57 |
| LDA | 0.57 | 0.55 | 0.55 | 0.55 | 0.57 | 0.55 |
| KNN | 0.45 | 0.55 | 0.58 | 0.58 | 0.55 | 0.59 |
| NB  | 0.53 | 0.52 | 0.52 | 0.52 | 0.53 | 0.53 |
| SVM | 0.48 | 0.52 | 0.53 | 0.52 | 0.55 | 0.55 |
| XGB | 0.59 | 0.58 | 0.58 | 0.58 | 0.59 | 0.58 |
| DT  | 0.51 | 0.51 | 0.51 | 0.50 | 0.52 | 0.51 |
| RF  | 0.55 | 0.55 | 0.56 | 0.56 | 0.55 | 0.55 |
| GB  | 0.58 | 0.57 | 0.58 | 0.57 | 0.58 | 0.58 |
| AB  | 0.50 | 0.51 | 0.51 | 0.51 | 0.50 | 0.51 |
| ET  | 0.54 | 0.54 | 0.55 | 0.56 | 0.56 | 0.54 |


######  Precision

|     | WS   | SS   | MM   | MA   | RS   | QT   |
|-----|------|------|------|------|------|------|
| LR  | 0,30 | 0,37 | 0,31 | 0,33 | 0,42 | 0,35 |
| LDA | 0,37 | 0,34 | 0,34 | 0,34 | 0,37 | 0,33 |
| KNN | 0,24 | 0,37 | 0,40 | 0,39 | 0,33 | 0,39 |
| NB  | 0,34 | 0,32 | 0,32 | 0,32 | 0,34 | 0,31 |
| SVM | 0,17 | 0,28 | 0,21 | 0,21 | 0,30 | 0,31 |
| XGB | 0,39 | 0,37 | 0,37 | 0,37 | 0,39 | 0,37 |
| DT  | 0,36 | 0,34 | 0,34 | 0,35 | 0,33 | 0,36 |
| RF  | 0,31 | 0,34 | 0,36 | 0,35 | 0,43 | 0,38 |
| GB  | 0,42 | 0,34 | 0,35 | 0,33 | 0,42 | 0,36 |
| AB  | 0,32 | 0,31 | 0,31 | 0,31 | 0,32 | 0,31 |
| ET  | 0,38 | 0,31 | 0,33 | 0,35 | 0,38 | 0,31 |


######  Recall

|     | WS   | SS   | MM   | MA   | RS   | QT   |
|-----|------|------|------|------|------|------|
| LR  | 0,31 | 0,35 | 0,31 | 0,32 | 0,37 | 0,34 |
| LDA | 0,36 | 0,34 | 0,34 | 0,34 | 0,36 | 0,34 |
| KNN | 0,25 | 0,34 | 0,37 | 0,36 | 0,33 | 0,37 |
| NB  | 0,34 | 0,33 | 0,33 | 0,33 | 0,34 | 0,31 |
| SVM | 0,23 | 0,29 | 0,29 | 0,28 | 0,31 | 0,31 |
| XGB | 0,38 | 0,36 | 0,36 | 0,36 | 0,38 | 0,36 |
| DT  | 0,34 | 0,33 | 0,34 | 0,34 | 0,33 | 0,36 |
| RF  | 0,32 | 0,33 | 0,34 | 0,34 | 0,35 | 0,36 |
| GB  | 0,39 | 0,34 | 0,35 | 0,33 | 0,39 | 0,35 |
| AB  | 0,32 | 0,31 | 0,31 | 0,31 | 0,32 | 0,31 |
| ET  | 0,35 | 0,32 | 0,33 | 0,34 | 0,36 | 0,32 |



######  F1-Score
