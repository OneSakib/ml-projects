## Phase from ML/DL models
#### Typical ML Engineering Phases
1. Problem Definition
        ↓
2. Data Collection
        ↓
3. EDA (Exploratory Data Analysis)
        ↓
4. Data Preprocessing / Feature Engineering
        ↓
5. Model Selection
        ↓
6. Model Training
        ↓
7. Model Evaluation
        ↓
8. Model Optimization / Tuning
        ↓
9. Deployment / Hosting
        ↓
10. Monitoring & Maintenance



##### 1. Problem Definition
Sabse pehle decide karte hain:
- Problem kya hai?
- Classification / Regression / Clustering?
- Input kya hai?
- Output kya chahiye?
- Success metric kya hoga?
Example:
Customer churn predict karna hai → Classification → output 0/1.


#### 2. Data Collection
Data sources:
- CSV / Excel
- Database
- APIs
- Web scraping
- Kaggle
- Logs
- Sensors etc.


#### 2. Data Collection
Data sources:
- CSV / Excel
- Database
- APIs
- Web scraping
- Kaggle
- Logs
- Sensors etc.
3. EDA
Data ko samajhna:
Missing values
Duplicate values
Outliers
Distribution
Correlation
Class imbalance
Categorical vs numerical columns

TOOLS:
pandas
numpy
matplotlib
seaborn


#### 4. Preprocessing + Feature Engineering
Data ko model ke usable format mein convert karna.
For example:
Missing values → Imputation
Categorical → Encoding
Numerical → Scaling
Text → Tokenization / Embeddings
Outliers → Handling
Features → Creation/Selection

Example:
Age: 25
Gender: Male
City: Mohali

becomes something like:
Age = 25
Gender_Male = 1
City_Mohali = 1


#### 5. Model Selection
Different algorithms try karte hain:
Linear Regression
Logistic Regression
Decision Tree
Random Forest
XGBoost
SVM
Neural Network

Problem ke according suitable model choose karte hain.


#### 6. Model Training
Training data se model patterns learn karta hai.
model.fit(X_train, y_train)


Yahi actual learning phase hai.


#### 7. Model Evaluation
Model unseen data par kaisa perform kar raha hai?
Classification:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
Confusion Matrix

Regression:
MAE
MSE
RMSE
R²

Important: training accuracy alone sufficient nahi hoti.



#### 8. Model Optimization / Tuning
Performance improve karte hain:
Hyperparameter tuning
Feature selection
Cross-validation
Regularization
Model architecture changes
Data balancing

Example:
RandomForest(    n_estimators=200,    max_depth=10)


#### 9. Deployment / Hosting
Trained model ko actual application mein use karte hain.
For example:
ML Model
   ↓
FastAPI
   ↓
Docker
   ↓
AWS EC2 / Kubernetes
   ↓
Frontend / Mobile App

Example API:
POST /predict

Input:
```python
{
    "age": 30,
    "income": 50000
}

Output:
{
    "prediction": "Will Churn"
}

```


#### 10. Monitoring & Maintenance
Ye ML Engineer ke role ka bahut important part hai jo beginners often miss karte hain.
Production mein monitor karte hain:
Model performance
Prediction distribution
Data drift
Concept drift
Latency
Errors
CPU / GPU / Memory

Agar real-world data change ho gaya aur model ki performance gir gayi:
New Data
   ↓
Retraining
   ↓
New Model
   ↓
Evaluation
   ↓
Redeployment 