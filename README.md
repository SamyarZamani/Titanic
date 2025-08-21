# 🚢 Titanic Survival Prediction

This project is part of a classic Kaggle competition:  
**[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/)**  

The goal of the project is to predict the survival of passengers aboard the Titanic using various machine learning algorithms.  
This dataset is often considered the "Hello World" of data science and is a great starting point for learning supervised machine learning techniques.

---

## 📂 Dataset

The dataset contains passenger information such as age, sex, ticket class, fare, and whether they survived the disaster.  

- **train.csv** → Training data with labels (`Survived` column)  
- **test.csv** → Test data without labels  
- **gender_submission.csv** → Sample submission file  

🔗 Dataset Source: [Titanic Dataset on Kaggle](https://www.kaggle.com/competitions/titanic/data)

---

## 🛠️ Project Workflow

1. **Data Loading & Cleaning**
   - Load the dataset (`train.csv` and `test.csv`)
   - Handle missing values (e.g., `Age`, `Cabin`, `Embarked`)
   - Convert categorical variables (`Sex`, `Embarked`) into numerical features using encoding
   - Feature engineering (e.g., Family size, Title extraction)

2. **Exploratory Data Analysis (EDA)**
   - Displayed first rows of the dataset
   - General information about columns
   - Missing value analysis
   - Descriptive statistics
   - Visualizations:
     - Distribution of survivors vs. non-survivors
     - Survival by gender, class, and age groups

3. **Modeling**
   - Trained multiple machine learning models:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Gradient Boosting
   - Hyperparameter tuning with **GridSearchCV**
   - Performance evaluation using **accuracy score**

4. **Evaluation & Results**
   - Compared models using a summary table
   - Selected the best model based on accuracy

---

## 📊 Results

| Model                | Best Parameters                             | Accuracy |
|----------------------|---------------------------------------------|----------|
| Logistic Regression  | {'C': 0.1}                                  | 0.7989   |
| Random Forest        | {'max_depth': 6, 'n_estimators': 200}       | 0.8212   |
| Support Vector Machine (SVM) | {'C': 1, 'kernel': 'rbf'}           | 0.8156   |
| K-Nearest Neighbors  | {'n_neighbors': 5}                          | 0.8156   |
| Gradient Boosting    | {'learning_rate': 0.01, 'n_estimators': 200}| 0.7933   |

✅ **Best Model: Random Forest (82.12% accuracy)**  
The Random Forest model outperformed other algorithms, which is expected since ensemble methods handle small and imbalanced datasets more effectively.  

---

## 🔍 Analysis

- **Why Random Forest performed best?**
  - It combines multiple decision trees to reduce variance and improve stability.
  - Handles missing values and categorical data better compared to linear models.
- **SVM and KNN** also performed competitively, showing that kernel-based and distance-based methods adapt well to this dataset.
- **Logistic Regression** gave a solid baseline but lacks the flexibility to capture complex interactions.
- **Gradient Boosting** underperformed slightly due to the dataset size and potential overfitting with certain hyperparameters.

---

## 📈 Future Improvements

- Use **cross-validation** instead of a single train-test split to improve robustness.  
- Perform **feature scaling** and more **feature engineering** (e.g., extracting deck from cabin, grouping ticket types).  
- Try more advanced models:
  - XGBoost
  - LightGBM
  - Neural Networks
- Evaluate with other metrics beyond accuracy:
  - Precision, Recall, F1-score
  - Confusion Matrix
  - ROC-AUC

---

## 🖼️ Visualizations

Some of the key plots generated in this project include:

- Countplot of survivors (0 = Died, 1 = Survived)  
- Survival rate by gender (females had higher survival rate)  
- Impact of passenger class (`Pclass`) on survival  
- Age distribution of survivors vs. non-survivors  

---

## 📌 Conclusion

This project demonstrates the complete workflow of a **machine learning classification problem**:
from raw dataset exploration and preprocessing, through training multiple models and tuning their hyperparameters, to selecting the best performing algorithm.  

The **Random Forest** model proved to be the most effective with **82.12% accuracy**, showing that ensemble methods are powerful even on small, structured datasets like Titanic.

---

## 🚀 How to Run

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-username/titanic-survival-prediction.git
   cd titanic-survival-prediction

2. Install required libraries:  
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook:  
   ```bash
   jupyter notebook Main.ipynb
