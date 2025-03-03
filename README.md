# Decoding-Inherited-Genetic-Disorders-Machine-Learning-for-Diagnosis-and-Prediction

# Overview  
This project focuses on predicting birth asphyxia using machine learning models, emphasizing inherited genetic disorders. By analyzing genetic data, blood cell counts, and clinical symptoms, we developed and evaluated models to identify patterns indicative of genetic disorders. The study demonstrates the importance of data preprocessing, feature selection, and model optimization in building predictive models for medical conditions.

# Project Aim  
- To develop machine learning models capable of predicting birth asphyxia based on genetic and clinical data.  
- To improve early diagnosis and provide insights for effective treatment.  
- To assess the risk of inheriting genetic disorders from parents.  

# Dataset  
The dataset was obtained from Kaggle:  
🔗 [Predict the Genetic Disorders Dataset](https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes)  

It includes:  
- Patient demographics (age, genetic heritage, etc.)  
- Molecular data (maternal & paternal gene expressions)  
- Clinical attributes (blood cell counts, symptoms, etc.)  

# Methodology  

## 1. Data Preprocessing  
- Identified categorical and numerical variables.  
- Handled missing values using imputation (mean, median, mode).  
- Applied one-hot encoding to categorical variables.  
- Normalized numerical data using Min-Max scaling.  

## 2. Exploratory Data Analysis (EDA)  
- Bar Plots: Visualized the distribution of birth asphyxia cases.  
- Correlation Matrix: Used Spearman correlation to assess relationships between variables.  
- Mutual Information (MI): Analyzed key variables affecting birth asphyxia.  

## 3. Machine Learning Models  
We implemented and compared four ML models:  
- Decision Tree  
- Logistic Regression  
- Random Forest  
- Gradient Boosting  

Data Splitting:  
- Training Set: 80%  
- Test Set: 20%  

## 4. Model Evaluation & Improvements  
- Class Imbalance Handling: Used SMOTE to balance minority classes.  
- Hyperparameter Tuning: Applied Grid Search to optimize model performance.  
- ROC Analysis: Evaluated model performance using AUROC curves.  

# Results  

| Model               | Accuracy | Best Parameters |
|---------------------|----------|----------------|
| Decision Tree      | 62.3%    | max_depth=None, min_samples_split=2 |
| Logistic Regression | 55.5%    | C=0.1, penalty='l1', solver='liblinear' |
| Random Forest      | 63.2%    | max_depth=None, n_estimators=200 |
| Gradient Boosting  | 62.8%    | learning_rate=0.2, max_depth=7, n_estimators=300 |

- Gradient Boosting showed the best overall performance after tuning.  
- ROC curves confirmed weak associations between features and birth asphyxia, indicating the need for further data refinement.  

# Conclusion  
- The models, particularly Logistic Regression and Gradient Boosting, showed promise but struggled with class imbalance.  
- Despite optimization, results suggest limited predictive power, indicating the need for better feature selection and more diverse datasets.  

# Technologies Used  
- Programming Languages: Python  
- Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn  
- Machine Learning Techniques: Feature Engineering, Hyperparameter Tuning, Class Imbalance Handling (SMOTE)  

# References  
1. Alasadi, S. A., & Bhaya, W. S. (2017). Review of data preprocessing techniques in data mining.  
2. [NCBI - Diagnosis of Genetic Diseases](https://www.ncbi.nlm.nih.gov/books/NBK132142/)  
3. Yu, T., & Zhu, H. (2020). Hyper-parameter optimization: A review of algorithms and applications.  
4. Chen, X., & Ishwaran, H. (2012). Random forests for genomic data analysis.  

