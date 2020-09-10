# Loan Prediction
This repository contains my very first machine learning project, which objective is to properly classify eligible and non-eligilbe customers, in order to automate the loan application process. This particular classification problem is available at: [Analytics Vidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/) as a part of their hackathons. As the dataset was rather balanced (the proportion of eligible vs. non-eligible customers was 7:3), accuracy (percentage of cerrectly predicted cases) was the metric used to evaluate the results. After testing a wide range of models, SVC with a sigmoid kernel became the method of choice with the accuracy of 82.74%.

The project is divided into 7 parts:
1. Problem Statement
2. Hypothesis generation
3. Data collection
4. Exploratory Data Analysis (EDA)
5. Data pre-processing
6. Model building and evaluating
7. Summary

# Problem statement
"Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer      first applies for home loan and after that company validates the customer eligibility for loan.
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers."

## What does this imply in terms of machine learning modeling?
Basically customers can be divided into two groups: those who are eligilble for loan and those who are not. It means that this is a classification problem and that potential models which can be helpful for segmenting loan applications are:
1. Logistic Regression
2. K-nearest neighbors
3. SVM (Support Vector Machine classifier)
4. Naive Bayes algorithm
5. Decision tree classifier
6. Random forest classifier

Apart from that it is worth to develop and evaluate XGBoost and CatBoost classifiers, as examples of ensemble learning algorithms, which has been shown to perform better in some cases in comparison to above mentioned models.
