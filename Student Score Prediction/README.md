# 📈 Student Score Prediction – ML Regression Project

This project is part of my 30-Day Machine Learning Challenge with Elevvo Internship.  
The goal is to predict students' exam scores based on the number of study hours using linear and polynomial regression.

## 💡 Project Objective

- Predict exam scores using:
  - ✅ Linear Regression
  - ✅ Polynomial Regression (Bonus)
- Evaluate and compare both models
- Visualize the relationship and model predictions
  
## 📊 Dataset

- Source: [Student Scores Dataset](https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv)
- 25 samples with two columns:
  - `Hours`: Hours studied
  - `Scores`: Exam results

## 🧠 Tools & Libraries

- Python  
- Pandas, NumPy  
- Matplotlib  
- Scikit-learn (LinearRegression, PolynomialFeatures, model evaluation tools)
  
## ✅ Results

| Model               | MSE   | R² Score |
|---------------------|--------|----------|
| Linear Regression   | 18.94 | 0.97     |
| Polynomial (deg=2)  | 21.07 | 0.96     |

✅ The linear model performed slightly better, with a simple and accurate fit.  
The polynomial model added curve flexibility but did not outperform the linear one in this case.

## ✍️ How to Use
You can modify the dataset or model parameters (like polynomial degree) inside the script to explore different results.
For example, try degree=3 in PolynomialFeatures for better flexibility.
## 📈 Visualizations

![plot](preview.png)
