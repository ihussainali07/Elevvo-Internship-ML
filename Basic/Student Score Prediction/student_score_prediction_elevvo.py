# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)

# Add synthetic features for experimentation
np.random.seed(42)
df['Sleep_Hours'] = np.random.uniform(5, 9, len(df))  # Random sleep hours
df['Participation_Score'] = np.random.uniform(0.5, 1.0, len(df))  # Random score between 0.5 to 1.0

# Display the dataset with new features
print("Dataset with additional features:\n", df.head())

# Visualize original Hours vs Scores
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.grid(True)
plt.show()

# -------------------------------
# Model 1: Linear Regression (Hours only)
# -------------------------------
X1 = df[['Hours']]
y = df['Scores']

X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X1_train, y_train)
y_pred_lin = lin_model.predict(X1_test)

# -------------------------------
# Model 2: Polynomial Regression (Hours only)
# -------------------------------
poly = PolynomialFeatures(degree=2)
X1_poly = poly.fit_transform(X1)
X1p_train, X1p_test, y_train, y_test = train_test_split(X1_poly, y, test_size=0.2, random_state=42)
poly_model = LinearRegression()
poly_model.fit(X1p_train, y_train)
y_pred_poly = poly_model.predict(X1p_test)

# -------------------------------
# Model 3: Linear Regression (with additional features)
# -------------------------------
X2 = df[['Hours', 'Sleep_Hours', 'Participation_Score']]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)
feature_model = LinearRegression()
feature_model.fit(X2_train, y_train)
y_pred_feat = feature_model.predict(X2_test)

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n📊 {name} Performance:")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("MSE:", mean_squared_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R² Score:", r2_score(y_true, y_pred))

# Evaluate all models
evaluate_model("Linear Regression (Hours only)", y_test, y_pred_lin)
evaluate_model("Polynomial Regression (Hours only)", y_test, y_pred_poly)
evaluate_model("Linear Regression (Hours + Sleep + Participation)", y_test, y_pred_feat)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X1_test, y_test, color='red', label='Actual')
plt.plot(X1_test, y_pred_lin, color='blue', label='Linear Prediction')
plt.title('Linear Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()

plt.subplot(1, 2, 2)
# For polynomial, we need sorted values for smooth curve
X_plot = np.linspace(df['Hours'].min(), df['Hours'].max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot_poly = poly_model.predict(X_plot_poly)

plt.scatter(X1_test, y_test, color='red', label='Actual')
plt.plot(X_plot, y_plot_poly, color='green', label='Polynomial Prediction')
plt.title('Polynomial Regression')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.legend()

plt.tight_layout()
plt.show()
