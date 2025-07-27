# 🛍️ Customer Segmentation using K-Means Clustering

This project uses unsupervised machine learning to segment mall customers based on their **annual income** and **spending score** using the **K-Means clustering algorithm**.

## 📁 Dataset

- **File Name:** `Mall_Customers.csv`
- **Records:** 10 customers
- **Source:** [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation)
- **Attributes:**
  - `CustomerID`
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

## 📊 Project Features

- Loads and previews customer data
- Plots **Gender Distribution**
- Normalizes data using `StandardScaler`
- Uses **Elbow Method** to find optimal cluster count
- Applies **KMeans Clustering**
- Visualizes clusters based on income and spending behavior

## 📌 How to Run

1. Make sure Python 3 is installed.
2. Install the required libraries:
   ```bash
   pip install pandas matplotlib scikit-learn
