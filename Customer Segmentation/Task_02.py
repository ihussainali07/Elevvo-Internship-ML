import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

csv_path = "D:/INFORMATION/Softwares/VS Code Files/Mall_Customers.csv"
df = pd.read_csv(csv_path)

print("First 5 rows of dataset:")
print(df.head())

gender_counts = df['Gender'].value_counts()
print("\nGender distribution:\n", gender_counts)

plt.figure(figsize=(6, 4))
plt.bar(gender_counts.index, gender_counts.values, color=['skyblue', 'lightcoral'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(axis='y')
plt.show()

required_columns = ['Annual Income (k$)', 'Spending Score (1-100)']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: '{col}' in dataset")

X = df[required_columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

max_clusters = min(len(df), 10)
inertia = []
K_range = range(1, max_clusters + 1)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method - Optimal number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

n_clusters = min(5, len(df))  
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(n_clusters):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'],
                color=colors[i % len(colors)], label=f'Cluster {i}')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
