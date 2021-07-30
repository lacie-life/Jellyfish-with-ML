# Import pandas for data processing
import pandas as pd

# Read the dataset
dataset = pd.read_csv("studentclusters.csv")
X = dataset.copy()

# Visualise the data using Scatter plot
X.plot.scatter(x='marks', y='shours')


# Fit and Transform the data for MinMax normalization
from sklearn.preprocessing import minmax_scale
X_scaled = minmax_scale(X)


# import KMeans for clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)

# Fit the input data. Create labels and get inertia
kmeans.fit(X_scaled)
inertia = kmeans.inertia_
labels = kmeans.labels_

# Visualise the clusters
labels = pd.DataFrame(labels)
df = pd.concat([X, labels], axis=1)
df = df.rename(columns={0:'label'})

df.plot.scatter(x='marks', y='shours', c='label', colormap='Set1')


# Elbow method to determine optimum clusters
inertia = []

for i in range(2,16):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt

plt.plot(range(2,16), inertia, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Squared Sum (Inertia)')







