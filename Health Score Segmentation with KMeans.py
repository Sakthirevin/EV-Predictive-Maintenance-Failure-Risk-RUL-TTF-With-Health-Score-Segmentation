import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("EV_Predictive_Maintenance.csv")

# Reduce dataset size for testing (optional)
df = df.sample(n=20000, random_state=42)

# Drop unneeded columns
df = df.drop(columns=['Timestamp', 'Component_Health_Score'])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA (reduce dimensions before clustering & t-SNE)
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# KMeans clustering on PCA-reduced data
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Evaluate clustering quality -measure of how well-separated the clusters are
silhouette = silhouette_score(X_pca, clusters)
print("Silhouette Score:", round(silhouette, 4)) #Silhouette coefficient ranges between −1 and 1

# Re-label clusters by average PCA magnitude (so Poor/Moderate/Good makes sense)
centroid_order = kmeans.cluster_centers_.mean(axis=1).argsort()
cluster_label_map = {idx: label for idx, label in zip(centroid_order, ['Poor', 'Moderate', 'Good'])}
labels_named = pd.Series(clusters).map(cluster_label_map)

# Prepare colors
colors = {'Poor': 'red', 'Moderate': 'blue', 'Good': 'green'}

# t-SNE for 2D visualization
tsne = TSNE(n_components=2, perplexity=30, max_iter=250, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plotting
plt.figure(figsize=(10, 7))
for label in ['Poor', 'Moderate', 'Good']:
    idx = labels_named == label
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                label=label, s=10, alpha=0.6, color=colors[label])

plt.title("Health Status Segmentation via KMeans (PCA + t-SNE)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Clustered Health Status")
plt.grid(True)
plt.tight_layout()
plt.show()