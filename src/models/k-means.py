import utility
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------- initialized data -------
X_train_DF_KMeans = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
scaler = StandardScaler()
xtrain_KMeans = scaler.fit_transform(X_train_DF_KMeans)
ytrain_KMeans = Y_train_DF


# ------- create basic KMeans model -------
model_kmeans = KMeans(n_clusters=2, random_state=42)
model_kmeans.fit(xtrain_KMeans)
y_kmeans = model_kmeans.predict(xtrain_KMeans)

score = accuracy_score(y_true=ytrain_KMeans, y_pred=y_kmeans)
print(f'100% data KMeans accuracy: {score}')
matrix = confusion_matrix(y_true=ytrain_KMeans, y_pred=y_kmeans)
print(f'Confusion matrix:\n{matrix}\n')


# ------- Create PCA - Reduce dimension -------
pca = PCA(n_components=2)
pca.fit(xtrain_KMeans)
print(f'Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}')
print(f'Total explained variance (PC1 + PC2): {pca.explained_variance_ratio_.sum()}')
kmeans_pca = pca.transform(xtrain_KMeans)
kmeans_pca = pd.DataFrame(kmeans_pca, columns=['PC1', 'PC2'])
kmeans_pca['y'] = ytrain_KMeans


# ------- Visualize PCA -------
sns.scatterplot(x='PC1', y='PC2', hue='y', data=kmeans_pca)
plt.title("PCA - Reduced Dimensionality")
plt.show()


# ------- Number of clusters evaluation -------
dbi_list = []
sil_list = []
for n_clusters in range(2, 10):
    model_kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=1000, n_init=30)
    model_kmeans.fit(xtrain_KMeans)
    assignment = model_kmeans.predict(xtrain_KMeans)
    sil = silhouette_score(xtrain_KMeans, assignment)
    dbi = davies_bouldin_score(xtrain_KMeans, assignment)
    sil_list.append(sil)
    dbi_list.append(dbi)


# ------- Plot evaluation metrics -------
plt.plot(range(2, 10), sil_list, marker='o')
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()

plt.plot(range(2, 10), dbi_list, marker='o')
plt.title("Davies-Bouldin Index vs Number of Clusters")
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Bouldin Index")
plt.show()


# ------- Create an updated KMeans model -------
model_kmeans = KMeans(n_clusters=2, random_state=42, max_iter=1000, n_init=30)
model_kmeans.fit(xtrain_KMeans)
y_kmeans = model_kmeans.predict(xtrain_KMeans)
print(f'KMeans clustering accuracy: {accuracy_score(y_true=ytrain_KMeans, y_pred=y_kmeans)}')

 
# ------- Visualize clusters with centroids -------
sns.scatterplot(x='PC1', y='PC2', hue='y', data=kmeans_pca)
plt.scatter(pca.transform(model_kmeans.cluster_centers_)[:, 0], pca.transform(model_kmeans.cluster_centers_)[:, 1], marker='+', s=100, color='red')
plt.title("KMeans Clustering with Centroids")
plt.show()


# ------- Different clustering algorithm (Spectral Clustering) -------
spectral_model = SpectralClustering(n_clusters=8, random_state=42, n_init=10, n_neighbors=10, degree=3)
y_spectral = spectral_model.fit_predict(xtrain_KMeans)
print("Spectral clustering accuracy: ", accuracy_score(y_true=ytrain_KMeans, y_pred=y_spectral))


# ------- Evaluate KMeans model on 70% train data -------
X_train = pd.get_dummies(X_train_DF, columns=['cp', 'restecg', 'slope', 'ca', 'thal'])
Y_train = Y_train_DF
Xtrain, Xvalid, Ytrain, Yvalid = train_test_split(scaler.fit_transform(X_train), Y_train, test_size=0.3, random_state=42)
new_model_kmeans = KMeans(n_clusters=2, random_state=42, max_iter=500, n_init=10)
new_model_kmeans.fit(Xtrain)
print(f'70% (train) data KMeans accuracy: {accuracy_score(y_true=Ytrain, y_pred=new_model_kmeans.predict(Xtrain))}')
print(f'Confusion matrix:\n {confusion_matrix(y_true=Ytrain, y_pred=new_model_kmeans.predict(Xtrain))}\n')
print(f'30% (valid) data KMeans accuracy: {accuracy_score(y_true=Yvalid, y_pred=new_model_kmeans.predict(Xvalid))}')
print(f'Confusion matrix:\n{confusion_matrix(y_true=Yvalid, y_pred=new_model_kmeans.predict(Xvalid))}\n')


# ------- Graph PCA organized -------
pca.fit(X_train_DF_KMeans)
kmeans_pca = pca.transform(X_train_DF_KMeans)
kmeans_pca = pd.DataFrame(kmeans_pca, columns=['PC1', 'PC2'])
kmeans_pca['clustering'] = model_kmeans.predict(X_train_DF_KMeans)


# ------- Scatter plot with cluster centers -------
sns.scatterplot(x='PC1', y='PC2', hue='clustering', data=kmeans_pca)
plt.title("PCA Visualization with KMeans Clustering")
plt.scatter(pca.transform(model_kmeans.cluster_centers_)[:, 0], pca.transform(model_kmeans.cluster_centers_)[:, 1], marker='+', s=100, color='green')
plt.show()


# ------- Save the model -------
joblib.dump(model_kmeans, 'model_kmeans.pkl')
joblib.dump(new_model_kmeans, 'new_model_kmeans.pkl')