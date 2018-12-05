from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import contingency_matrix
from sklearn.decomposition import PCA

df = pd.read_csv("drogas_preprocessadas.csv", index_col=0)

X = df.values

# k means determine k
wss = []
s_gmm = []
s_kmeans = []
ch_gmm = []
ch_kmeans = []

K = range(2,20)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(X)
    kmeans.fit(X)
    gmm = GaussianMixture(n_components=k, covariance_type='full')
    gmm.fit(X)
    
    labels_kmeans = kmeans.predict(X)
    labels_gmm = gmm.predict(X)
    
    # Quanto maior, melhor -> entre -1 e 1
    s_kmeans.append(metrics.silhouette_score(X, labels_kmeans, metric='euclidean'))
    s_gmm.append(metrics.silhouette_score(X, labels_gmm, metric='euclidean'))
    
    ch_kmeans.append(metrics.calinski_harabaz_score(X, labels_kmeans))
    ch_gmm.append(metrics.calinski_harabaz_score(X, labels_gmm))
        
    wss.append(kmeans.inertia_)
            
# Plot the elbow
plt.plot(K, wss, 'bx-')
plt.xlabel('k')
plt.ylabel('WSS')
plt.title('The Elbow Method showing the optimal k')
plt.show()

plt.plot(K, s_kmeans, 'xr-') # plotting t, a separately 
plt.plot(K, s_gmm, 'ob-')
plt.legend(["kmeans", "gmm"])
plt.xlabel('k')
plt.ylabel('Mean Silhouette Coefficient')
plt.title('Mean Silhouette Coefficient for each k')
plt.show()

plt.plot(K, ch_kmeans, 'xr-') # plotting t, a separately 
plt.plot(K, ch_gmm, 'ob-')
plt.legend(["kmeans", "gmm"])
plt.xlabel('k')
plt.ylabel('Calinski and Harabaz score')
plt.title('Calinski and Harabaz score for each k')
plt.show()


#------------------------- Plot dos cluster gerados pelo K-means---------------

kmeans = KMeans(n_clusters=3).fit(X)
kmeans.fit(X)
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)

labels_kmeans = kmeans.predict(X)

pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

df_res_kmeans = pd.DataFrame({'PC1': pca_X[:,0], 'PC2': pca_X[:,1], 'Cluster': labels_kmeans})


l0 = plt.scatter(df_res_kmeans[df_res_kmeans['Cluster'] == 0].PC1, 
                 df_res_kmeans[df_res_kmeans['Cluster'] == 0].PC2, 
                 marker='o', color='b')
l1 = plt.scatter(df_res_kmeans[df_res_kmeans['Cluster'] == 1].PC1, 
                 df_res_kmeans[df_res_kmeans['Cluster'] == 1].PC2, 
                 marker='o', color='r')
l2 = plt.scatter(df_res_kmeans[df_res_kmeans['Cluster'] == 2].PC1, 
                 df_res_kmeans[df_res_kmeans['Cluster'] == 2].PC2, 
                 marker='o', color='y')

plt.legend((l0, l1, l2),
           ('Cluster 1', 'Cluster 2', 'Cluster 3'),  
           loc='lower left',
           fontsize=8)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Clusters gerados pelo K-means")

plt.show()

#--------------------------Plot dos cluster gerados pelo GMM ------------------

labels_gmm = gmm.predict(X)

pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)

df_gmm = pd.DataFrame({'PC1': pca_X[:,0], 'PC2': pca_X[:,1], 'Cluster': labels_gmm})


l0 = plt.scatter(df_gmm[df_gmm['Cluster'] == 0].PC1, 
                 df_gmm[df_gmm['Cluster'] == 0].PC2, 
                 marker='o', color='b')
l1 = plt.scatter(df_gmm[df_gmm['Cluster'] == 1].PC1, 
                 df_gmm[df_gmm['Cluster'] == 1].PC2, 
                 marker='o', color='r')
l2 = plt.scatter(df_gmm[df_gmm['Cluster'] == 2].PC1, 
                 df_gmm[df_gmm['Cluster'] == 2].PC2, 
                 marker='o', color='y')

plt.legend((l0, l1, l2),
           ('Cluster 1', 'Cluster 2', 'Cluster 3'),  
           loc='lower left',
           fontsize=8)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Clusters gerados pelo Bayesian Gaussian Mixture")

plt.show()


#------------------------ Plot Trees -----------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus    
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)

model.fit(X, list(df_res_kmeans.Cluster))
dot_data = export_graphviz(model.estimators_[5], out_file=None, 
                                feature_names=(df.columns),  
                                class_names=['Cluster 1', 'Cluster 2', 'Cluster 3'])

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())


df_importance = pd.DataFrame({"feature": list(df.columns), "importance": model.feature_importances_})

df_importance = df_importance.sort_values('importance')
features_importantes = list(df_importance.tail(7).feature)

df_features_importantes = df[features_importantes]
newX = df_features_importantes.values

new_model = RandomForestClassifier(n_estimators=10)

new_model.fit(newX, list(df_res_kmeans.Cluster))
dot_data = export_graphviz(new_model.estimators_[5], out_file=None, 
                                feature_names=features_importantes,  
                                class_names=['Cluster 1', 'Cluster 2', 'Cluster 3'])

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

df_original = pd.read_csv("BD/drug_consumption_original.data", names=['col'+str(i) for i in range(31)], index_col=0)
df_original.drop("col3", axis=1, inplace=True)#apagar país
df_original.drop("col4", axis=1, inplace=True)#apagar etnia
df_original.columns = df.columns

cols_subs = ["alcool", "anfetamina", "nitrato_amilato", "benzodiazepina", "cafeina", "maconha", "chocolate", "cocaina", "crack", "ecstase", "heroina", "ketamina", "legalidade", "lsd", "metadona", "cogumelos", "nicotina", "semeron", "vsa"]

#df_original = df_original[features_importantes]

vals_dict = {"CL"+str(i): i for i in range(7)}

def substituir(data):
    return vals_dict[data]

for col in cols_subs:
    df_original[col] = df_original[col].apply(substituir)
    
X_original = df_original.values

model_original = RandomForestClassifier(n_estimators=10, max_depth=4, min_samples_leaf = 50)

model_original.fit(X_original, list(df_res_kmeans.Cluster))
dot_data = export_graphviz(model_original.estimators_[5], out_file=None, 
                                feature_names=list(df_original.columns),  
                                class_names=['Cluster 1', 'Cluster 2', 'Cluster 3'])

graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

df_original['cluster'] = df_res_kmeans.Cluster

ax = (df_original[df_original.cluster == 0])[features_importantes].apply(np.median, axis=0).plot.bar()
ax.set_ylabel("Categoria Mediana")
ax.set_title("Mediana das Categorias das Principais Features no Cluster 1")

ax = (df_original[df_original.cluster == 1])[features_importantes].apply(np.median, axis=0).plot.bar()
ax.set_ylabel("Categoria Mediana")
ax.set_title("Mediana das Categorias das Principais Features no Cluster 2")

ax = (df_original[df_original.cluster == 2])[features_importantes].apply(np.median, axis=0).plot.bar()
ax.set_ylabel("Categoria Mediana")
ax.set_title("Mediana das Categorias das Principais Features no Cluster 3")

#------------------------- Percentual de cada Cluster -------------------------

df_res_kmeans[df_res_kmeans.Cluster == 0].shape[0]/df_res_kmeans.shape[0]
df_res_kmeans[df_res_kmeans.Cluster == 1].shape[0]/df_res_kmeans.shape[0]
df_res_kmeans[df_res_kmeans.Cluster == 2].shape[0]/df_res_kmeans.shape[0]