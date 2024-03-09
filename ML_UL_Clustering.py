import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score,completeness_score,homogeneity_score

#k-Means Clustering
#https://www.datacamp.com/tutorial/k-means-clustering-python
home_data = pd.read_csv('dados/housing.csv')

#plotando um gráfico para encontrar o valor K ideal em um algoritmo de agrupamento k-means
plt.subplot(131)
data = list(zip( home_data['longitude'], home_data['latitude'], home_data['median_house_value']))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow method')

plt.subplot(132)
sns.scatterplot(x = home_data['longitude'], y = home_data['latitude'], hue = home_data['median_house_value'])
plt.title("Dados Originais")

#Separando em treino e teste
X_train, X_test, y_train, y_test = (train_test_split(home_data[['latitude', 'longitude']],
                                                     home_data[['median_house_value']], test_size=0.33, random_state=0))

#Normalizando os dados
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

#Ajustando e avaliando o modelo
# Ajustar o modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X_train)

print ('K-means Clustering')
print(f'Silhouette Score: {silhouette_score(X_train, kmeans.labels_):.2f}')
print(f'Completeness Score: {completeness_score(y_train.values.ravel() , kmeans.labels_):.2f}')
print(f'Homogeneity Score: {homogeneity_score(y_train.values.ravel() , kmeans.labels_):.2f}')

plt.subplot(133)
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.title("Dados Originais")
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#DBSCAN
#https://www.datacamp.com/tutorial/dbscan-macroscopic-investigation-python

df = pd.read_csv('dados/customers.csv') #extraido o arquivo csv

plt.subplot(121)
plt.scatter(x=df['Grocery'], y=df['Milk'])
plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Dados Originais")

# Preprocessamento
df = df[["Grocery", "Milk"]]
df = df.astype("float32", copy=False)

#Dimensione os features usando StandardScaler
scaler = StandardScaler()
df_scaled =scaler.fit_transform(df)

#construindo um objeto DBSCAN  15 pontos de dados em uma vizinhança de raio 0,5
dbsc = DBSCAN(eps=0.5, min_samples=15).fit(df_scaled)
#extraindo os cluster labels e valores discrepantes para traçar os resultados.
labels = dbsc.labels_

print('DBSCAN')
print(f'Silhouette Score: {silhouette_score(df_scaled, labels):.2f}')

# Plotando os clusters atribuídos pelo DBSCAN
plt.subplot(122)
plt.scatter(df.loc[(labels != -1), 'Grocery'], df.loc[(labels != -1), 'Milk'], c=labels[labels!=-1], cmap='viridis', label='Clustered Data')
plt.scatter(df.loc[(labels == -1), 'Grocery'], df.loc[(labels == -1), 'Milk'], c='black', marker='x', label='Noise')
plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Dados com DBSCAN")
plt.legend()
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#Hierarchical Clustering
#https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python

df = pd.read_csv('dados/xy.csv') #extraindo os dados

#plotando os dados originais
plt.subplot(231)
plt.scatter(df['x'],df['y'])
plt.title("Dados Originais")

#Dimensione os features usando StandardScaler
data_scaler = StandardScaler() #criando um objeto do StandardScaler()
scaled_data = data_scaler.fit_transform(df) #Transformar os dados e padroniza-los.

#Plotando os diveros Dendrograma

# Dendrograma para o método complete
complete_clustering = linkage(scaled_data, method="complete", metric="euclidean")
plt.subplot(232)
dendrogram(complete_clustering)
plt.title("Complete-linkage clustering")

# Dendrograma para o método average
average_clustering = linkage(scaled_data, method="average", metric="euclidean")
plt.subplot(233)
dendrogram(average_clustering)
plt.title("Average-linkage clustering")

# Dendrograma para o método single
single_clustering = linkage(scaled_data, method="single", metric="euclidean")
plt.subplot(234)
dendrogram(single_clustering)
plt.title("Single-linkage clustering")

# Dendrograma para o método ward
ward_clustering = linkage(df, method="ward", metric="euclidean")
plt.subplot(235)
dendrogram(ward_clustering)
plt.title("Ward's-linkage clustering")

#Dendrograma para o método median
median_clustering = linkage(df, method="median", metric="euclidean")
plt.subplot(236)
dendrogram(median_clustering)
plt.title("Median-linkage clustering")

plt.tight_layout()
plt.show()