import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score,completeness_score,homogeneity_score

#k-Means Clustering Simples
#Lendo o arquivo CSV em um DataFrame
iris = pd.read_csv("dados/data_iris.csv")

# label_encoder object knows
label_encoder = preprocessing.LabelEncoder()

#Percorrendo Todas as colunas da lista categorical gerada anteriormente
# O label_encoder esta transformando categorico para numerico
iris['variedade']= label_encoder.fit_transform(iris['variedade'])

#Separando os recursos das variáveis de destino
X = iris.drop(columns=['variedade'])
y = iris['variedade']

#Feature Scaling
ms = MinMaxScaler()
X = ms.fit_transform(X)

#Encontrando o número ideal de clusters para classificação k-means
plt.subplot(121)
css = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    css.append(kmeans.inertia_)
plt.plot(range(1, 11), css)
plt.title('The elbow method')
plt.xlabel('Numero de clusters')
plt.ylabel('CSS') #Soma dos quadrados do cluster

#Implementando Clustering K-Means
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualisando os clusters
plt.subplot(122)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

#Plotando os centroids dos clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 50, c = 'red', label = 'Centroids')
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#Verificar quantas amostras foram rotuladas corretamente
labels = kmeans.labels_
correct_labels = sum(y_kmeans == labels)

print('k-Means Clustering Simples')
print("Resultado: %d de %d amostras foram rotuladas da Iris" % (correct_labels, y.size),'\n')

#k-Means Clustering em Geografia
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
                                                     home_data[['median_house_value']],
                                                     test_size=0.33, random_state=0))

#Normalizando os dados
X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

#Ajustando e avaliando o modelo
# Ajustar o modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(X_train)

print ('K-means Clustering Geografico')
print(f'Silhouette Score: {silhouette_score(X_train, kmeans.labels_):.2f}')
print(f'Completeness Score: {completeness_score(y_train.values.ravel() , kmeans.labels_):.2f}')
print(f'Homogeneity Score: {homogeneity_score(y_train.values.ravel() , kmeans.labels_):.2f}')

plt.subplot(133)
sns.scatterplot(data = X_train, x = 'longitude', y = 'latitude', hue = kmeans.labels_)
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.title("Dados Originais")
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#DBSCAN Outlier
# Carregar os dados
mall_custom = pd.read_csv('dados/mall_customers.csv')
X_train = mall_custom[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#Executar DBSCAN
clustering = DBSCAN(eps=12.5, min_samples=4).fit(X_train)
DBSCAN_dataset = X_train.copy()
DBSCAN_dataset.loc[:, 'Cluster'] = clustering.labels_

# Identificar outliers
outliers = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

# Criar a figura e os subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plotar o primeiro scatterplot
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                hue='Cluster', palette='Set2', ax=axes[0], legend='full', s=200)

# Plotar o segundo scatterplot
sns.scatterplot(x='Age', y='Spending Score (1-100)',
                data=DBSCAN_dataset[DBSCAN_dataset['Cluster'] != -1],
                hue='Cluster', palette='Set2', ax=axes[1], legend='full', s=200)

# Adicionar outliers aos plots
axes[0].scatter(outliers['Annual Income (k$)'], outliers['Spending Score (1-100)'], s=10, label='outliers', c="k")
axes[1].scatter(outliers['Age'], outliers['Spending Score (1-100)'], s=10, label='outliers', c="k")

# Adicionar legendas aos subplots
axes[0].legend()
axes[1].legend()

# Configurar o tamanho da fonte das legendas
plt.setp(axes[0].get_legend().get_texts(), fontsize='12')
plt.setp(axes[1].get_legend().get_texts(), fontsize='12')

# Mostrar os plots
plt.show()

print('\nDBSCAN Outliers :\n',outliers,'\n')

#DBSCAN Ruido
#https://www.datacamp.com/tutorial/dbscan-macroscopic-investigation-python

customers = pd.read_csv('dados/customers.csv') #extraido o arquivo csv

plt.subplot(121)
plt.scatter(x=customers['Grocery'], y=customers['Milk'])
plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Dados Originais")

# Preprocessamento
customers = customers[["Grocery", "Milk"]]
customers = customers.astype("float32", copy=False)

#Dimensione os features usando StandardScaler
scaler = StandardScaler()
customers_scaled =scaler.fit_transform(customers)

#construindo um objeto DBSCAN  15 pontos de dados em uma vizinhança de raio 0,5
dbsc = DBSCAN(eps=0.5, min_samples=15).fit(customers_scaled)
#extraindo os cluster labels e valores discrepantes para traçar os resultados.
labels = dbsc.labels_

#Armazenando o ruido
DBSCAN_dataset = customers.copy()
DBSCAN_dataset.loc[:, 'Cluster'] = dbsc.labels_
ruido = DBSCAN_dataset[DBSCAN_dataset['Cluster'] == -1]

print('DBSCAN Tabela de Ruido')
print(f'Silhouette Score: {silhouette_score(customers_scaled, labels):.2f}')
print('Ruido : \n',ruido)

# Plotando os clusters atribuídos pelo DBSCAN
plt.subplot(122)
plt.scatter(customers.loc[(labels != -1), 'Grocery'],
            customers.loc[(labels != -1), 'Milk'], c=labels[labels!=-1],
            cmap='viridis', label='Clustered Data')

plt.scatter(customers.loc[(labels == -1), 'Grocery'],
            customers.loc[(labels == -1), 'Milk'],
            c='black', marker='x', label='Noise')

plt.xlabel("Groceries")
plt.ylabel("Milk")
plt.title("Dados com DBSCAN")
plt.legend()
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

#Hierarchical Clustering
#https://www.datacamp.com/tutorial/introduction-hierarchical-clustering-python

df_xy = pd.read_csv('dados/xy.csv') #extraindo os dados

#plotando os dados originais
plt.subplot(231)
plt.scatter(df_xy['x'],df_xy['y'])
plt.title("Dados Originais")

#Dimensione os features usando StandardScaler
data_scaler = StandardScaler() #criando um objeto do StandardScaler()
scaled_data = data_scaler.fit_transform(df_xy) #Transformar os dados e padroniza-los.

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
ward_clustering = linkage(df_xy, method="ward", metric="euclidean")
plt.subplot(235)
dendrogram(ward_clustering)
plt.title("Ward's-linkage clustering")

#Dendrograma para o método median
median_clustering = linkage(df_xy, method="median", metric="euclidean")
plt.subplot(236)
dendrogram(median_clustering)
plt.title("Median-linkage clustering")

plt.tight_layout()
plt.show()