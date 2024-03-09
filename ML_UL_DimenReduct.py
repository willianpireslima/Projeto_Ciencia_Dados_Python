import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA ,FastICA
from sklearn.manifold import TSNE
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import seaborn as sns

#Algortimo PCA para imagens

#0_Extraindo os Dados
df = pd.read_csv('dados/data_iris.csv')

#1_Armazenado as 4 colunas do dataframe
X = df.drop('variedade', axis=1)

#2_Transformando e padroniza-los os dados
scaler = StandardScaler().fit(X)
scaled_x = scaler.transform(X)

#3_Aplicando o PCA
pca = PCA(n_components=2) #Reduzindo os compentes de 4 para 2 para que seja visualiado no grafico
pca.fit(scaled_x) # encaixar os dados em um modelo,
x_PCA = pca.transform(scaled_x) #transformar os dados em um formato mais adequado ao modelo

new_df = pd.DataFrame(x_PCA, columns=['pc1', 'pc2']) #passando os dados é nomeando as colunas para o DT
new_df['variedade'] =df['variedade'] # adcionado  coluna target do grafico original

plt.subplot(131)
sns.scatterplot(x = new_df['pc1'], y = new_df['pc2'], hue = new_df['variedade'])
plt.title("PCA do Grafico Iris")

#t-SNE para images
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

new2_df = pd.DataFrame(X_tsne, columns=['pc1', 'pc2']) #passando os dados é nomeando as colunas para o DT
new2_df['variedade'] =df['variedade'] # adcionado  coluna target do grafico original

plt.subplot(132)
sns.scatterplot(x = new2_df['pc1'], y = new_df['pc2'], hue = new_df['variedade'])
plt.title("t-SNE do Grafico Iris")

#ICA
ica = FastICA(n_components=2)
x_ICA = ica.fit_transform(X)  # estimated independent sources

new3_df = pd.DataFrame(x_ICA, columns=['pc1', 'pc2']) #passando os dados é nomeando as colunas para o DT
new3_df['variedade'] =df['variedade'] # adcionado  coluna target do grafico original

plt.subplot(133)
sns.scatterplot(x = new3_df['pc1'], y = new3_df['pc2'], hue = new3_df['variedade'])
plt.title("ICA do Grafico Iris")
plt.tight_layout()
plt.show()

#Algortimo PCA para reduzir a dimensionalidade da img

# Carregar a imagem
img = imread("dados/arara.pgm")

# Converter a imagem em um vetor unidimensional
img_flat = img.flatten()

# Aplicar a normalização
img_flat = img_flat.astype(np.float64) / 255.0

# Aplicar o PCA
pca = PCA(n_components=0.95)  # Seleciona componentes que explicam 80% da variância
img_PCA = pca.fit_transform(img_flat.reshape(-1, 1))

# Reconstruir a imagem a partir dos componentes principais transformados
img_reconstruida = pca.inverse_transform(img_PCA).reshape(img.shape)

# Visualizar a imagem original
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Imagem Original')

# Visualizar a imagem original
plt.subplot(132)
plt.imshow(img_PCA, cmap='gray')
plt.title('Imagem com PCA')

# Visualizar a imagem reconstruída
plt.subplot(133)
plt.imshow(img_reconstruida, cmap='gray')
plt.title('Imagem Reconstruída com PCA')
plt.tight_layout()  # Ajusta automaticamente a disposição das figuras
plt.show()

print('Tamanho da Imagem Original             :',img.shape)
print('Tamanho da Imagem com PCA              :',img_PCA.shape)
print('Tamanho da Imagem Reconstruída com PCA :',img_reconstruida.shape)