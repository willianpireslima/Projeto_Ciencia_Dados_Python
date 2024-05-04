import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

#1 Extraindo as imagens do banco de dados e

# Especifique o caminho para o arquivo ZIP contendo as imagens
zip_file_path = 'dados/orl_faces.zip'

# Inicialize uma lista para armazenar todas as imagens
images = []

# Abra o arquivo ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Percorra os diretórios de cada sujeito (s1, s2, ..., s40)
    for subject_number in range(1, 41):
        subject_directory = f's{subject_number}/'

        # Percorra as 10 imagens (1.pgm, 2.pgm, ..., 10.pgm) de cada sujeito
        for image_number in range(1, 11):
            image_filename = f'{image_number}.pgm'
            image_path = os.path.join(subject_directory, image_filename)

            # Leia o conteúdo do arquivo diretamente em memória
            with zip_ref.open(image_path) as file:
                image_bytes = file.read()

            # Converta os bytes em uma matriz NumPy
            image_data = np.frombuffer(image_bytes, dtype=np.uint8)

            # Carregue a imagem usando OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                images.append(image)
            else:
                print(f'Não foi possível carregar a imagem: {image_path}')

#2 Fazendo a redução da Dimensionalidade Imagem Por Imagem

# Inicialize uma lista para armazenar todas as imagens reconstruídas
imagens_reconstruidas = []

# Inicialize uma lista para armazenar a variancia explicada
variancia_explicada_reduzida = []

#Variaiveis usadas para mostrar o PCs minimos usados
min_rms_error = float('inf')  # Inicialize o erro mínimo como infinito
optimal_n_components = None  # Inicialize o número ótimo de PCs como None

# Realizando a análise de PCA para todas as imagens
for image in images:
    
    # Converta a imagem em uma matriz de recursos
    features = image.reshape(-1, 1).astype(float)

    # Aplique o PCA para reduzir a dimensionalidade
    n_componentes = 1
    pca = PCA(n_components=n_componentes)
    pca.fit(features)

    # Projete a matriz de recursos nas primeiras componentes principais
    representacao_discreta = pca.transform(features)

    # Reconstrua a imagem a partir da representação discreta
    imagens_reconstruida = pca.inverse_transform(representacao_discreta)

    # A imagem reconstruída está em forma de vetor, converta-a de volta em uma imagem
    imagens_reconstruida = imagens_reconstruida.reshape(image.shape).astype(np.uint8)

    # Adicione a imagem reconstruída à lista de imagens reconstruídas
    imagens_reconstruidas.append(imagens_reconstruida)
    
    # Calcule a variância explicada após a redução da dimensionalidade
    variancia_explicada_reduzida.append(np.sum(pca.explained_variance_ratio_))

    # Calcule o erro RMS entre a imagem original e a reconstruída
    rms_error = root_mean_squared_error(image, imagens_reconstruida)
    
    # Atualize o número mínimo de PCs se o erro RMS for menor
    if rms_error < min_rms_error:
      min_rms_error = rms_error
      optimal_n_components = n_componentes

#3 Realizando Metricas de Comparacao Numerica entre Original E a Reduzida

# Obtenha os Autovetores e Autovalores
autovetores = pca.components_
autovalores = pca.explained_variance_

# Escolha a imagem que deseja comparar (por exemplo, a primeira imagem)
image_original = image
image_reconstruida = imagens_reconstruidas[0]

# Calcule o erro quadrático médio (MSE)
mse = root_mean_squared_error(image_original, image_reconstruida)

# Calcule o erro quadrático médio da raiz (RMSE)
rmse = np.sqrt(mse)

# Calcule o erro médio absoluto (MAE)
mae = mean_absolute_error(image_original, image_reconstruida)

# Calcule o erro percentual absoluto médio (MAPE)
mape = np.mean(np.abs((image_original - image_reconstruida) / image_original)) * 100

# Calcule o logaritmo do erro quadrático médio da raiz (RMSLE)
rmsle = np.sqrt(np.mean(np.log1p(image_original) - np.log1p(image_reconstruida))**2)

#Printandos as Metricas
print("Metricas De Comparacao")
print("Erro Quadrático Médio (MSE)          : ",mse)
print("Erro Quadrático Médio da Raiz (RMSE) : ",rmse)
print("Erro Médio Absoluto (MAE)            : ",mae)
print("Erro Percentual Absoluto Médio (MAPE): ", mape)
print("Log do RMSE (RMSLE)                  : " ,rmsle)
print("Autovetores (Componentes Principais) : ",autovetores)
print("Autovalores (Variância Explicada)    : ",autovalores)
print("Número Mínimo de PCs Encontrados:", optimal_n_components)

#4 Plotando a Variância Explicada Cumulativa
print("\nVariância Explicada Cumulativa\n")
plt.plot(range(1, len(variancia_explicada_reduzida) + 1), np.cumsum(variancia_explicada_reduzida), marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Cumulativa')
plt.title('Variância Explicada Cumulativa')
plt.show()

#5 Mostra como forma de comparar a imagem origina e reduzida

#Escolha a imagem de teste para ser usada na reconstrucao e busca
image_test = imagens_reconstruidas[0]

def reconstruir_com_min_pcs(image_test, optimal_n_components):
    features = image_test.reshape(-1, 1).astype(float)
    pca = PCA(n_components=optimal_n_components)
    pca.fit(features)
    representacao_discreta = pca.transform(features)
    imagens_reconstruida = pca.inverse_transform(representacao_discreta)
    return imagens_reconstruida.reshape(image_test.shape).astype(np.uint8)

# Usar a função para reconstruir a imagem de teste
image_test_reconstruida = reconstruir_com_min_pcs(image_test, optimal_n_components)

print("\nComparacao entre Imagem Original e Reconstruida\n")

# Exibir Imagem Original e Reconstruída
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_test, cmap='gray')
plt.title('Imagem de Teste Original')

plt.subplot(1, 2, 2)
plt.imshow(image_test_reconstruida, cmap='gray')
plt.title('Imagem de Teste Reconstruída')
plt.show()

#6 Plotando O Grafico de erros (RMS)

print("\nMostrando o Grafico de Erros\n")

# Inicialize uma lista para armazenar os erros (RMS)
erros_rms = []

# Percorra todas as imagens reconstruídas, exceto a imagem de teste
for imagem_reconstruida in imagens_reconstruidas[1:]:
    # Calcule o erro RMS entre a imagem de teste e a imagem reconstruída
    erro_rms = root_mean_squared_error(image_test, imagem_reconstruida)
    erros_rms.append(erro_rms)

# Visualize o gráfico de erro (RMS) em função do número de componentes principais
plt.plot(range(1, len(erros_rms) + 1), erros_rms, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Erro RMS')
plt.title('Gráfico de Erro (RMS)')
plt.show()

#7 Realizando a Busca de Imagens

print("\nRealizando a Busca de Imagens\n")

# Encontre o índice da imagem com o erro mínimo (RMS)
indice_menor_erro = np.argmin(erros_rms)

# Exiba a imagem de teste e as nove imagens da mesma pessoa
plt.figure(figsize=(12, 6))
plt.subplot(3, 4, 1)
plt.imshow(image_test, cmap='gray')
plt.title('Imagem de Teste')

# Exiba as nove imagens da mesma pessoa
pessoa_idx = indice_menor_erro // 10  # Índice da pessoa correspondente
for i in range(1, 10):
    img_idx = pessoa_idx * 10 + i
    img = imagens_reconstruidas[img_idx]
    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Imagem {i}', y=1.02)  # Ajuste o valor de 'y' para controlar o espaçamento
plt.subplots_adjust(hspace=0.5)  # Ajusta o espaçamento vertical entre os subplots
plt.show()