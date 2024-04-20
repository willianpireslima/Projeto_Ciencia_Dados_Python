import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import cv2

#Treinado os dados
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()# Carregando o dataset MNIST

#Preprocessando os dados
x_train = x_train / 255.0
x_test = x_test / 255.0

#Construindo o modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilando e treinado o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10,validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTeste Acuracia: {test_acc} \n")

#Checando o Overfitting e underfitting
plt.xlabel("Model Complexity - epochs")
plt.ylabel("Error Rate")
plt.title("Loss Curve")
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Test Loss', color='orange')
plt.show()

#Salvar modelo
model.save('dados/mnist.keras')

#Carregando  o modelo salvo
loaded_model = tf.keras.models.load_model('dados/mnist.keras')
loaded_model.summary()

# Especificar o caminho para o arquivo ZIP contendo as imagens
zip_file_path = 'dados/digitos.zip'

# Inicializar uma lista para armazenar todas as imagens
images = []

# Abra o arquivo ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Percorrer os diretórios de cada sujeito (s1, s2, ..., s40)
    for subject_number in range(1, 2):
        subject_directory = f's{subject_number}/'

        # Percorrer as 20 imagens (1.pgm, 2.pgm, ..., 10.pgm) de cada sujeito
        for image_number in range(1, 21):
            image_filename = f'{image_number}.png'
            image_path = os.path.join(subject_directory, image_filename)

            # Ler o conteúdo do arquivo diretamente em memória
            with zip_ref.open(image_path) as file:
                image_bytes = file.read()

            # Converter os bytes em uma matriz NumPy
            image_data = np.frombuffer(image_bytes, dtype=np.uint8)

            # Carregar a imagem usando OpenCV
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Redimensionar a imagem para corresponder à forma de entrada do modelo
            image = np.expand_dims(image, axis=0)

            if image is not None:
                images.append(image)
            else:
                print(f'Não foi possível carregar a imagem: {image_path}')

# Converter a lista de imagens em uma matriz numpy
images = np.array(images)

# Ajustar a forma das imagens
images = images.reshape(-1, 28, 28)

# Prever os rótulos das imagens usando o modelo carregado
predictions = loaded_model.predict(images)

# Exibir as imagens e suas previsões
plt.figure(figsize=(20, 10))
for i in range(len(images)):
    plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray')
    plt.title(f'Label Prevista: {np.argmax(predictions[i])}')
    plt.axis('off')
    plt.tight_layout()
plt.show()
