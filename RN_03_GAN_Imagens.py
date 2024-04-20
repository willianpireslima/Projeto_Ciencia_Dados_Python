import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import imageio.v2 as imageio
import os

# Definindo os hiperparâmetros
batch_size = 64
epochs = 10000
z_dim = 20

# Noise para visualização
z_vis = tf.random.normal([10, z_dim])

# Carregando os dados (MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # Normalização dos dados
x_iter = iter(tf.data.Dataset.from_tensor_slices(x_train).shuffle(4 * batch_size).batch(batch_size).repeat())

# Definindo a Rede Neural: Gerador (Generator)
G = tf.keras.models.Sequential([
    tf.keras.layers.Dense(28 * 28 // 2, activation='relu'),  # Camada densa com ativação ReLU
    tf.keras.layers.Dense(28 * 28, activation='sigmoid'),  # Camada densa com ativação Sigmoid
    tf.keras.layers.Reshape((28, 28))  # Reformulando para o formato de imagem (28x28)
])

# Definindo a Rede Neural: Discriminador (Discriminator)
D = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),  # Achata a imagem em um vetor unidimensional
    tf.keras.layers.Dense(28 * 28 // 2, activation='relu'),  # Camada densa com ativação ReLU
    tf.keras.layers.Dense(1)  # Camada densa de saída com 1 neurônio (saída binária)
])

# Funções de perda (Loss functions)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # Função de entropia cruzada binária

def G_loss(D, x_fake): # Função de perda do Gerador (Generator)
    return cross_entropy(tf.ones_like(D(x_fake)),
                         D(x_fake))  # Gerador tenta enganar o Discriminador (rótulos positivos)
def D_loss(D, x_real, x_fake):# Função de perda do Discriminador (Discriminator)
    return (cross_entropy(tf.ones_like(D(x_real)), D(x_real)) +  # Discriminador classifica imagens reais corretamente
            cross_entropy(tf.zeros_like(D(x_fake)), D(x_fake)))  # Discriminador classifica imagens falsas corretamente

# Otimizadores (Optimizers)
G_opt = tf.keras.optimizers.Adam(1e-4)  # Otimizador para o Gerador
D_opt = tf.keras.optimizers.Adam(1e-4)  # Otimizador para o Discriminador

# Treinamento
for epoch in range(epochs):
    z_mb = tf.random.normal([batch_size, z_dim])  # Amostras de ruído aleatório
    x_real = next(x_iter)  # Amostras de imagens reais

    # Gravação das operações para cálculo dos gradientes
    with tf.GradientTape() as G_tape, tf.GradientTape() as D_tape:
        x_fake = G(z_mb)  # Gerar imagens falsas
        G_loss_curr = G_loss(D, x_fake)  # Calcula a perda do Gerador
        D_loss_curr = D_loss(D, x_real, x_fake)  # Calcula a perda do Discriminador

    # Calcula os gradientes
    G_grad = G_tape.gradient(G_loss_curr, G.trainable_variables)
    D_grad = D_tape.gradient(D_loss_curr, D.trainable_variables)

    # Aplica os gradientes aos parâmetros do modelo usando os otimizadores
    G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
    D_opt.apply_gradients(zip(D_grad, D.trainable_variables))

    # Exibe resultados periodicamente
    if epoch % 100 == 0:
        # Imprime as perdas
        print('epoch: {}; G_loss: {:.6f}; D_loss: {:.6f}'.format(epoch + 1, G_loss_curr, D_loss_curr))

        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(G(z_vis)[i, :, :] * 255.0)
            plt.axis('off')
        plt.savefig('dados/image_at_epoch_{:04d}.png'.format(epoch))  # Salva as imagens geradas

#Salvando as Imagens em um unico arquivo Gif animado
def make_gif(frame_folder, output_gif):
    # Carrega os frames da pasta
    frames = [imageio.imread(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]

    # Escreve o GIF
    imageio.mimwrite(output_gif, frames, duration=0.5, loop=0)

    #Deletar Imagens Usadas no Gif
    for epoch in range(epochs):
        filename = 'dados/image_at_epoch_{:04d}.png'.format(epoch)
        if os.path.exists(filename):
            os.remove(filename)

make_gif('./dados', 'dados/dcgan.gif')


#https://medium.com/@marcodelpra/generative-adversarial-networks-dba10e1b4424

