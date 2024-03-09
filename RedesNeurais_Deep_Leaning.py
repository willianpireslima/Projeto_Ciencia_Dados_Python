import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

#https://www.analyticsvidhya.com/blog/2021/10/beginners-guide-on-how-to-train-a-classification-model-with-tensorflow/
#https://medium.com/swlh/introduction-to-deep-learning-using-keras-and-tensorflow-part2-284746ab4442

#0_Extraindo os Dados
df = pd.read_csv('dados/winequalityN.csv')

df = df.dropna()
df['is_white_wine'] = [
    1 if typ == 'white' else 0 for typ in df['type']
]

df['is_good_wine'] = [
    1 if quality >= 6 else 0 for quality in df['quality']
]
df.drop('quality', axis=1, inplace=True)
df.drop('type', axis=1, inplace=True)

#Mostrando a coorelaçao entre as variaveis em um heatmap
sns.heatmap(df.corr(),annot=True, vmin=0, vmax=1)
plt.show()

#1_Separando os Dados e os mapeando

# Variáveis independentes (features)
X = df.drop('is_good_wine', axis=1)

# Variável dependente (target)
y = df['is_good_wine']

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20,random_state=42)

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

#4_Modelando a Rede Neural
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'), #Input  Layer (1)
    tf.keras.layers.Dense(256, activation='relu'), #Hidden Layer (2)
    tf.keras.layers.Dense(256, activation='relu'), #Hidden Layer (3)
    tf.keras.layers.Dense(1, activation='sigmoid') #Output Layer (4)
])
model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)
#5_Treinando o modelo com dados de treinamento e validação
history = model.fit(X_train, y_train, epochs=100,validation_data=(X_test, y_test))

#6_Cheacando as Metricas
loss, binary_accuracy, precision, recall = model.evaluate(X_test, y_test)

print('\nMetricas do Modelo')
print(f'Loss            : {loss:.4f}')
print(f'Binary Accuracy : {binary_accuracy:.4f}')
print(f'Precision       : {precision:.4f}')
print(f'Recall          : {recall:.4f}')

#7_Salvando o Modelo
model.save('my_model.keras')  # Salvar modelo

#8_Checando o Overfitting e underfitting
plt.xlabel("Model Complexity - epochs")
plt.ylabel("Error Rate")
plt.title("Loss Curve")
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Test Loss', color='orange')
plt.show()