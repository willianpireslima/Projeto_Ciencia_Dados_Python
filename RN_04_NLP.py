import pandas as pd
import random
import matplotlib.pyplot as plt
#para o pre-processamento do texto
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras import layers

#0_Carregamento de dados e Data Exploration
train_df = pd.read_csv("dados/disaster_tweet_train.csv")

#Exibindo info dos Dados
print("\nInformações do Dados:")
print(train_df.info())

print("\nExibindo Parte Dados:")
print(train_df.head(5).to_string())

print("\nExibindo Amostra da Classe Target:")
print(train_df.target.value_counts())
print(f"\nQuantidade de Amostras: {len(train_df)}")

# WORD-COUNT
train_df['word_count'] = train_df['text'].apply(lambda x: len(str(x).split()))
print(f'Media de Palavras em Disaster tweets     : {train_df[train_df['target']==1]['word_count'].mean():.2f}')
print(f'Media de Palavras em Non-Disaster tweets : {train_df[train_df['target']==0]['word_count'].mean():.2f}')
# CHARACTER-COUNT
train_df['char_count'] = train_df['text'].apply(lambda x: len(str(x)))
print(f'Media de Caracteres Disaster tweets      : {train_df[train_df['target']==1]['char_count'].mean():.2f}')
print(f'edia de Caracteres Non-Disaster tweets   : {train_df[train_df['target']==0]['char_count'].mean():.2f}')

#Visualizando alguns exemplos de treinamento aleatórios
print('\nExibindo Amostras Aleatorias de Dados:')
random_index = random.randint(0, len(train_df)-5) # create random indexes not higher than the total number of samples
for row in train_df[["text", "target"]][random_index:random_index+5].itertuples():
  _, text, target = row
  print(f"Target: {target}", "(real disaster)" if target > 0 else "(not real disaster)")
  print(f"Text:\n{text}\n")
  print("---\n")

#PLOTTING WORD-COUNT
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
train_words=train_df[train_df['target']==1]['word_count']
ax1.hist(train_words,color='red')
ax1.set_title('Disaster tweets')
train_words=train_df[train_df['target']==0]['word_count']
ax2.hist(train_words,color='green')
ax2.set_title('Non-disaster tweets')
fig.suptitle('Words per tweet')
plt.show()

#Removendo as colunas não úteis para a análise
train_df = train_df.drop(['id'], axis=1)
train_df = train_df.drop(['keyword'], axis=1)
train_df = train_df.drop(['location'], axis=1)

#1_Processamento do Texto
def preprocess(text):#converter para minúsculas, retirar e remover pontuações
    text = text.lower()
    text = text.strip()
    text = re.compile('<.*?>').sub('', text)
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text
def stopword(string):#Remocao de StopWord
    a = [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

#LEMMATIZATION
wl = WordNetLemmatizer() #Inicializando o  lemmatizer

def get_wordnet_pos(tag): ##Esta é uma função auxiliar para mapear tags de posição NTLK
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string))  # Get position tags
    a = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in
         enumerate(word_pos_tags)]  # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
train_df['clean_text'] = train_df['text'].apply(lambda x: finalpreprocess(x))
print(train_df.head(5).to_string())

#2_Define o vetorizador de texto
text_vectorizer = TextVectorization(
    max_tokens=6515,  #Define o número máximo de tokens no vocabulário
    output_mode="int", #Define o modo de saída como números inteiros
    output_sequence_length=15 # Define o comprimento máximo das sequências de saída
)

text_vectorizer.adapt(train_df["clean_text"]) #Ajusta o vetorizador de texto ao texto de treinamento

#Obtenha o índice da palavra do tokenizer do text tokenizer
word_index = text_vectorizer.get_vocabulary()
word_index_dict = {word: index for index, word in enumerate(word_index)}

print("Exibindo 10 Primeiras palavras do Word Index:")
partial_word_index = dict(list(word_index_dict.items())[:10])
print(partial_word_index)

#Crie uma amostra de sentença e o tokenize
sample_sentence = "There's a flood in my street!"
print(f'Amostra             :{sample_sentence}')
print(f'Tokenize da Amostra : {text_vectorizer([sample_sentence])}')

#Obtenha as palavras únicas no vocabulário
top_5_words = word_index[:5] # most common tokens (notice the [UNK] token for "unknown" words)
bottom_5_words = word_index[-5:] # least common tokens
print(f"\nNúmero de palavras no vocabulário : {len(word_index)}")
print(f"As 5 palavras mais comuns         : {top_5_words}")
print(f"As 5 palavras menos comuns        : {bottom_5_words}")

#3_Aplicando o embedding
tf.random.set_seed(42)
embedding = layers.Embedding(input_dim=1000, # set input shape
                             output_dim=128, # set size of embedding vector
                             embeddings_initializer="uniform", # default, intialize randomly
                             name="embedding_1")

#Obtenha uma sentença aleatória do conjunto de treinamento
random_sentence = random.choice(train_df["clean_text"])
print(f"\nTexto Original: {random_sentence}\
      \n\nVersion Embedded:")

#Embed  a sentensa aleatoria  (transforme-a em representação numérica)
sample_embed = embedding(text_vectorizer([random_sentence]))
print(sample_embed)

#4_Realizando o Deep Learning

#Dividindo os dados em conjuntos de treinamento e teste
X = train_df["clean_text"].to_numpy()
y = train_df["target"].to_numpy()

#Separando os dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.20, random_state=42)

#Definindo o modelo Sequencial
model = tf.keras.Sequential([
    #Camada de entrada (Input layer): a entrada é uma string
    tf.keras.layers.Input(shape=(1,), dtype="string"),
    text_vectorizer,#Text vectorization layer (transforms words into numbers)
    embedding,#Embedding layer
    tf.keras.layers.GlobalAveragePooling1D(),#Camada de pool para reduzir a dimensionalidade da embedding
    #Dense layer for output
    tf.keras.layers.Dense(1, activation="sigmoid")  # Using sigmoid for binary classification
], name="model_dense")

#Compilando o modelo
model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

#Treinando o modelo
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Avaliação do modelo nos dados de teste
loss, accuracy = model.evaluate(X_test, y_test)

print('\nMetricas')
print(f'Accuracy : {accuracy:.2f}')
print(f'Perda    : {loss:.2f}')

#8_Checando o Overfitting e underfitting
plt.xlabel("Model Complexity - epochs")
plt.ylabel("Error Rate")
plt.title("Loss Curve")
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Test Loss', color='orange')
plt.show()

#References
#https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/
#https://medium.com/analytics-vidhya/nlp-tutorial-for-text-classification-in-python-8f19cd17b49e
#https://jingchaozhang.github.io/Natural-Language-Processing-in-TensorFlow/
#https://github.com/Krishnarohith10/nlp-getting-started
#https://medium.com/geekculture/nlp-with-tensorflow-keras-explanation-and-tutorial-cae3554b1290