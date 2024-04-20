import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

# Lendo o arquivo CSV em um DataFrame
df = pd.read_csv('dados/heart_attack_prediction.csv')

# Função para dividir uma coluna em duas
df[['Blood_Sistolik_Pres','Blood_Diyastolik _Pres']] = df['Blood Pressure'].str.split('/', expand=True)

# Transformar as colunas  em inteiros
df[['Blood_Sistolik_Pres','Blood_Diyastolik _Pres']] = \
    (df[['Blood_Sistolik_Pres','Blood_Diyastolik _Pres']].astype(int))

#Removendo as colunas não úteis para a análise
df = df.drop(['Patient ID'], axis=1)
df = df.drop(['Blood Pressure'], axis=1) #Removendo agora que foi dividida

#Renomeando As colunas para encurtar
df = df.rename(columns={"Sedentary Hours Per Day": "Sedentary_HPD",
                        "Exercise Hours Per Week": "Exercise_HPW",
                        "Previous Heart Problems": "P_Heart Problems",
                        "Physical Activity Days Per Week": "Physical_ADPW",
                        "Sleep Hours Per Day": "Sleep_HPD",
                        "Blood_Sistolik_Pres": "Blood_Sisto",
                        "Blood_Diyastolik _Pres": "Blood_Diya",
                        "Alcohol Consumption": "Alcohol Consum"
                       })

#Exibindo info dos Dados
print("\nInformações do Dados:")
print(df.info())

print("\nExibindo Parte Dados:")
print(df.head(5).to_string())

#Exploratory Data Analysis
print("\nSeparando os Dados em Numericos e Categoricos:\n")

#Encontrando e armazenando as Colunas Numericas
numerical=[]
for col in df.columns:
    if(df[col].dtypes!='object'):
        numerical.append(col)
print(f"Há um total de {len(numerical)} colunas numéricas no dataset")
print(numerical)

#Encontrando e aramazenando as Colunas Categóricas
categorical=[]
for col in df.columns:
    if(df[col].dtypes=='object'):
        categorical.append(col)
print(f"\nHá um total de {len(categorical)} colunas categóricas no dataset")
print(categorical)

#Plotando as categoarias numericas em histograma
df[numerical].hist(bins=15, figsize=(15, 6), layout=(3, 7))
plt.subplots_adjust(wspace=0.5, hspace=0.9)
plt.show()

#Plotando as categoricas em pizza
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
# Iterando sobre as colunas categóricas e plotando um gráfico de pizza para cada uma
for variable, subplot in zip(categorical, ax.flatten()):
    pie = df[variable].value_counts().plot(kind='pie', ax=subplot, autopct='%1.1f%%', labeldistance=1.2)
    subplot.set_title(variable)
    subplot.set_ylabel('')  # Remove o rótulo do eixo y para os gráficos de pizza
    # Ajustando a distância das legendas de porcentagem dentro do gráfico de pizza
# Ajustando o layout
plt.tight_layout()
plt.show()

#Limpeza de Dados
print("\nRealizado a Limpeza de Dados\n")

#1)Removendo valores ausentes
print("1 - Removendo Valores Ausentes")
print('Total de Valores Ausentes: ',df.isnull().sum().sum())
print('Processando -------------------')
#Elimine as linhas com valores ausentes
df = df.dropna()
print('Total de Valores Ausentes: ',df.isnull().sum().sum())

#2)Removendo Valores Duplicado
print("\n2 - Removendo Valores Duplicados")
print('Total de Valores Duplicados: ',df.duplicated().sum())
print('Processando -----------------------')
#Removendo as duplicatas
df = df.drop_duplicates()
print('Total de Valores Duplicados: ',df.duplicated().sum())

#4)Conversao categorico para numerico
print("\n4 - Conversao Variaveis Categoricas Para Numericas")
print('\nProcessando --------------------------------------------------\n')

# Criação de um objeto LabelEncoder
label_encoder = preprocessing.LabelEncoder()

#Percorrendo Todas as colunas da lista categorical gerada anteriormente
# O label_encoder esta transformando categorico para numerico
for var in categorical:
    df[var] = label_encoder.fit_transform(df[var])

print('Colunas Numericas')
print(df[categorical].head(5).to_string())

#5)Removendo os Outliers
print("\n5 - Removendo Outliers")
print('Total de Linhas: ', df.shape[0])
outlier = df.shape[0]

#Usando cálculos de mediana e IQR, os valores discrepantes são
# identificados e esses pontos de dados devem ser removidos
for coluna in df.iloc[:, :4].columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[df[coluna].between(lower_bound, upper_bound)]

print('Total Outliers: ', outlier - df.shape[0],'\n')

# Calculando as correlações em recaçao a um pametro especifico
heart_attack_corr = df.corr()["Heart Attack Risk"]
heart_attack_corr = heart_attack_corr.drop("Heart Attack Risk", axis=0).sort_values(ascending=False)

# Criando o gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x=heart_attack_corr.values, y=heart_attack_corr.index)
plt.xlabel("Correlation")
plt.ylabel("Variable")
plt.title("Correlation with Heart Attack Risk")
plt.subplots_adjust(left=0.3)
plt.show()

#1_Aplicando o Algortimo Do Machine Learning

X = df.drop(['Heart Attack Risk'], axis=1) # Variáveis independentes (features)
y = df['Heart Attack Risk'] # Variável dependente (target)

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20)

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

#_Usando o Algoritmo k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print('Aplicando o Algortimo KNeighborsClassifier')
# Calculando métricas
print(f'Acuracia cros  : {cross_val_score(knn, X_train, y_train, cv=5).mean():.2f}')
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision      : {precision_score(y_test, y_pred, zero_division=1):.2f}")
print(f"Recall         : {recall_score(y_test, y_pred, zero_division=1):.2f}")
print(f"F1 Score       : {f1_score(y_test, y_pred, zero_division=1):.2f}")