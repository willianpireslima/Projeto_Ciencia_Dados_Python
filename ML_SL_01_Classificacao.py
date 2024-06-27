import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# https://www.kaggle.com/datasets/sudarshan24byte/online-food-dataset

# Lendo o arquivo CSV em um DataFrame
df = pd.read_csv('dados/onlinefoods.csv')

# Exibindo info dos Dados
print("\nInformações do Dados:")
print(df.info())

print("\nExibindo Parte Dados:")
print(df.head(5).to_string())

# Removendo as colunas não úteis para a análise
df = df.drop(['latitude', 'longitude', 'Pin code', 'Unnamed: 12'], axis=1)

# Exploratory Data Analysis
print("\nSeparando os Dados em Numericos e Categoricos:")

# Encontrando e armazenando as Colunas Numericas
numerical = []
for col in df.columns:
    if (df[col].dtypes != 'object'):
        numerical.append(col)
print(f"Há um total de {len(numerical)} colunas numéricas no dataset")
print(numerical)

# Encontrando e aramazenando as Colunas Categóricas
categorical = []
for col in df.columns:
    if (df[col].dtypes == 'object'):
        categorical.append(col)
print(f"\nHá um total de {len(categorical)} colunas categóricas no dataset")
print(categorical)

# Plotando os dados numericos em histograma
df[numerical].hist(bins=15, figsize=(15, 6), layout=(1, 2))
plt.show()

# Plotando os dados categoricos em pizza
fig, ax = plt.subplots(3, 3, figsize=(20, 10))
# Iterando sobre as colunas categóricas e plotando um gráfico de pizza para cada uma
for variable, subplot in zip(categorical, ax.flatten()):
    pie = df[variable].value_counts().plot(kind='pie', ax=subplot, autopct='%1.1f%%', labeldistance=1.2)
    subplot.set_title(variable)
    subplot.set_ylabel('')  # Remove o rótulo do eixo y para os gráficos de pizza
    # Ajustando a distância das legendas de porcentagem dentro do gráfico de pizza
# Ajustando o layout
plt.tight_layout()
plt.show()

# Limpeza de Dados
print("\nRealizado a Limpeza de Dados")

# 1)Removendo valores ausentes
print("1 - Removendo Valores Ausentes")
print('Total de Valores Ausentes Identificados: ', df.isnull().sum().sum())
# Elimine as linhas com valores ausentes
df = df.dropna()

# 2)Removendo Valores Duplicado
print("\n2 - Removendo Valores Duplicados")
print('Total de Total de Valores Duplicados Identificados: ', df.duplicated().sum())
# Removendo as duplicatas
df = df.drop_duplicates()

# 4)Conversao categorico para numerico
print("\n4 - Conversao Variaveis Categoricas Para Numericas\n")

# label_encoder object knows
label_encoder = LabelEncoder()

# Percorrendo Todas as colunas da lista categorical gerada anteriormente
for var in categorical:
    df[var] = label_encoder.fit_transform(df[var]) # O label_encoder esta transformando categorico para numerico

print('Novo Dataframe com somente Colunas Numericas\n')
print(df[categorical].head(5).to_string())

# 5)Removendo os Outliers
print("\n5 - Removendo Outliers")

# Contador de outliers
total_outliers_identificados = 0

# Usando cálculos de mediana e IQR, os valores discrepantes são
# identificados e esses pontos de dados devem ser removidos
for coluna in df.iloc[:, :4].columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Contar outliers na coluna atual
    outliers_coluna = df[~df[coluna].between(lower_bound, upper_bound)]
    total_outliers_identificados += outliers_coluna.shape[0]

    # Remover outliers da coluna atual
    df = df[df[coluna].between(lower_bound, upper_bound)]

print('Total de Outliers identificados: ', total_outliers_identificados)

# Mapa de calor de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Calculando as correlações em recaçao a um pametro especifico
feedback_corr = df.corr()["Feedback"]
feedback_corr = feedback_corr.drop("Feedback", axis=0).sort_values(ascending=False)

# Criando o gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x=feedback_corr.values, y=feedback_corr.index)
plt.xlabel("Correlation")
plt.ylabel("Variable")
plt.title("Correlation with feedback")
plt.subplots_adjust(left=0.3)
plt.show()

# Salvar o DataFrame modificado em um novo arquivo CSV
df.to_csv('dados/onlinefoods_Limpo.csv', index=False, mode='w')

# Realizando o Balanceamento de dados com Oversampling (Sobreamostragem)
print('\nVerificando Balanceamento de Dados\n')
sns.countplot(x=df['Feedback']).set_title("Contagem da Variavel Target")
plt.show()

print('\nContagem da Variavel Target Numericamente\n')
print(df.groupby('Feedback').size())

# Separando os Dados e os mapeando
X = df.drop(['Feedback'], axis=1)  # Variáveis independentes (features)
y = df['Feedback']  # Variável dependente (target)

# Seed para reproduzir o mesmo resultado"
seed = 100
# Cria o balanceador SMOTE"
smote_bal = SMOTE(random_state=seed)
# Aplica o balanceador"
X_over, y_over = smote_bal.fit_resample(X, y)

sns.countplot(x=y_over).set_title("Variavel Target Apos Balaceamento")
plt.show()
# Verificando o balanceamento após SMOTE
print('\nVerificando Balanceamento Após SMOTE\n')
print(pd.Series(y_over).value_counts())

# Separando os dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X_over, y_over, test_size=.20)

print('\nInfomacoes da separacao dos Dados')
print(f'Tamanho Total dos Dados : {X.shape[0]}')
print(f"Tamanho Dados de Treino : {X_train.shape[0]} Por: {X_train.shape[0] / X.shape[0] * 100:.2f}%")
print(f"Tamanho Dados de Teste  : {X_test.shape[0]}  Por: {X_test.shape[0] / X.shape[0] * 100:.2f}%")

print('\nAplicando os Algortimos de Machine Learning Random Forest\n')

# 3_Dimensione os features usando StandardScaler
scaler = StandardScaler()  # criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train)  # Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test)  # Transformar os dados

# Machine Learning
rf = RandomForestClassifier(max_depth=10, n_estimators=408)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Calculando métricas
print(f'Acuracia cros  : {cross_val_score(rf, X_train, y_train, cv=5).mean():.4f}')
print(f"Accuracy       : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision      : {precision_score(y_test, y_pred, zero_division=1):.4f}")
print(f"Recall         : {recall_score(y_test, y_pred, zero_division=1):.4f}")
print(f"F1 Score       : {f1_score(y_test, y_pred, zero_division=1):.4f}\n")

# Iniciando a Verificaao de underfitting e overfitting
y_pred_train = rf.predict(X_train)  # Fazendo previsões nos dados de treinamento
y_pred_test = rf.predict(X_test)  # Fazendo previsões nos dados de teste
# Calculando métricas de treinamento e teste
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Verificando os dados
if train_accuracy < 0.6 and test_accuracy < 0.6:
    print("Modelo está underfitting.")
elif train_accuracy > 0.9 and test_accuracy < 0.7:
    print("Modelo está overfitting.")
else:
    print("Modelo está com bom desempenho.")

# Criando a confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm)
plt.show()

# Salvando O Modelo usando o Algoritmo k-nearest neighbors como exemplo
# Salvar modelo treinado em um arquivo
joblib.dump(rf, 'dados/modelo_rf.pkl')


