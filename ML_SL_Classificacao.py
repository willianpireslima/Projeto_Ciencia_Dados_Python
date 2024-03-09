#https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
#https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
#https://www.datacamp.com/tutorial/decision-tree-classification-python
#https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#https://www.datacamp.com/tutorial/understanding-logistic-regression-python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

#0_Extraindo os Dados
df = pd.read_csv('dados/diabetes.csv')

#1_Separando os Dados e os mapeando

# Variáveis independentes (features)
X = df[['Gravidez','Glicose','PressaoArterial','EspessuraDaPele','Insulina','IMC','DiabetesPedigree','Idade']]

# Variável dependente (target)
y = df['Resultado']

#Mostrando a coorelaçao entre as variaveis em um heatmap
sns.heatmap(df.corr(),annot=True, vmin=0, vmax=1)
plt.show()

#2_Separando od dados em conjuntos de treinamento e teste
(X_train, X_test, y_train, y_test) = train_test_split(X,y, test_size = .20)

print('Infomacoes das Dados')
print(f'Tamanho Total dos Dados : {X.shape[0]}')
print(f"Tamanho Dados de Treino : {X_train.shape[0]} Por: {X_train.shape[0]/X.shape[0]*100:.2f}%")
print(f"Tamanho Dados de Teste  : {X_test.shape[0]}  Por: {X_test.shape[0]/X.shape[0]*100:.2f}%")

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

#4_Medtricas para a Avaliacao do Modelo de Classificacao
def metric (y_test, y_pred,modelo, X_train, y_train):
    print(f'Acuracia cros  : {cross_val_score(modelo, X_train, y_train, cv=5).mean():.2f}')
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision      : {precision_score(y_test, y_pred):.2f}")
    print(f"Recall         : {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score       : {f1_score(y_test, y_pred):.2f}\n")


#4_Usando o Algoritmo LinearRegression
ln = linear_model.RidgeClassifier() # Criando uma instância  regressao linear
ln.fit(X_train, y_train) # Treinando o SVM
y_pred = ln.predict(X_test) # Fazendo previsões nos dados de teste
print('\nRegressao linear')
metric (y_test, y_pred,ln, X_train, y_train)

#5_Usando o Algoritmo Naive Bayes Gaussiano
nbg = GaussianNB()  #Criando o modelo Naive Bayes Gaussiano
nbg.fit(X_train, y_train) #Treinando o modelo
y_pred = nbg.predict(X_test) #Fazendo previsões
print(f'Gaussian Naive Bayes')#Calculando a acurácia
metric (y_test, y_pred,nbg, X_train, y_train)

#6_Usando o Algoritmo Logistic Regression
logreg = LogisticRegression(random_state=16) # instanciando o modelo
logreg.fit(X_train, y_train)# preenchendo o modelo com dados
y_pred = logreg.predict(X_test)
print(f'Regressao logistica')
metric (y_test, y_pred,logreg, X_train, y_train)

#7_Usando o Algoritmo k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"k-nearest neighbors")
metric (y_test, y_pred,knn, X_train, y_train)

#8_Usando o Algoritmo Decision tree
dt = DecisionTreeClassifier() # Cria o objeto classificador da Árvore de Decisão
dt = dt.fit(X_train,y_train) # Treinar classificador de árvore de decisão
y_pred = dt.predict(X_test) #Prever a resposta do conjunto de dados de teste
print(f'Arvore de decisao')
metric (y_test, y_pred,dt, X_train, y_train)

#9_Usando o Algoritmo Support Vector Machine
clf = svm.SVC(kernel='linear') # Criando uma instância do SVM de classificacao SVC
clf.fit(X_train, y_train) # Treinando o SVM
y_pred = clf.predict(X_test) # Fazendo previsões nos dados de teste
print(f'SVM')
metric (y_test, y_pred,clf, X_train, y_train)

#10_Usando o Algoritmo Random forests
rf = RandomForestClassifier(n_estimators=10) # Criando uma instância do SVM de classificacao SVC
rf.fit(X_train, y_train) # Treinando o SVM
y_pred = rf.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Random forests')
metric (y_test, y_pred,rf, X_train, y_train)

#11_Usando o Algoritmo GradientBoostingRegressor
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0) # Criando uma instância do SVM de classificacao SVC
gb.fit(X_train, y_train) # Treinando o SVM
y_pred = gb.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Gradient Boosting')
metric (y_test, y_pred,gb, X_train, y_train)

#12_Usando o Voting
vt = VotingClassifier(estimators=[('gb', gb), ('rf', rf), ('lr', ln)]) # Criando uma instância do SVM de classificacao SVC
vt.fit(X_train, y_train) # Treinando o SVM
y_pred = vt.predict(X_test) # Fazendo previsões nos dados de teste
print(f'Voting')
metric (y_test, y_pred,vt, X_train, y_train)

#10_Salvar modelo treinado em um arquivo
joblib.dump(knn, 'modelo_knn.pkl')

#11_Simulado Uma nova entrada de dados com o modeloo ja treinando
df2 = pd.read_csv('dados/diabetes.csv')

X_new =df2[['Gravidez','Glicose','PressaoArterial','EspessuraDaPele','Insulina','IMC','DiabetesPedigree','Idade']]
# Pré-processamento dos novos dados
X_new_scaled = scaler.transform(X_new)  # Aplicar a mesma transformação que você aplicou aos dados de treinamento

#Fazendo previsões com o modelo treinado
y_new_pred = knn.predict(X_new_scaled)

#Aqui as previsão e comparada com os dados reais
print('\nInserido um novo Conjunto de Dados')
print(f'Porcentagem de Acerto : {(y_new_pred==df2['Resultado']).sum()/df2['Resultado'].shape[0]*100:.2f}%')

