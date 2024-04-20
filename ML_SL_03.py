#https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
#https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
#https://www.datacamp.com/tutorial/decision-tree-classification-python
#https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#https://www.datacamp.com/tutorial/understanding-logistic-regression-python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import joblib
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
print(f"Tamanho Dados de Teste  : {X_test.shape[0]}  Por: {X_test.shape[0]/X.shape[0]*100:.2f}%\n")

#3_Dimensione os features usando StandardScaler
scaler = StandardScaler() #criando um objeto do StandardScaler()
X_train = scaler.fit_transform(X_train) #Transformar os dados e padroniza-los.
X_test = scaler.transform(X_test) #Transformar os dados

# Definindo uma lista de modelos
model_list = [
    RidgeClassifier(),
    GaussianNB(),
    LogisticRegression(random_state=16),
    KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean'),
    DecisionTreeClassifier(),
    SVC(kernel='linear'),
    RandomForestClassifier(n_estimators=10),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
    VotingClassifier(estimators=[('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)),
                                 ('rf', RandomForestClassifier(n_estimators=10)),
                                 ('lr', LogisticRegression(random_state=16))])
]

# Loop sobre os modelos
for model in model_list:
    model.fit(X_train, y_train)  # Treinando o modelo
    y_pred = model.predict(X_test)  # Fazendo previsões nos dados de teste

    # Imprimindo o nome do modelo
    print(f'{type(model).__name__}')

    # Calculando métricas
    print(f'Acuracia cros  : {cross_val_score(model, X_train, y_train, cv=5).mean():.2f}')
    print(f"Accuracy       : {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision      : {precision_score(y_test, y_pred,zero_division=1):.2f}")
    print(f"Recall         : {recall_score(y_test, y_pred,zero_division=1):.2f}")
    print(f"F1 Score       : {f1_score(y_test, y_pred,zero_division=1):.2f}\n")

#7_Usando o Algoritmo k-nearest neighbors
knn = KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#10_Salvar modelo treinado em um arquivo
joblib.dump(knn, 'modelo_knn.pkl')



