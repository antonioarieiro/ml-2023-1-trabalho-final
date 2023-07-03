import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

nomes_colunas = ['comprimento da sépala (cm)', 'largura da sépala (cm)', 'comprimento da pétala (cm)',
                 'largura da pétala (cm)', 'espécies']
dados = pd.read_csv("dataset.csv", names=nomes_colunas)

print(dados.espécies.value_counts())
# print(dados.dtypes)

X = dados[['comprimento da sépala (cm)', 'largura da sépala (cm)', 'comprimento da pétala (cm)',
           'largura da pétala (cm)']].values
y = dados['espécies'].values
Xtr, Xtes, ytr, ytes = train_test_split(X, y, test_size=0.4, random_state=0)

classificador = DecisionTreeClassifier(criterion='entropy', max_depth=4)
classificador.fit(Xtr, ytr)

# print(classificador.feature_importances_)

ytr = classificador.predict(Xtes)
acertos = accuracy_score(ytes, ytr)

print('relatório de classificação :\n', classification_report(ytes, ytr))
print('matriz de confusão :\n', confusion_matrix(ytes, ytr))
print("Precisão do classificador: {}%" .format(round(acertos * 100, 2)))

atributos = ['comprimento da sépala', 'largura da sépala', 'comprimento da pétala', 'largura da pétala']
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
tree.plot_tree(classificador, feature_names=atributos, class_names=classificador.classes_, filled=True)

previsao1 = classificador.predict([[5.1, 5.7, 5.3, 0.1]])
previsao2 = classificador.predict([[4.1, 4.7, 4, 1.5]])
previsao3 = classificador.predict([[5.1, 6.7, 6, 1.1]])

print(f"Éspecie prevista para a amostra 1: {previsao1.item()}")
print(f"Éspecie prevista para a amostra 2: {previsao2.item()}")
print(f"Éspecie prevista para a amostra 3: {previsao3.item()}")
