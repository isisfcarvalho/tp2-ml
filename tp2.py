# --------------------------------------------------------- Importações
import pandas as pd
import numpy as np
import math
from copy import deepcopy
from ucimlrepo import fetch_ucirepo
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import plotly.express as px
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
# ------------------------------------------------- Leitura e tratamento de dados
tic_tac_toe_endgame = fetch_ucirepo(id=101) 

X = tic_tac_toe_endgame.data.features 
y = tic_tac_toe_endgame.data.targets 
y = y.replace({'positive': 1, 'negative': -1})

enc = OneHotEncoder()
X = enc.fit_transform(X).toarray()
data_df = pd.DataFrame(X, columns=enc.get_feature_names_out())

# ----------- Implementação da classe BoostingAlgorithm, que contém toda a lógica de treinamento e teste de um modelo Adaboost
class BoostingAlgorithm:
    def __init__(self, X, y, train_index, test_index, num_iters):
        """
        Inicialização de um objeto, com os atributos necessários para o treino e teste do modelo.
        """
        self.X_train = X.iloc[train_index]
        self.y_train = y.iloc[train_index]
        self.X_test = X.iloc[test_index]
        self.y_test = y.iloc[test_index]
        self.num_instances = self.X_train.shape[0]
        self.W = np.array([1/self.num_instances] * self.num_instances)
        self.num_iters = num_iters
        self.alfas = []
        self.stumps = []
        self.erros_treino = []

    def train_model(self):
        """
        Treina o modelo Adaboost e retorna a média dos erros de treino obtidos.
        """
        for it in range(self.num_iters):
            # Fit do stump
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(self.X_train, self.y_train, sample_weight=self.W)
            y_pred = stump.predict(self.X_train)
            self.stumps.append(deepcopy(stump))
            # Cálculo do erro
            erro = self.W[y_pred != self.y_train.values.flatten()].sum()
            self.erros_treino.append(erro)
            # Cálculo do alfa
            alfa = (1/2)*(math.log((1 - erro)/erro))
            self.alfas.append(alfa)
            # Atualização dos pesos
            self.W *= np.exp(-alfa * self.y_train.values.flatten() * y_pred)
            self.W /= np.sum(self.W)
        
    def test_model(self):
        """
        Testa o modelo treinado anteriormente, compondo a saída final como o sinal da soma das respostas dos stumps, ponderados por alfa.
        Retorna a acurácia do modelo sobre os dados de teste.
        """
        y_pred_final = np.zeros(len(self.y_test))
        for stump, alfa in zip(self.stumps, self.alfas):
            y_pred = stump.predict(self.X_test)
            y_pred = y_pred * alfa
            y_pred_final += y_pred

        y_pred_final = np.sign(y_pred_final).astype(int)
        acc = accuracy_score(self.y_test, y_pred_final)   

        return acc
    
# ------------------------------------------------------------ Experimentos

skf = KFold(n_splits=5, shuffle=True, random_state=42)
num_iters = np.linspace(10, 500, 50).astype(int)       # Número de iterações do algoritmo a serem testados
results = []

for n_iter in tqdm(num_iters):
    accs = []
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        boost = BoostingAlgorithm(data_df, y, train_index, test_index, n_iter)
        boost.train_model()
        acc = boost.test_model()
        accs.append(acc)
    mean_acc = np.mean(accs)
    results.append([n_iter, mean_acc])

# ------------------------------------------ Resultados dos experimentos e geração de gráfico
results_df = pd.DataFrame(results, columns=['Número de iterações', 'Acurácia'])
results_df.to_csv('results.csv', index=None)

fig = px.line(results_df, 'Número de iterações', 'Acurácia', width=700, height=400)
fig.update_layout(xaxis = dict(tickmode = 'linear',
                                tick0 = 0,
                                dtick = 50, 
                                showgrid=False
                            ), 
                  title = {
                            'text': "Acurácia de acordo com o número de iterações",
                            'y': 0.92,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'}
)
fig.show()