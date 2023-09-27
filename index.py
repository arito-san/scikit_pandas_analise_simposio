# Importação de bibliotecas necessárias
from flask import Flask, Response, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import json

# Criação de uma instância Flask
app = Flask(__name__)

# Carregamento do conjunto de dados do Titanic a partir de um arquivo CSV
df = pd.read_csv("./assets/titanic.csv", sep=',')

# Tratamento de valores ausentes na coluna "Age" com a média das idades
imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])

# Codificação one-hot da coluna "Sex" para transformar valores categóricos em numéricos
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Seleção das características (features) que serão usadas para treinar o modelo
features = ['Pclass', 'Age', 'Fare', 'SibSp', 'Parch', 'Sex_male']

# Definição das características (X) e dos rótulos (y)
X = df[features]  # Características
y = df['Survived']  # Rótulos

# Dividimos os dados em um conjunto de treinamento e um conjunto de teste (80% treinamento, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criação de um modelo de classificação Random Forest
model = RandomForestClassifier(random_state=42)

# Treinamento do modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazemos previsões com o modelo usando os dados de teste
y_pred = model.predict(X_test)

# Calculamos a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

# Criação de um novo passageiro com características específicas para fazer previsões
def new_passenger(data):
    return pd.DataFrame({
        'Pclass': data['Pclass'],        # Classe do passageiro
        'Age': data['Age'],              # Idade do passageiro
        'Fare': data['Fare'],            # Tarifa paga pelo passageiro
        'SibSp': data['SibSp'],          # Número de irmãos/cônjuges a bordo
        'Parch': data['Parch'],          # Número de pais/filhos a bordo
        'Sex_male': [data['Sex_male']]   # Gênero do passageiro codificado (1 para masculino)
})

def probability(data):
# Faz a previsão de probabilidade de sobrevivência para o novo passageiro
    probability = model.predict_proba(new_passenger(data))
# A probabilidade de sobrevivência estará na segunda coluna (índice 1) para a classe "1" (sobrevivência)
    survival_probability = probability[0][1]  # Probabilidade de sobreviver
# Exibe a probabilidade de sobrevivência no terminal
    return print(f"Probabilidade de sobrevivência: {survival_probability * 100:.2f}%")

# Definição de uma rota no Flask que retorna "Hello, World!" como resposta
@app.route('/')
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/newPassenger', methods=['POST'])
def newPassenger():
    body = request.get_json()
    probability(body)
    try:
        return response(201,"newPassenger",body, "Criado com sucesso")
    except Exception as e:
        print ()
        return response(400,"newPassenger",{},"Erro ao cadastrar usuário")
    
# Configuração do response
def response(status,contentName, content, mensagem=False):
    body = {}
    body[contentName] = content
    if(mensagem):
        body["mensagem"]=mensagem
    return Response(json.dumps(body), status=status, mimetype="application/json")

# Configuração do servidor Flask para execução
if __name__ == "__main__":
    with app.app_context():
        app.run(debug=True)