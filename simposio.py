# Importação de bibliotecas necessárias
from flask import Flask, Response, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import json

# configurações do flask/pandas
app = Flask(__name__)
df = pd.read_csv("./assets/data.csv", sep=",",encoding='utf-8')
print("a",df)
# x =

# y =

# Pré processamento de dados para treino e teste.
# X_train, X_test, y_train, y_test = train_test_split(
# x, y, test_size=0.2, random_state=42)
# cols = X_train.columns
# sca = preprocessing.MinMaxScaler()
# X_train = sca.fit_transform(X_train)
# X_train = pd.DataFrame(X_train, columns=cols)
# X_test = sca.transform(X_test)
# X_test = pd.DataFrame(X_test, columns=cols)
# print(X_test)

# Criação e treinamento de um modelo de classificação Random Forest e Decision Tree
# model_random = RandomForestClassifier(random_state=42)
# model_random.fit(X_train, y_train)
# y_pred = model_random.predict(X_test)
# Tree
# model_tree = DecisionTreeClassifier()
# model_tree.fit(X_train, y_train)
# rresult = model_tree.predict(X_test)

# Resultados
# accTree = classification_report(y_test, rresult)
# accRandom = classification_report(y_test, y_pred)
# print('Tree',accTree)
# print('Random',accRandom)


def hello_world():
    return "<p>abcdeefghijklmnopqrstuvwxyz</p>"


if __name__ == "__main__":
    with app.app_context():
        app.run(debug=True)
