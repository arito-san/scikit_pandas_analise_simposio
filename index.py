# Importação de bibliotecas necessárias
from flask import Flask, Response, request,render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import json

# Criação de uma instância Flask
db = SQLAlchemy()
app = Flask(__name__, template_folder='static', static_folder='static', static_url_path='/')
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.sqlite3"
# Carregamento do conjunto de dados do Titanic a partir de um arquivo CSV
df = pd.read_csv("./assets/data.csv", sep=';', encoding="ISO-8859-1")

class Student(db.Model):
    id = db.Column( db.Integer, primary_key=True,autoincrement=True, nullable=False)
    name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(100), nullable=True)
    grade = db.Column(db.Float, nullable=True)
    def to_json(self):
        return {
            "id":self.id,
            "name":self.name,
            "phone":self.phone,
            "grade":self.grade
        }
    
colunas_confirmadas = ['dia_semana', 'idade', 'sexo', 'horario', 'fase_dia', 'sentido_via','tipo_pista', 'tracado_via','condicao_metereologica','tipo_veiculo']
df_filtrado = df[colunas_confirmadas]
## Filtrar sexo
df_filtrado['sexo'] = df_filtrado['sexo'].replace(
    {'Masculino': 0, 'Feminino': 1})
## filtrar dias)_da_semana
dias_da_semana = {
    'segunda-feira': 1,
    'terça-feira': 2,
    'quarta-feira': 3,
    'quinta-feira': 4,
    'sexta-feira': 5,
    'sábado': 6,
    'domingo': 7
}

## filtrar dia_semana
df_filtrado['dia_semana'] = df_filtrado['dia_semana'].map(dias_da_semana)
valores_unicos = df_filtrado['dia_semana'].unique()

## filtrar condicoes_meteorologicas
condicoes_meteorologicas = {
    'Céu Claro': 1,
    'Chuva': 2,
    'Garoa/Chuvisco': 3,
    'Ignorado': 4,
    'Nublado': 5,
    'Sol': 6,
    'Vento': 7
}
df_filtrado['condicao_metereologica'] = df_filtrado['condicao_metereologica'].map(
    condicoes_meteorologicas)

## filtrar sentido_via
sentido_via = {
    'Crescente': 1,
    'Decrescente': 2,
    'Não informado': 3
}
df_filtrado['sentido_via'] = df_filtrado['sentido_via'].map(
    sentido_via)
##filtrar tipo_veiculo
tipo_veiculo = {
    'Automóvel': 1,
    'Bicicleta': 2,
    'Caminhão': 3,
    'Caminhão-trator': 4,
    'Caminhonete': 5,
    'Camioneta': 6,
    'Carro de mão': 7,
    'Carroça-charrete': 8,
    'Ciclomoto': 9,
    'Micro-ônibus': 10,
    'Motocicleta': 11,
    'Motoneta': 12,
    'Não Informado': 13,
    'Ônibus': 14,
    'Outros': 15,
    'Reboque': 16,
    'Semireboque': 17,
    'Utilitário': 18,
}
df_filtrado['tipo_veiculo'] = df_filtrado['tipo_veiculo'].map(
    tipo_veiculo)

## filtrar fase_dia
fase_dia = {
    'Amanhecer': 1,
    'Anoitecer': 2,
    'Plena Noite': 3,
    'Pleno dia': 4,
}
df_filtrado['fase_dia'] = df_filtrado['fase_dia'].map(fase_dia)

## filtrar tipo_pista
tipo_pista = {
    'Dupla': 1,
    'Múltipla': 2,
    'Simples': 3,
}
df_filtrado['tipo_pista'] = df_filtrado['tipo_pista'].map(tipo_pista)

## filtrar tipo_pista
tracado_via = {
    'Curva': 1,
    'Desvio Temporário': 2,
    'Interseção de vias': 3,
    'Não Informado': 4,
    'Ponte': 5,
    'Reta': 6,
    'Retorno Regulamentado': 7,
    'Túnel': 8,
    'Viaduto': 9,
}
df_filtrado['tracado_via'] = df_filtrado['tracado_via'].map(tracado_via)
df_filtrado['idade'] = df_filtrado['idade'].round().astype(int)
df_filtrado.loc[:, 'idade'] = df_filtrado['idade'].round().astype(int)
df_filtrado['horario'] = pd.to_datetime(df_filtrado['horario']).dt.hour
df_filtrado = df_filtrado[df_filtrado['sexo'].isin([0, 1])]
df_filtrado['dia_semana'].value_counts()
df_filtrado['sexo'].value_counts()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
faixas_etarias = pd.cut(df_filtrado['idade'], bins)
faixas_etarias.value_counts()


X = df_filtrado[['idade', 'sexo', 'horario', 'fase_dia', 'sentido_via','tipo_pista', 'tracado_via','condicao_metereologica','tipo_veiculo']]
y = df_filtrado['dia_semana']



imputer = SimpleImputer(strategy='most_frequent')
X = imputer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
modelo_arvore = RandomForestClassifier(
    random_state=42, n_estimators=15, criterion="entropy")
modelo_arvore.fit(X_train, y_train)
y_pred = modelo_arvore.predict(X_test)
modelo_arvore.predict_proba(X_test)
precision = precision_score(y_test, y_pred, average='weighted')
acc = accuracy_score(y_test, y_pred)
precisionScore = f'Precisão: {precision:.2f}'
print(precisionScore)
a = classification_report(y_test, y_pred)
print(a)
def driver(data):
    return pd.DataFrame({
        'idade': data['idade'],
        'horario': data['horario'],
        'sentido_via': [data['sentido_via']],
        'tipo_pista': [data['tipo_pista']],
        'tracado_via': [data['tracado_via']],
        'condicao_metereologica': [data['condicao_metereologica']],
        'fase_dia': [data['fase_dia']],
        'sexo': [data['sexo']],
        'tipo_veiculo': [data['tipo_veiculo']],
    }) 

def dia_mais_provavel(data):
    probabilidades = modelo_arvore.predict_proba(driver(data))
    dias_da_semana = ['Segunda', 'Terça', 'Quarta',
                      'Quinta', 'Sexta', 'Sábado', 'Domingo']

    probabilidade_maxima = max(probabilidades[0])
    dia_mais_probavel = dias_da_semana[probabilidades[0].tolist().index(
        probabilidade_maxima)]
    return dia_mais_probavel, probabilidade_maxima

@app.route('/')
def hello_world():
    try:
        return render_template('index.html')
    except Exception as e:
        print(e)
    
@app.route('/newDriver', methods=['POST'])
def newPassenger():
    body = request.get_json()
    dia, prob = dia_mais_provavel(body)
    try:
        return response(201,"dia", dia, prob)
    except Exception as e:
        print ()
        return response(400,"newDriver",{},"Erro ao calcular probabilidade")
    
@app.route('/createStudents', methods=['POST'])
def createStudents():
    body = request.get_json()
    try:
        existing_student = Student.query.filter_by(phone=body['phone']).first()
        if existing_student:
            return response(400, "students", {}, "Número de telefone já cadastrado")
        
        q1_correct = body['q1'] == 4
        q2_correct = body['q2'] == 3
        q3_correct = body['q3'] == 4
        q4_correct = body['q4'] == 3
        q5_correct = body['q5'] == 4
        q6_correct = body['q6'] == 3
        q7_correct = body['q7'] == 4
        q8_correct = body['q8'] == 3
        q9_correct = body['q9'] == 4
        q10_correct = body['q10'] == 2
        points = 0
        if q1_correct:
            points += 1
        if q2_correct:
            points += 1
        if q3_correct:
            points += 1
        if q4_correct:
            points += 1
        if q5_correct:
            points += 1
        if q6_correct:
            points += 1
        if q7_correct:
            points += 1
        if q8_correct:
            points += 1
        if q9_correct:
            points += 1
        if q10_correct:
            points += 1
        
        students = Student(name=body['name'], phone= body['phone'], grade= points)
        db.session.add(students)
        db.session.commit()
        return response(201,"students",students.to_json(), "Criado com sucesso")
    except Exception as e:
        print(e)
        return response(400,"students",{},"Erro ao criar usuário")
    
@app.route('/searchAllStudents', methods=['GET'])
def allStudents():
    try:
        students_obj = Student.query.all()
        students_json = [students.to_json() for students in students_obj]
        return response(201,"students",students_json, "Todos os usuários cadastrado.")
    except Exception as e:
        print (e)
def response(status,contentName, content, message=False):
    body = {}
    body[contentName] = content
    if(message):
        body["message"]=message
    return Response(json.dumps(body), status=status, mimetype="application/json")

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
        db.init_app(app)
        db.create_all()
        app.run(debug=True,host='0.0.0.0', port=5000)
