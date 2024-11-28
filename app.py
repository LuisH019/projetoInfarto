from flask import Flask, render_template, request
from predict import predict
import pandas as pd
from graphics import listGraphics

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def formulario():
    probabilidade = None
    predicao = None

    if request.method == "POST":
        # Captura os valores do formulário
        dados_usuario = pd.DataFrame({
            'BMI': [float(request.form.get("BMI"))],
            'Smoking': [float(request.form.get("Smoking"))],
            'AlcoholDrinking': [float(request.form.get("AlcoholDrinking"))],
            'Stroke': [float(request.form.get("Stroke"))],
            'PhysicalHealth': [float(request.form.get("PhysicalHealth"))],
            'MentalHealth': [float(request.form.get("MentalHealth"))],
            'Sex': [float(request.form.get("Sex"))],
            'AgeCategory': [float(request.form.get("AgeCategory"))],
            'Diabetic': [float(request.form.get("Diabetic"))],
            'PhysicalActivity': [float(request.form.get("PhysicalActivity"))],
            'SleepTime': [float(request.form.get("SleepTime"))],
            'Asthma': [float(request.form.get("Asthma"))],
            'KidneyDisease': [float(request.form.get("KidneyDisease"))],
            'SkinCancer': [float(request.form.get("SkinCancer"))]
        })

        # Realizar a predição usando os dados inseridos
        predicao, probabilidade = predict(dados_usuario)

        return render_template("formulario.html", probabilidade=probabilidade, predicao=predicao)

    return render_template("formulario.html", probabilidade=probabilidade, predicao=predicao)

@app.route("/graphics")
def graphics():
    return render_template("graphics.html", listGraphics=listGraphics())

if __name__ == "__main__":
    app.run(debug=True)
