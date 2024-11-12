from flask import Flask, render_template, request
import numpy as np
from predict import predict

app = Flask(__name__)

# Função para carregar os pesos
def load_weights():
    N = 30  # Quantidade de neurônios na camada escondida
    m = 14  # Quantidade de features no dataset após o drop
    W1 = np.random.rand(N, m + 1)  # Pesos da primeira camada (30 neurônios, 15 entradas incluindo o viés)
    W2 = np.random.rand(1, N + 1)  # Pesos da segunda camada (1 saída, 31 entradas incluindo o viés)
    return W1, W2

# Inicializar os pesos
W1, W2 = load_weights()

@app.route("/", methods=["GET", "POST"])
def formulario():
    probabilidade = None
    predicao = None

    if request.method == "POST":
        # Captura os valores do formulário
        dados_usuario = [
            float(request.form.get("BMI")),
            float(request.form.get("Smoking")),
            float(request.form.get("AlcoholDrinking")),
            float(request.form.get("Stroke")),
            float(request.form.get("PhysicalHealth")),
            float(request.form.get("MentalHealth")),
            float(request.form.get("PhysicalActivity")),
            float(request.form.get("AgeCategory")),
            float(request.form.get("Sex")),
            float(request.form.get("Diabetic")),
            float(request.form.get("SleepTime")),
            float(request.form.get("Asthma")),
            float(request.form.get("KidneyDisease")),
            float(request.form.get("SkinCancer"))
        ]

        X_novo = np.array(dados_usuario)  # X_novo agora tem 14 elementos

        # Realizar a predição usando os dados inseridos
        probabilidade = predict(X_novo, W1, W2)
        predicao = "Doença cardíaca" if probabilidade >= 0.5 else "Sem doença cardíaca"

        return render_template("formulario.html", probabilidade=probabilidade, predicao=predicao)

    return render_template("formulario.html", probabilidade=probabilidade, predicao=predicao)

if __name__ == "__main__":
    app.run(debug=True)


# teremos que armazenar os pesos após o treinamento para realizar a predição.
# usar p np.load para carregar os pesos ao usar valores que o usuário inseriu
