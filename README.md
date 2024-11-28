# IA de Previsão de Doença Cardiaca

## Informações Gerais sobre o Projeto
A IA de Previsão de Doenças Cardíacas desenvolvida em Python é uma aplicação projetada para fornecer informações precisas sobre a probabilidade de ocorrência de doenças cardíacas. Ela permite realizar análises detalhadas com base em dados colocados pelo usuário, ajudando a identificar riscos e promovendo uma melhor prevenção.

## Como Executar o Projeto
Para executar o projeto utilize o VS Code:

### Utilizando o VS Code
1. Clone o repositório do GitHub localmente usando o comando:
git clone https://github.com/LuisH019/projetoInfarto.git
2. Abra o VS Code e selecione a pasta do projeto.
3. Certifique-se de ter a extensão adequada de Python no VS Code, como o "Pandas".
4. Navegue até a classe principal do projeto e execute-a diretamente no VS Code.

## Principais dados utilizados 
- IMC (BMI): É uma medida que usa a altura e o peso para verificar se uma pessoa está em um peso saudável.
- Fumante (Smoking): Indica se a pessoa é fumante ou não.
- Consumo de Alcool (AlcoholDrinking): Refere-se aos hábitos de consumo de bebidas alcoólicas.
- Derrame (Stroke): Indica se a pessoa já sofreu um acidente vascular cerebral (AVC).
- Saúde Física (PhysicalHealth): Avalia o estado geral da saúde física.
- Saúde Mental (MentalHealth): Refere-se ao bem-estar psicológico e emocional da pessoa.
- PhysicalActivity (Atividade Física): Indica o nível de exercícios ou atividades físicas regulares que a pessoa realiza.
- Idade (AgeCategory): Representa a faixa etária da pessoa.
- Sexo (Sex): Refere-se ao sexo biológico da pessoa.
- Diabetes (Diabetic): Indica se a pessoa tem diabetes.
- Horas de Sono (SleepTime): Representa a quantidade média de horas de sono por noite.
- Asma (Asthma): Indica se a pessoa tem asma.
- Doença Renal (KidneyDisease): Refere-se à presença de problemas nos rins.
- Câncer de Pele (SkinCancer): Indica se a pessoa teve ou tem câncer de pele.

## Estrutura do Projeto
- app.py: Arquivo principal que contém o código do aplicativo. Este script inicializa a aplicação e gerencia a lógica central.
- data/: Diretório que armazena dados utilizados pelo aplicativo, como dataset.
- models/: Contém modelos preditivos ou de dados, incluindo scripts e arquivos relacionados ao treinamento e validação de modelos.
- static/images/: Pasta destinada ao armazenamento de imagens estáticas usadas no aplicativo, como logotipos ou gráficos gerados.
- templates/: Diretório que contém templates HTML para renderização das páginas do aplicativo, facilitando a geração dinâmica de conteúdo web.
- utils/: Inclui funções utilitárias e scripts auxiliares que suportam a lógica do aplicativo, como processamento de dados ou configuração.

## Funcionalidades
- Gráficos: Visualizações interativas e de alto impacto criadas com Plotly, oferecendo uma experiência de análise visual.
- Previsões: Analisa instantaneamente os dados de saúde fornecidos pelo usuário, utilizando um modelo preditivo avançado para calcular e exibir uma estimativa personalizada do risco de doença cardíaca.

## Uso do ChatGPT
Durante o desenvolvimento deste projeto, o ChatGPT foi utilizado como uma ferramenta auxiliar para:
- Organização do código e estrutura do projeto.
- Correção de erros que não eram facilmente identificados.
- Esclarecimento de dúvidas sobre a implementação de funcionalidades específicas.
- Verificação se a implementação estava alinhada com os requisitos e especificações do projeto.
