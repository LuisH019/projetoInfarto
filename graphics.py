import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv('heart_2020_cleaned.csv')
df = df.drop(['Race', 'DiffWalking', 'GenHealth'], axis=1)

dir = 'static/images/'

def generateGraphics():
    plt.style.use('dark_background')
    plt.rcParams['axes.facecolor'] = '#212529'
    plt.rcParams['axes.edgecolor'] = '#adb5bd'
    plt.rcParams['axes.labelcolor'] = '#adb5bd'
    plt.rcParams['xtick.color'] = '#adb5bd'
    plt.rcParams['ytick.color'] = '#adb5bd'
    plt.rcParams['text.color'] = '#adb5bd'
    plt.rcParams['figure.facecolor'] = '#212529'
    plt.rcParams['savefig.facecolor'] = '#212529'

    df['HeartDisease'] = df['HeartDisease'].map({1: 'Sim', -1: 'Não'})
    df['HeartDisease'].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
    plt.title('Frequência de Doença Cardíaca')
    plt.xlabel('Condição')
    plt.ylabel('Frequência')
    plt.savefig(dir + 'plot01.png')
    plt.close()

    df['BMI'].plot(kind='hist', bins=20, color='lightblue', edgecolor='black')
    plt.title('Distribuição de IMC')
    plt.xlabel('IMC')
    plt.ylabel('Frequência')
    plt.savefig(dir + 'plot02.png')
    plt.close()

    df.boxplot(column='SleepTime', by='HeartDisease', grid=False)
    plt.title('Tempo de Sono por Doença Cardíaca')
    plt.suptitle('')  # Remove título automático
    plt.xlabel('Doença Cardíaca')
    plt.ylabel('Horas de Sono')
    plt.savefig(dir + 'plot03.png')
    plt.close()

    df['Smoking'] = df['Smoking'].map({1: 'Sim', -1: 'Não'})
    pd.crosstab(df['Smoking'], df['HeartDisease']).plot(kind='bar', stacked=True, color=['lightcoral', 'lightgreen'])
    plt.title('Distribuição de Fumantes por Doença Cardíaca')
    plt.xlabel('Fumante')
    plt.ylabel('Frequência')
    plt.legend(['Sem Doença', 'Com Doença'])
    plt.savefig(dir + 'plot05.png')
    plt.close()

    df.groupby('AgeCategory')['HeartDisease'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, color=['lightcoral', 'lightblue'])
    plt.title('Distribuição de Doença Cardíaca por Idade')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Proporção')
    plt.legend(['Sem Doença', 'Com Doença'])
    plt.savefig(dir + 'plot06.png')
    plt.close()

    numeric_columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
    correlation = df[numeric_columns].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Mapa de Correlação')
    plt.savefig(dir + 'plot07.png')
    plt.close()

    df['PhysicalActivity'] = df['PhysicalActivity'].map({1: 'Sim', -1: 'Não'})
    pd.crosstab(df['PhysicalActivity'], df['HeartDisease']).plot(kind='bar', color=['lightgreen', 'lightcoral'])
    plt.title('Atividade Física vs Doença Cardíaca')
    plt.xlabel('Pratica Atividade Física')
    plt.ylabel('Frequência')
    plt.legend(['Sem Doença', 'Com Doença'])
    plt.xticks(rotation=0)
    plt.savefig(dir + 'plot09.png')
    plt.close()

    sns.boxplot(data=df, x='HeartDisease', y='SleepTime', palette='Set2')
    plt.title('Tempo de Sono vs Doença Cardíaca')
    plt.xlabel('Doença Cardíaca')
    plt.ylabel('Horas de Sono')
    plt.savefig(dir + 'plot10.png')
    plt.close()

    age_diabetes = pd.crosstab(df['AgeCategory'], [df['Diabetic'], df['HeartDisease']])
    age_diabetes.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightcoral', 'lightblue', 'orange', 'green'])
    plt.title('Distribuição por Idade: Diabetes e Doença Cardíaca')
    plt.xlabel('Faixa Etária')
    plt.ylabel('Frequência')
    plt.legend(['Não Diabético - Sem Doença', 'Não Diabético - Com Doença', 
                'Diabético - Sem Doença', 'Diabético - Com Doença'], bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dir + 'plot11.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(df['BMI'], df['MentalHealth'], alpha=0.6, color='purple')
    plt.title('Relação entre IMC e Saúde Mental')
    plt.xlabel('IMC')
    plt.ylabel('Dias com Problemas Mentais')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(dir + 'plot12.png')
    plt.close()


def listGraphics():
    return [dir + f for f in os.listdir(dir)]

generateGraphics()