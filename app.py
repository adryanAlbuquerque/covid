import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# =========================
# CONFIGURAÇÕES INICIAIS
# =========================
st.set_page_config(page_title="Análise COVID-19 - Machine Learning em Saúde", layout="wide")

st.title("Análise de Dados da COVID-19 no Brasil")
st.markdown("""
Este projeto demonstra a aplicação de **técnicas de Machine Learning** na área da saúde,
utilizando dados reais da COVID-19 no Brasil.

A análise inclui:
- **Aprendizado Supervisionado:** modelo de **Regressão Linear** para entender a relação entre casos e óbitos.  
- **Aprendizado Não Supervisionado:** **K-Means** para identificar grupos de estados com padrões semelhantes.  
""")

# =========================
# FUNÇÃO PARA CARREGAR DADOS
# =========================
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"state": "estado", "totalCases": "casos", "deaths": "obitos"})
    df = df[["date", "estado", "casos", "obitos"]]
    df = df[df["estado"] != "TOTAL"]
    return df

dados = carregar_dados()

# =========================
# MENU LATERAL
# =========================
pagina = st.sidebar.radio(
    "Navegação",
    ["Exploração de Dados", "Aprendizado Supervisionado", "Aprendizado Não Supervisionado (K-Means)"]
)

# =========================
# EXPLORAÇÃO DE DADOS
# =========================
if pagina == "Exploração de Dados":
    st.header("Exploração dos Dados (EDA)")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    st.subheader("Casos e Óbitos por Estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("casos", ascending=False), x="estado", y="casos", color="steelblue")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Os estados mais populosos, como São Paulo e Minas Gerais, apresentam o maior número de casos registrados.")

    st.subheader("Letalidade (%) por Estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False), x="estado", y="letalidade", color="gray")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Letalidade (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Estados com letalidade mais alta podem indicar maior impacto da pandemia, menor testagem ou subnotificação de casos leves.")

# =========================
# APRENDIZADO SUPERVISIONADO
# =========================
elif pagina == "Aprendizado Supervisionado":
    st.header("Aprendizado Supervisionado — Regressão Linear")

    st.markdown("""
    Nesta seção, aplicamos **Regressão Linear** para investigar a relação entre o número de casos e o número de óbitos por COVID-19 nos estados brasileiros.
    """)

    df_modelo = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()

    X = df_modelo[["casos"]]
    y = df_modelo["obitos"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("Resultados do Modelo")
    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:,.0f}")
    col3.metric("RMSE", f"{rmse:,.0f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=y_test, y=y_pred, color="steelblue")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
    ax.set_xlabel("Óbitos Reais")
    ax.set_ylabel("Óbitos Preditos")
    ax.set_title("Relação entre Óbitos Reais e Preditos")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:**  
    - O modelo apresenta uma boa correlação entre casos e óbitos, o que é esperado em uma pandemia.  
    - Um valor de **R² moderado** indica que há relação direta, mas outros fatores (idade, infraestrutura de saúde etc.) também influenciam.  
    - O **MAE** e o **RMSE** mostram o erro médio entre os valores reais e os preditos.
    """)

# =========================
# APRENDIZADO NÃO SUPERVISIONADO — K-MEANS
# =========================
elif pagina == "Aprendizado Não Supervisionado (K-Means)":
    st.header("Aprendizado Não Supervisionado — Agrupamento com K-Means")

    st.markdown("""
    O **K-Means** é uma técnica de **aprendizado não supervisionado** que agrupa dados com características semelhantes.  
    Aqui, agrupamos os estados brasileiros de acordo com o número total de casos e óbitos, criando **clusters** com perfis parecidos.
    """)

    dados_cluster = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()

    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados_cluster[["casos", "obitos"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    dados_cluster["cluster"] = kmeans.fit_predict(dados_norm)

    st.subheader("Estados Agrupados por Perfil da Pandemia")
    st.dataframe(dados_cluster.sort_values("cluster"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dados_cluster, x="casos", y="obitos", hue="cluster", palette="viridis", s=100)
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Óbitos Confirmados")
    ax.set_title("Agrupamento de Estados segundo Casos e Óbitos (K-Means)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:**  
    - Cada cor representa um *cluster* (grupo de estados com características semelhantes).  
    - Estados no grupo superior (mais à direita e acima) possuem altos números de casos e óbitos — são os mais impactados.  
    - Estados mais à esquerda representam locais com menor impacto.  
    - O **K-Means** é útil para **descobrir padrões escondidos nos dados**, sem precisar de rótulos prévios.
    """)

# =========================
# RODAPÉ
# =========================
st.markdown("---")
st.caption("Projeto desenvolvido para a disciplina Machine Learning Aplicado à Saúde — SENAC 2025")
