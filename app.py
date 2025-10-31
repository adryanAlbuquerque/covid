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

# Configurações iniciais
st.set_page_config(page_title="Análise COVID-19 - Machine Learning em Saúde", layout="wide")
st.title("Análise de Dados da COVID-19 no Brasil")
st.markdown("""
Este projeto demonstra a aplicação de técnicas de Machine Learning na área da saúde,
usando dados reais da COVID-19 no Brasil.  
Inclui visão geral, análise por estado, aprendizado supervisionado e não supervisionado.
""")

# Carregar dados
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

# Menu lateral
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Exploração de Dados", "Análise Individual por Estado", 
     "Aprendizado Supervisionado", "Aprendizado Não Supervisionado (K-Means)"]
)

# =========================
# VISÃO GERAL DO BRASIL
# =========================
if pagina == "Visão Geral":
    st.header("Visão Geral do Brasil")
    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{dados_atuais['casos'].sum():,}".replace(",", "."))
    c2.metric("Óbitos Totais", f"{dados_atuais['obitos'].sum():,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{dados_atuais['letalidade'].mean():.2f}")

    st.markdown("**Interpretação:** Essa visão geral mostra o panorama da COVID-19 no Brasil, destacando os estados mais impactados e a letalidade média.")

# =========================
# EXPLORAÇÃO DE DADOS
# =========================
elif pagina == "Exploração de Dados":
    st.header("Exploração dos Dados (EDA)")
    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    st.subheader("Casos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("casos", ascending=False), x="estado", y="casos", color="steelblue")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Estados mais populosos, como São Paulo, Minas Gerais e Paraná, apresentam o maior número de casos confirmados.")

    st.subheader("Óbitos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("obitos", ascending=False), x="estado", y="obitos", color="indianred")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Óbitos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** O padrão de óbitos acompanha os casos, evidenciando maior mortalidade nos estados mais impactados.")

    st.subheader("Taxa de letalidade (%) por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False), x="estado", y="letalidade", color="gray")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Letalidade (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Letalidades mais altas podem indicar menor testagem ou sistemas de saúde sobrecarregados.")

# =========================
# ANÁLISE INDIVIDUAL POR ESTADO
# =========================
elif pagina == "Análise Individual por Estado":
    st.header("Análise Individual por Estado")
    estados = sorted(dados["estado"].unique())
    estado_sel = st.selectbox("Selecione o estado:", estados)
    df_estado = dados[dados["estado"] == estado_sel]

    st.subheader(f"Evolução dos casos e óbitos em {estado_sel}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_estado["date"], df_estado["casos"], label="Casos", color="steelblue")
    ax.plot(df_estado["date"], df_estado["obitos"], label="Óbitos", color="indianred")
    ax.legend()
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade")
    st.pyplot(fig)
    plt.close(fig)

    taxa_crescimento = df_estado["casos"].pct_change().mean() * 100
    st.metric("Crescimento médio diário de casos (%)", f"{taxa_crescimento:.2f}")

    st.markdown("**Interpretação:** Esse gráfico mostra a evolução da COVID-19 no estado selecionado e a taxa média de crescimento diário de casos.")

# =========================
# APRENDIZADO SUPERVISIONADO
# =========================
elif pagina == "Aprendizado Supervisionado":
    st.header("Aprendizado Supervisionado — Regressão Linear")
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

    st.markdown("**Interpretação:** O modelo estima óbitos a partir de casos confirmados. Apesar de simples, revela a tendência geral da pandemia.")

# =========================
# APRENDIZADO NÃO SUPERVISIONADO — K-MEANS
# =========================
elif pagina == "Aprendizado Não Supervisionado (K-Means)":
    st.header("Aprendizado Não Supervisionado — K-Means")
    dados_cluster = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()

    k = st.slider("Selecione a quantidade de clusters:", min_value=2, max_value=6, value=3)

    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados_cluster[["casos", "obitos"]])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    dados_cluster["cluster"] = kmeans.fit_predict(dados_norm)

    st.subheader("Estados agrupados por perfil da pandemia")
    st.dataframe(dados_cluster.sort_values("cluster"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=dados_cluster, x="casos", y="obitos", hue="cluster", palette="viridis", s=100)
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Óbitos Confirmados")
    ax.set_title(f"Agrupamento de Estados segundo Casos e Óbitos (K={k})")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:** Cada cor representa um grupo de estados com comportamento semelhante.  
    Clusters ajudam a identificar padrões ocultos e regiões com níveis parecidos de impacto da pandemia.
    """)

# Rodapé
st.markdown("---")
st.caption("Projeto desenvolvido para Machine Learning Aplicado à Saúde — SENAC 2025")
