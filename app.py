import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este projeto demonstra a aplicação de **Aprendizado de Máquina Supervisionado** em dados reais da **COVID-19 no Brasil**.  
O objetivo é analisar, visualizar e prever o número de óbitos com base na quantidade de casos confirmados.
""")

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

pagina = st.sidebar.radio("Navegação", ["Exploração dos Dados", "Análise por Estado", "Modelagem Supervisionada"])

# =====================================================
# EXPLORAÇÃO DOS DADOS
# =====================================================
if pagina == "Exploração dos Dados":
    st.header("Exploração dos Dados")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{dados_atuais['casos'].sum():,}".replace(",", "."))
    c2.metric("Óbitos Totais", f"{dados_atuais['obitos'].sum():,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{dados_atuais['letalidade'].mean():.2f}")

    st.subheader("Distribuição Geral")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(dados_atuais["casos"], bins=10, color="steelblue", kde=True)
        ax.set_title("Distribuição de Casos por Estado")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(dados_atuais["obitos"], bins=10, color="indianred", kde=True)
        ax.set_title("Distribuição de Óbitos por Estado")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("Esses histogramas mostram a variação entre os estados, com forte concentração de casos e óbitos nos estados mais populosos.")

    st.subheader("Correlação entre Casos e Óbitos")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=dados_atuais, x="casos", y="obitos", color="darkblue")
    ax.set_title("Relação entre Casos e Óbitos por Estado")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("A correlação é forte e positiva — quanto mais casos, maior o número de óbitos.")

    st.subheader("Letalidade (%) por Estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False),
                x="estado", y="letalidade", color="gray")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Letalidade (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Estados com maior letalidade podem ter menor testagem ou infraestrutura hospitalar.")

    st.subheader("Mapa de Correlação")
    corr = dados_atuais[["casos", "obitos", "letalidade"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    ax.set_title("Correlação entre Variáveis")
    st.pyplot(fig)
    plt.close(fig)

# =====================================================
# ANÁLISE INDIVIDUAL
# =====================================================
elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")
    estados = sorted(dados["estado"].unique())
    estado_sel = st.selectbox("Selecione o estado:", estados)
    df_estado = dados[dados["estado"] == estado_sel]

    st.subheader(f"Evolução da COVID-19 em {estado_sel}")
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

    st.markdown("Este gráfico mostra o crescimento acumulado de casos e óbitos ao longo do tempo.")

# =====================================================
# MODELAGEM SUPERVISIONADA
# =====================================================
elif pagina == "Modelagem Supervisionada":
    st.header("Predição de Óbitos com Regressão Linear")

    st.markdown("Aplicação de um modelo de **Regressão Linear** para prever o número de óbitos com base no total de casos confirmados.")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    X = dados_atuais[["casos"]]
    y = dados_atuais["obitos"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R²", f"{r2:.2f}")
    col2.metric("MAE", f"{mae:.0f}")
    col3.metric("RMSE", f"{rmse:.0f}")

    st.subheader("Desempenho do Modelo")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={"s": 70, "color": "steelblue"}, line_kws={"color": "red"})
    ax.set_xlabel("Óbitos Reais")
    ax.set_ylabel("Óbitos Preditos")
    ax.set_title("Predição com Regressão Linear")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:**  
    - O **R²** mostra quanto da variação dos óbitos é explicada pelos casos confirmados.  
    - O **MAE** e o **RMSE** indicam o erro médio das previsões.  
    - O modelo é simples e demonstra bem o conceito de aprendizado supervisionado.
    """)

# =====================================================
# RODAPÉ
# =====================================================
st.markdown("---")
st.caption("Projeto SENAC — Machine Learning Aplicado à Saúde | 2025")
