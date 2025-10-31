import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =======================
# CONFIGURAÇÕES GERAIS
# =======================
st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")

st.title("Análise de COVID-19 no Brasil")
st.markdown("""
Este projeto foi desenvolvido como parte da disciplina **Machine Learning Aplicado à Saúde**.  
O objetivo é explorar dados reais da **COVID-19** no Brasil, realizando análises descritivas e aplicando uma técnica de **aprendizado não supervisionado (K-Means)** para identificar padrões entre os estados.
""")

# =======================
# FUNÇÃO DE CARREGAMENTO
# =======================
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

# =======================
# MENU LATERAL
# =======================
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Análise por Estado", "Aprendizado Não Supervisionado (K-Means)"]
)

# =======================
# VISÃO GERAL
# =======================
if pagina == "Visão Geral":
    st.header("Visão Geral da COVID-19 no Brasil")

    dados_atuais = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    total_casos = int(dados_atuais["casos"].sum())
    total_obitos = int(dados_atuais["obitos"].sum())
    letalidade_media = dados_atuais["letalidade"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais", f"{total_casos:,}".replace(",", "."))
    c2.metric("Óbitos Totais", f"{total_obitos:,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{letalidade_media:.2f}")

    st.subheader("Casos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("casos", ascending=False), x="estado", y="casos", color="steelblue")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Casos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Os estados com maior número de casos são geralmente os mais populosos, como São Paulo, Minas Gerais e Paraná.")

    st.subheader("Óbitos confirmados por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("obitos", ascending=False), x="estado", y="obitos", color="indianred")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Óbitos Confirmados")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** O padrão de óbitos acompanha os casos confirmados, mostrando maior impacto nos estados mais densamente povoados.")

    st.subheader("Taxa de letalidade (%) por estado")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False), x="estado", y="letalidade", color="gray")
    ax.set_xlabel("Estado")
    ax.set_ylabel("Letalidade (%)")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("**Interpretação:** Estados com letalidade mais alta podem indicar subnotificação de casos leves ou dificuldades no sistema de saúde.")

# =======================
# ANÁLISE POR ESTADO
# =======================
elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")

    estados = sorted(dados["estado"].unique())
    estado_sel = st.selectbox("Selecione o estado:", estados)
    df_estado = dados[dados["estado"] == estado_sel]

    st.subheader(f"Evolução de casos e óbitos em {estado_sel}")
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
    st.markdown("**Interpretação:** Este gráfico mostra a progressão temporal da COVID-19 no estado selecionado. A taxa de crescimento indica o ritmo médio de aumento de casos ao longo do período.")

# =======================
# K-MEANS (CLUSTERS)
# =======================
elif pagina == "Aprendizado Não Supervisionado (K-Means)":
    st.header("Aprendizado Não Supervisionado — Agrupamento com K-Means")

    st.markdown("""
    Nesta seção, aplicamos o algoritmo **K-Means** para agrupar os estados brasileiros de acordo com o número total de **casos** e **óbitos**.  
    O objetivo é identificar grupos de estados com padrões semelhantes de impacto da COVID-19.
    """)

    dados_cluster = dados.groupby("estado")[["casos", "obitos"]].max().reset_index()

    scaler = StandardScaler()
    dados_norm = scaler.fit_transform(dados_cluster[["casos", "obitos"]])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(dados_norm)
    dados_cluster["cluster"] = clusters

    st.subheader("Distribuição dos Estados por Grupo (Cluster)")
    st.dataframe(dados_cluster.sort_values("cluster"))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=dados_cluster,
        x="casos",
        y="obitos",
        hue="cluster",
        palette="viridis",
        s=100
    )
    ax.set_xlabel("Casos Confirmados")
    ax.set_ylabel("Óbitos Confirmados")
    ax.set_title("Agrupamento de Estados pelo Impacto da COVID-19 (K-Means)")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
    **Interpretação:**  
    - Cada cor representa um grupo (*cluster*) de estados com níveis semelhantes de casos e óbitos.  
    - Estados no grupo superior tendem a ter maior número de casos e óbitos — indicando maior impacto da pandemia.  
    - Os grupos inferiores representam estados menos afetados.  
    - O método **K-Means** é uma técnica de *aprendizado não supervisionado*, ou seja, o algoritmo cria os grupos automaticamente com base nas semelhanças entre os dados.
    """)

# =======================
# RODAPÉ
# =======================
st.markdown("---")
st.caption("Projeto SENAC — Machine Learning Aplicado à Saúde | 2025")
