import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests, io

st.set_page_config(page_title="COVID-19 — Análise Simplificada", layout="wide")
sns.set_style("whitegrid")

# === Função para carregar dados ===
@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"state": "estado", "totalCases": "casos", "deaths": "obitos"})
    df = df[df["estado"] != "TOTAL"]
    df = df.dropna(subset=["date", "estado", "casos", "obitos"])
    return df

# === Carregamento ===
dados = carregar_dados()
if dados.empty:
    st.error("Erro ao carregar os dados. Verifique sua conexão.")
    st.stop()

st.title("Análise Simplificada da COVID-19 no Brasil")
st.markdown("Este painel apresenta uma **análise exploratória clara e direta**, destacando quais estados foram mais afetados e padrões gerais da pandemia no Brasil.")

# === Escolha da página ===
pagina = st.sidebar.radio("Navegação", ["Exploração dos Dados", "Agrupamento (K-Means)"])

# ================================================================
# =====================  EXPLORAÇÃO SIMPLES  =====================
# ================================================================
if pagina == "Exploração dos Dados":
    st.header("Exploração dos Dados (EDA) — Estados Mais Afetados")

    dados_atuais = (
        dados.sort_values("date")
        .groupby("estado", as_index=False)
        .last()[["estado", "casos", "obitos"]]
    )
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    # Top estados por casos e óbitos
    top_casos = dados_atuais.nlargest(5, "casos")[["estado", "casos"]]
    top_obitos = dados_atuais.nlargest(5, "obitos")[["estado", "obitos"]]
    top_letal = dados_atuais.nlargest(5, "letalidade")[["estado", "letalidade"]]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Casos", f"{int(dados_atuais['casos'].sum()):,}".replace(",", "."))
    col2.metric("Total de Óbitos", f"{int(dados_atuais['obitos'].sum()):,}".replace(",", "."))
    col3.metric("Letalidade Média (%)", f"{dados_atuais['letalidade'].mean():.2f}")

    st.subheader("Ranking dos Estados Mais Afetados")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 por Casos Confirmados:**")
        st.table(top_casos)
    with c2:
        st.markdown("**Top 5 por Óbitos:**")
        st.table(top_obitos)

    st.subheader("Letalidade por Estado (%)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=dados_atuais.sort_values("letalidade", ascending=False),
                x="letalidade", y="estado", palette="Reds_r", ax=ax)
    ax.set_xlabel("Letalidade (%)")
    ax.set_ylabel("Estado")
    ax.set_title("Taxa de Letalidade por Estado")
    st.pyplot(fig)
    plt.close(fig)

    # Interpretações automáticas
    pior_estado_casos = top_casos.iloc[0]["estado"]
    pior_estado_obitos = top_obitos.iloc[0]["estado"]
    pior_estado_letal = top_letal.iloc[0]["estado"]
    media_letal = dados_atuais["letalidade"].mean()

    st.markdown(f"""
    ### 🧩 Interpretação dos Dados:
    - O estado com **maior número de casos** é **{pior_estado_casos}**.
    - O estado com **maior número de óbitos** é **{pior_estado_obitos}**.
    - O estado com **maior taxa de letalidade** é **{pior_estado_letal}**.
    - A **letalidade média nacional** é de aproximadamente **{media_letal:.2f}%**.
    - Estados com letalidade alta, mas poucos casos, indicam **baixa testagem** ou **atendimento limitado**.
    - Já estados com muitos casos, mas letalidade menor, sugerem **melhor capacidade de diagnóstico e suporte hospitalar**.
    """)

# ================================================================
# ===================  APRENDIZAGEM NÃO SUPERV.  =================
# ================================================================
elif pagina == "Agrupamento (K-Means)":
    st.header("Aprendizagem Não Supervisionada — Agrupamento de Estados")

    dados_atuais = (
        dados.sort_values("date")
        .groupby("estado", as_index=False)
        .last()[["estado", "casos", "obitos"]]
    )
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    X = dados_atuais[["casos", "obitos", "letalidade"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider("Escolha o número de grupos (k)", min_value=2, max_value=6, value=3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    dados_atuais["cluster"] = labels

    st.subheader("Tabela com Agrupamento")
    st.dataframe(dados_atuais.sort_values("cluster").reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=dados_atuais,
        x="casos", y="obitos",
        hue="cluster",
        palette="Set2", s=100, ax=ax
    )
    ax.set_title("Agrupamento de Estados (K-Means)")
    ax.set_xlabel("Casos")
    ax.set_ylabel("Óbitos")
    st.pyplot(fig)
    plt.close(fig)

    sil = silhouette_score(X_scaled, labels)
    st.markdown(f"**Índice Silhouette:** {sil:.2f} (quanto mais próximo de 1, melhor separação entre os grupos)")

    resumo = dados_atuais.groupby("cluster")[["casos", "obitos", "letalidade"]].mean().round(2).reset_index()
    st.subheader("Médias por Cluster")
    st.table(resumo)

    st.markdown("""
    ### 🧠 Interpretação:
    - Cada cluster representa um **perfil epidemiológico**.
    - Estados em clusters com mais casos e óbitos correspondem às **regiões mais populosas e urbanizadas**.
    - Clusters menores indicam **menor impacto** ou **melhor controle epidemiológico**.
    """)

st.markdown("---")
st.caption("Fonte dos dados: Adaptado de [wcota/covid19br](https://github.com/wcota/covid19br)")
