import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import requests
import io

st.set_page_config(page_title="Análise de COVID-19 no Brasil", layout="wide")
sns.set_style("whitegrid")

@st.cache_data
def carregar_dados():
    url = "https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        st.error(f"Erro ao carregar dados remotos: {e}")
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.rename(columns={"state": "estado", "totalCases": "casos", "deaths": "obitos"})
    df = df[["date", "estado", "casos", "obitos"]]
    df = df[df["estado"] != "TOTAL"]
    df = df.dropna(subset=["date", "estado", "casos", "obitos"])
    return df

dados = carregar_dados()
if dados.empty:
    st.title("Análise de COVID-19 no Brasil")
    st.error("Não foi possível carregar os dados. Verifique sua conexão ou tente novamente.")
    st.stop()

st.title("Análise de COVID-19 no Brasil")
st.markdown("Aplicação focada em **Exploração de Dados** e **Aprendizagem Não Supervisionada (K-Means)**. Todos os gráficos têm uma interpretação direta logo abaixo para apresentação.")

pagina = st.sidebar.radio("Navegação", ["Exploração dos Dados", "Análise por Estado", "Agrupamento (K-Means)"])

if pagina == "Exploração dos Dados":
    st.header("Exploração dos Dados (EDA)")

    dados_atuais = dados.sort_values("date").groupby("estado", as_index=False).last()[["estado", "casos", "obitos"]]
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Casos Totais (soma por estado)", f"{int(dados_atuais['casos'].sum()):,}".replace(",", "."))
    c2.metric("Óbitos Totais (soma por estado)", f"{int(dados_atuais['obitos'].sum()):,}".replace(",", "."))
    c3.metric("Letalidade Média (%)", f"{dados_atuais['letalidade'].mean():.2f}")

    st.subheader("Distribuição de Casos e Óbitos por Estado")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(dados_atuais["casos"], bins=10, kde=True, ax=ax, color="steelblue")
        ax.set_title("Distribuição de Casos (por estado)")
        ax.set_xlabel("Casos")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("Interpretação: a distribuição mostra concentração de casos nos estados maiores; há assimetria — poucos estados concentram a maior parte dos casos.")

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(dados_atuais["obitos"], bins=10, kde=True, ax=ax, color="indianred")
        ax.set_title("Distribuição de Óbitos (por estado)")
        ax.set_xlabel("Óbitos")
        st.pyplot(fig)
        plt.close(fig)
        st.markdown("Interpretação: padrão semelhante ao de casos; presença de outliers (estados com óbitos muito maiores).")

    st.subheader("Correlação Casos x Óbitos")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=dados_atuais, x="casos", y="obitos", s=100, ax=ax)
    for i, row in dados_atuais.iterrows():
        ax.text(row["casos"], row["obitos"], row["estado"], fontsize=8, alpha=0.8)
    ax.set_title("Casos vs Óbitos por Estado")
    ax.set_xlabel("Casos")
    ax.set_ylabel("Óbitos")
    st.pyplot(fig)
    plt.close(fig)
    corr = dados_atuais["casos"].corr(dados_atuais["obitos"])
    st.markdown(f"Interpretação: correlação Pearson entre casos e óbitos = **{corr:.2f}**. Isso indica relação positiva forte — casos tendem a acompanhar óbitos, porém não explicam totalmente as diferenças entre estados.")

    st.subheader("Heatmap de Correlação entre Variáveis")
    corrmat = dados_atuais[["casos", "obitos", "letalidade"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corrmat, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matriz de Correlação")
    st.pyplot(fig)
    plt.close(fig)
    st.markdown("Interpretação: a matriz ajuda a ver quais variáveis têm maior associação; casos e óbitos apresentam alta correlação, letalidade tem correlação distinta.")

elif pagina == "Análise por Estado":
    st.header("Análise Individual por Estado")
    estados = sorted(dados["estado"].unique())
    estado_sel = st.selectbox("Selecione o estado", estados, index=estados.index("SP") if "SP" in estados else 0)
    df_estado = dados[dados["estado"] == estado_sel].sort_values("date")

    st.subheader(f"Evolução temporal — {estado_sel}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_estado["date"], df_estado["casos"], label="Casos", color="steelblue", linewidth=2)
    ax.plot(df_estado["date"], df_estado["obitos"], label="Óbitos", color="indianred", linewidth=2)
    ax.set_xlabel("Data")
    ax.set_ylabel("Quantidade acumulada")
    ax.legend()
    ax.set_title(f"Casos e Óbitos acumulados em {estado_sel}")
    st.pyplot(fig)
    plt.close(fig)
    crescimento = df_estado["casos"].diff().fillna(0)
    crescimento_media = crescimento.mean()
    st.markdown(f"Interpretação: crescimento médio diário de novos casos (média simples) ≈ **{crescimento_media:.0f} casos/dia**. Observe picos e períodos de estabilização no gráfico temporal.")

    st.subheader("Resumo estatístico do estado")
    resumo = df_estado[["casos", "obitos"]].describe().loc[["min", "mean", "max"]].rename(index={"min":"mínimo","mean":"médio","max":"máximo"})
    st.table(resumo)

elif pagina == "Agrupamento (K-Means)":
    st.header("Agrupamento de Estados (K-Means)")
    dados_atuais = dados.sort_values("date").groupby("estado", as_index=False).last()[["estado", "casos", "obitos"]]
    dados_atuais["letalidade"] = (dados_atuais["obitos"] / dados_atuais["casos"]) * 100

    X = dados_atuais[["casos", "obitos"]].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = st.slider("Escolha o número de clusters (k)", min_value=2, max_value=6, value=3)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    dados_atuais["cluster"] = labels

    st.subheader("Tabela de estados com cluster atribuído")
    st.dataframe(dados_atuais.sort_values("cluster").reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", k)
    for cluster_id in sorted(dados_atuais["cluster"].unique()):
        subset = dados_atuais[dados_atuais["cluster"] == cluster_id]
        ax.scatter(subset["casos"], subset["obitos"], s=100, label=f"Cluster {cluster_id}", alpha=0.8)
        for _, row in subset.iterrows():
            ax.text(row["casos"], row["obitos"], row["estado"], fontsize=8)
    ax.set_xlabel("Casos")
    ax.set_ylabel("Óbitos")
    ax.set_title("Clusters de Estados por Casos e Óbitos")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    if len(dados_atuais) > k:
        sil = silhouette_score(Xs, labels)
        st.markdown(f"Índice Silhouette (qualidade do agrupamento): **{sil:.2f}**")
    else:
        st.markdown("Índice Silhouette: não disponível (poucos pontos).")

    resumo_clusters = dados_atuais.groupby("cluster")[["casos", "obitos", "letalidade"]].mean().round(2).reset_index()
    st.subheader("Média dos indicadores por cluster")
    st.table(resumo_clusters)

    st.markdown("Interpretação geral: os clusters separam estados por carga absoluta (casos/óbitos). Geralmente, um cluster agrupa grandes centros com alto número de casos e óbitos; outros reúnem estados com carga média ou baixa. Use essas informações para direcionar hipóteses sobre infraestrutura, testagem e fatores sociodemográficos.")

    csv = dados_atuais.to_csv(index=False).encode("utf-8")
    st.download_button(label="Baixar resultados (CSV)", data=csv, file_name="clusters_estados.csv", mime="text/csv")

st.markdown("---")
st.caption("Fonte dos dados: adaptado de wcota/covid19br (https://github.com/wcota/covid19br).")
