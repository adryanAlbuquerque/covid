# Machine Learning Aplicado à Saúde — Análise de Óbitos por COVID-19

## Objetivo  
Este projeto tem como objetivo demonstrar a aplicação de **técnicas de aprendizado supervisionado** em dados da área da saúde.  
O modelo utilizado foi a **Regressão Linear**, que busca estimar o número de óbitos a partir do total de casos confirmados de COVID-19.

O trabalho faz parte da disciplina **Machine Learning Aplicado à Saúde**, e tem como foco mostrar todas as etapas do processo de análise e modelagem de dados.

---

## Etapas do Projeto  
1. **Coleta e tratamento dos dados** — os dados utilizados representam um conjunto simplificado baseado na estrutura do dataset público de COVID-19 no Brasil, disponível em:  
   [wcota/covid19br – GitHub](https://github.com/wcota/covid19br)

2. **Análise Exploratória (EDA)** — análise gráfica e estatística da relação entre casos e óbitos, identificando tendências e correlações.

3. **Modelagem Supervisionada** — aplicação de um modelo de **Regressão Linear simples** para prever óbitos com base no número de casos.

4. **Interpretação dos resultados** — avaliação das métricas de desempenho e interpretação dos resultados obtidos pelo modelo.

---

## Métricas Utilizadas  
- **R² (Coeficiente de Determinação)** — indica o quanto o modelo explica da variabilidade dos dados.  
- **MAE (Erro Absoluto Médio)** — mede o erro médio absoluto entre as previsões e os valores reais.  
- **RMSE (Raiz do Erro Quadrático Médio)** — mostra o desvio médio entre os valores previstos e observados.

Essas métricas permitem avaliar se o modelo possui bom ajuste e quão próximos estão os valores previstos dos reais.

---

## Execução do Projeto  
1. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt


2. streamlit run app.py

3. Após iniciar, acesse o link exibido no terminal (geralmente):
http://localhost:8501
