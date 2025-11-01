Análise de Dados da COVID-19 no Brasil

Objetivo
Este projeto demonstra a aplicação de técnicas de Machine Learning na área da saúde, utilizando dados reais da COVID-19 no Brasil. 
O foco é compreender os padrões da pandemia por meio de análise exploratória de dados (EDA), modelos supervisionados (Regressão Linear) e não supervisionados (K-Means).

Etapas do Projeto

1. Coleta e Tratamento dos Dados
Os dados foram obtidos do repositório público:
Fonte: wcota/covid19br (GitHub) - https://github.com/wcota/covid19br
Arquivo utilizado: cases-brazil-states.csv

O dataset contém registros diários por estado, com as seguintes variáveis principais:
- date — data da observação
- state — sigla do estado brasileiro
- totalCases — número acumulado de casos confirmados
- deaths — número acumulado de óbitos

Foi feito tratamento das colunas, renomeação de variáveis e remoção da linha “TOTAL” para focar apenas em dados por estado.

2. Exploração dos Dados (EDA)
Esta etapa busca entender o panorama geral da COVID-19 no Brasil.

Principais análises:
- Total de casos e óbitos por estado.
- Taxa de letalidade média nacional e por estado.
- Gráficos comparativos mostrando quais estados foram mais afetados.

Interpretação:
Os estados mais populosos, como São Paulo e Minas Gerais, concentram o maior número de casos e óbitos.
Já a letalidade tende a variar conforme a infraestrutura hospitalar e o número de testagens em cada região.

3. Análise Individual por Estado
O usuário pode selecionar um estado específico e visualizar:
- A evolução temporal dos casos e óbitos.
- A taxa média de crescimento diário dos casos confirmados.

Interpretação:
Essa visão detalhada mostra como a pandemia evoluiu em cada estado, permitindo identificar picos de contágio e períodos de estabilização.

4. Aprendizado Supervisionado — Regressão Linear
Foi aplicada uma Regressão Linear simples para estimar o número de óbitos a partir do número de casos.

Métricas apresentadas:
- R² — indica o quanto os casos explicam os óbitos.
- MAE — erro médio absoluto.
- RMSE — raiz do erro quadrático médio.

Interpretação:
O modelo mostra uma relação linear positiva: quanto maior o número de casos, maior o número de óbitos.
Porém, o modelo é simplificado e não leva em conta fatores externos como vacinação, idade média da população ou políticas públicas.

5. Aprendizado Não Supervisionado — K-Means
Nesta etapa, utilizou-se o algoritmo K-Means para agrupar os estados brasileiros conforme o impacto da pandemia.

O usuário pode selecionar o número de clusters (grupos) desejado.

Saídas apresentadas:
- Tabela com os estados e o cluster correspondente.
- Gráfico de dispersão colorido, com cada cor representando um grupo de estados semelhantes.

Interpretação:
O K-Means agrupa os estados com base em padrões de comportamento:
- Um cluster pode representar estados muito afetados (altos números de casos e óbitos).
- Outro cluster pode representar estados com impacto moderado ou baixo.

Esses agrupamentos ajudam a identificar padrões regionais ocultos e podem auxiliar na tomada de decisões estratégicas em saúde pública.

6. Indicadores Nacionais
A aplicação também apresenta uma visão geral do país:
- Total de casos confirmados.
- Total de óbitos.
- Taxa média de letalidade.

Esses indicadores fornecem um resumo rápido da situação nacional, servindo como base para as análises por estado e os modelos de aprendizado.

Execução da Aplicação

1. Clonar o repositório
git clone https://github.com/seuusuario/covid-ml-brasil.git
cd covid-ml-brasil

2. Criar e ativar um ambiente virtual (opcional)
python -m venv venv

No Windows:
venv\Scripts\activate

No Mac/Linux:
source venv/bin/activate

3. Instalar as dependências
pip install -r requirements.txt

4. Executar a aplicação Streamlit
streamlit run app.py

5. Acessar no navegador
http://localhost:8501

A aplicação será aberta no navegador e exibirá as abas:
- Exploração dos Dados
- Análise por Estado
- Modelagem Supervisionada
- Modelagem Não Supervisionada

Estrutura do Projeto
.
├── app.py
├── requirements.txt
├── README.txt

Tecnologias Utilizadas
- Python 3.10+
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- Streamlit

Considerações Finais
Este projeto mostra como a análise de dados e o aprendizado de máquina podem apoiar decisões em saúde pública.
Mesmo com modelos simples, é possível identificar tendências, padrões regionais e relações importantes entre variáveis epidemiológicas.

O uso de Regressão Linear e K-Means demonstra duas abordagens complementares:
- Supervisionada: baseada em relações conhecidas entre variáveis.
- Não supervisionada: baseada na descoberta de padrões ocultos nos dados.

Essas técnicas reforçam o potencial da Ciência de Dados aplicada à Saúde.

Autor: Adryan Albuquerque


requirements.txt
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
