# Análise de Dados da COVID-19 no Brasil

## Objetivo
Este projeto demonstra a aplicação de técnicas de **Machine Learning** na área da saúde, utilizando dados reais da **COVID-19 no Brasil**.  
O foco é compreender os padrões da pandemia por meio de **análise exploratória de dados (EDA)**, **modelos supervisionados (Regressão Linear)** e **não supervisionados (K-Means)**.

---

## Etapas do Projeto

### 1. Coleta e Tratamento dos Dados
Os dados foram obtidos do repositório público:  
**Fonte:** [wcota/covid19br (GitHub)](https://github.com/wcota/covid19br)  
Arquivo utilizado: `cases-brazil-states.csv`

O dataset contém registros diários por estado, com as seguintes variáveis principais:
- **date** — data da observação  
- **state** — sigla do estado brasileiro  
- **totalCases** — número acumulado de casos confirmados  
- **deaths** — número acumulado de óbitos  

Foram tratados os nomes das colunas e removido o total nacional para focar nas análises por estado.

---

### 2. Exploração dos Dados (EDA)
A etapa de exploração visa entender o comportamento geral da pandemia no Brasil.

**Principais análises:**
- Ranking de estados mais afetados em número de casos e óbitos.  
- Cálculo da **taxa de letalidade (%)** por estado.  
- Visualização de gráficos de barras e tabelas comparativas.  

**Interpretação:**  
Os estados mais populosos (como São Paulo e Minas Gerais) apresentam os maiores números absolutos de casos e óbitos.  
Entretanto, estados menores podem ter **taxas de letalidade proporcionalmente mais altas**, indicando possíveis diferenças na estrutura de atendimento e testagem.

---

### 3. Análise Individual por Estado
Permite selecionar um estado específico e visualizar:
- Evolução temporal de **casos e óbitos**.  
- Cálculo da **taxa média de crescimento diário de casos**.  

**Interpretação:**  
O gráfico de linhas mostra a progressão acumulada ao longo do tempo, facilitando a observação de períodos de aumento rápido ou estabilização da pandemia em cada estado.

---

### 4. Aprendizado Supervisionado — Regressão Linear
Foi aplicado um modelo simples de **Regressão Linear** para verificar a relação entre o número de casos e o número de óbitos.

**Métricas utilizadas:**
- **R²** — mostra o quanto os casos explicam os óbitos.  
- **MAE** — erro absoluto médio.  
- **RMSE** — raiz do erro quadrático médio.  

**Interpretação:**  
O modelo mostra uma relação linear entre casos e óbitos, confirmando que o aumento de casos tende a resultar em mais óbitos.  
Contudo, o modelo é limitado, pois não considera fatores externos (como vacinação, faixa etária e infraestrutura hospitalar).

---

### 5. Aprendizado Não Supervisionado — K-Means
Nesta etapa, aplicou-se o algoritmo **K-Means** para identificar grupos de estados com comportamentos semelhantes quanto ao impacto da pandemia.

O usuário pode **selecionar o número de clusters** (grupos) e visualizar:
- Uma **tabela** indicando a qual cluster pertence cada estado.  
- Um **gráfico de dispersão** colorido, mostrando como os estados se agrupam de acordo com casos e óbitos.

**Interpretação:**  
Cada cor representa um grupo de estados com características parecidas.  
Por exemplo:
- Um cluster pode agrupar estados com **altos números de casos e óbitos**.  
- Outro pode representar estados **menos afetados**.  
Isso ajuda a entender padrões regionais e o comportamento coletivo das unidades federativas.

---

### 6. Visão Geral e Indicadores Nacionais
Além das análises específicas, a aplicação apresenta um **resumo geral do Brasil**, com:
- Total de casos confirmados.  
- Total de óbitos registrados.  
- Letalidade média nacional.  

Essas métricas ajudam a contextualizar as análises individuais e comparativas.

---

## Execução

### 1. Instale as dependências:
```bash
pip install -r requirements.txt
