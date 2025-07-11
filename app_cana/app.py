
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import genextreme
import emcee
import corner
import traceback
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from geopy.exc import GeocoderServiceError
import hashlib

# --- FUNCOES AUXILIARES ---
def formatar_reais(valor):
    """Formata um numero float no padrao R$ 1.000,00"""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def get_tuple_hash(data_tuple):
    """Cria um hash SHA256 para uma tupla de dados para usar como chave de cache."""
    return hashlib.sha256(str(data_tuple).encode()).hexdigest()

# --- Bloco 1: CARREGAMENTO DE DADOS (CACHEADO) ---

@st.cache_data
def carregar_dados_meteorologicos():
    """Carrega os dados meteorologicos de um arquivo Parquet."""
    try:
        caminho_completo = '../processamento_dados/dados_inmet_nacional_unificado_b.parquet'
        colunas_para_carregar = ["DATA", "CODIGO_ESTACAO", "NOME_ESTACAO", "LATITUDE", "LONGITUDE", "CHUVA"]
        df = pd.read_parquet(caminho_completo, columns=colunas_para_carregar)
        df["DATA"] = pd.to_datetime(df["DATA"], errors='coerce')
        if df.index.name == 'DATA': df = df.reset_index()
        df.dropna(subset=colunas_para_carregar, inplace=True)
        df.set_index("DATA", inplace=True)
        df['CHUVA'] = pd.to_numeric(df['CHUVA'], errors='coerce')
        df.dropna(subset=['CHUVA'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Arquivo mestre '{caminho_completo}' nao encontrado.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados meteorológicos: {e}")
        return None

@st.cache_data
def carregar_dados_produtividade():
    """Carrega os dados de produtividade da cana-de-açúcar."""
    try:
        caminho_arquivo = "SerieHistoricaCana.txt"
        df = pd.read_csv(caminho_arquivo, sep=';', encoding='latin1')
        df['ano_agricola'] = pd.to_numeric(df['ano_agricola'].astype(str).str.split('/').str[0], errors='coerce')
        if 'produtcao_atr_kg_t' in df.columns:
            df['produtividade'] = pd.to_numeric(df['produtcao_atr_kg_t'], errors='coerce')
        elif 'produtividade_t_ha' in df.columns:
            df['produtividade'] = pd.to_numeric(df['produtividade_t_ha'], errors='coerce')
        return df[['ano_agricola', 'produtividade']].dropna()
    except FileNotFoundError:
        st.error(f"Arquivo '{caminho_arquivo}' nao foi encontrado.")
        return None
    except Exception as e:
        st.error(f"Erro ao carregar dados de produtividade: {e}")
        return None

@st.cache_data(ttl=3600)
def buscar_coordenadas(query):
    """Função cacheada e robusta para buscar coordenadas de uma localização."""
    try:
        geolocator = Nominatim(user_agent="change_safra_app_v4", timeout=10)
        location = geolocator.geocode(query)
        if location: return (location.address, location.latitude, location.longitude)
        return (None, None, None)
    except GeocoderServiceError as e:
        st.sidebar.error(f"Serviço de geolocalização indisponível: {e}.")
        return (None, None, None)
    except Exception as e:
        st.sidebar.error(f"Erro ao buscar localização: {e}")
        return (None, None, None)

# --- Bloco 2: FUNÇÕES DE ANÁLISE E MODELAGEM ---

def analisar_eventos_extremos(df_meteorologico, fase):
    fases = {'Brotacao': (9, 11), 'Desenvolvimento': (12, 4), 'Maturacao': (5, 8)}
    inicio, fim = fases[fase]
    df_fase = df_meteorologico[((df_meteorologico.index.month >= inicio) | (df_meteorologico.index.month <= fim)) if inicio > fim else ((df_meteorologico.index.month >= inicio) & (df_meteorologico.index.month <= fim))].copy()
    df_fase['ano_agricola'] = df_fase.index.year.where(df_fase.index.month >= inicio, df_fase.index.year - 1)
    if 'CHUVA' not in df_fase.columns: return None
    return df_fase.groupby('ano_agricola')['CHUVA'].apply(lambda x: (x <= 1).cumsum().max())

def relacionar_eventos_com_produtividade(serie_maximos_veranico, df_producao):
    if serie_maximos_veranico is None or df_producao is None: return pd.DataFrame()
    df_analise = pd.DataFrame({'parametro_risco': serie_maximos_veranico})
    df_merged = df_producao.merge(df_analise, on='ano_agricola', how='inner')
    df_merged['var_produtividade'] = df_merged['produtividade'].pct_change() * 100
    df_merged['houve_perda'] = (df_merged['var_produtividade'] < -5).astype(int)
    return df_merged

def executar_amostragem_gev(data_tuple, progress_bar):
    """Executa a amostragem MCMC para o modelo GEV e atualiza uma barra de progresso."""
    data = pd.Series(data_tuple)
    def log_prob(params, data_points):
        mu, log_sigma, xi = params; sigma = np.exp(log_sigma)
        if sigma <= 0 or not np.isfinite(np.sum(genextreme.logpdf(data_points, c=-xi, loc=mu, scale=sigma))): return -np.inf
        return np.sum(genextreme.logpdf(data_points, c=-xi, loc=mu, scale=sigma))
    initial = np.array([np.mean(data), np.log(np.std(data)), 0.01])
    nwalkers, ndim, n_iter = 50, 3, 2000
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[data])
    for i, _ in enumerate(sampler.sample(p0, iterations=n_iter)):
        if i % 100 == 0:
            progress_bar.progress(i / n_iter, text=f"Simulação GEV: {i}/{n_iter} iterações")
    progress_bar.progress(1.0, text="Simulação GEV concluída!")
    return sampler.get_chain(discard=500, thin=15, flat=True)

def executar_amostragem_logistica(X_tuple, y_tuple, progress_bar):
    """Executa a amostragem MCMC para o modelo Logístico e atualiza uma barra de progresso."""
    X = pd.Series(X_tuple); y = pd.Series(y_tuple)
    def log_prob(params, X_points, y_points):
        beta0, beta1 = params
        prob = 1 / (1 + np.exp(-(beta0 + beta1 * X_points)))
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        log_likelihood = np.sum(y_points * np.log(prob) + (1 - y_points) * np.log(1 - prob))
        if not np.isfinite(log_likelihood): return -np.inf
        return log_likelihood
    initial = np.array([0.0, 0.0])
    nwalkers, ndim, n_iter = 50, 2, 2000
    p0 = initial + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, args=[X, y])
    for i, _ in enumerate(sampler.sample(p0, iterations=n_iter)):
        if i % 100 == 0:
            progress_bar.progress(i / n_iter, text=f"Simulação Logística: {i}/{n_iter} iterações")
    progress_bar.progress(1.0, text="Simulação Logística concluída!")
    return sampler.get_chain(discard=500, thin=15, flat=True)

# --- Bloco 3: FUNÇÕES DE PLOTAGEM ---
def plotar_analise_eventos(df_analise):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    sns.regplot(data=df_analise, x='parametro_risco', y='var_produtividade', ax=axes[0], line_kws={"color": "#008080"})
    axes[0].set_title('Produtividade vs. Parâmetro de Risco'); axes[0].grid(True)
    sns.lineplot(data=df_analise, x='ano_agricola', y='parametro_risco', ax=axes[1], marker='o', color="#008080")
    axes[1].set_title('Série Histórica do Parâmetro de Risco'); axes[1].grid(True)
    plt.tight_layout(); return fig

def plotar_ajuste_gev(samples, data):
    mu, log_sigma, xi = np.median(samples, axis=0); sigma = np.exp(log_sigma)
    fig, ax = plt.subplots()
    ax.hist(data, bins='auto', density=True, alpha=0.6, label='Dados Históricos')
    x_plot = np.linspace(data.min(), data.max(), 100)
    pdf_plot = genextreme.pdf(x_plot, c=-xi, loc=mu, scale=sigma)
    ax.plot(x_plot, pdf_plot, 'r-', lw=2, label='Modelo GEV Ajustado'); ax.legend(); return fig

def plotar_corner(samples, labels):
    return corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)

def plotar_modelo_logistico(samples, X, y):
    beta0, beta1 = np.median(samples, axis=0); fig, ax = plt.subplots()
    ax.scatter(X, y, alpha=0.6, label='Anos com Quebra de Safra (Y=1)')
    x_range = np.linspace(X.min(), X.max(), 100)
    prob = 1 / (1 + np.exp(-(beta0 + beta1 * x_range)))
    ax.plot(x_range, prob, 'r-', label='Curva Logística Ajustada'); ax.set_title('Modelo de Perda'); ax.legend(); return fig

# --- Bloco 4: INTERFACE PRINCIPAL DO STREAMLIT ---

st.set_page_config(layout="wide", page_title="Calculadora Cana - ChangeSafra", page_icon="../processamento_dados/logo_change_safra_transparent.png")

col_logo, col_header_text = st.columns([1, 4])
with col_logo:
    st.image("../processamento_dados/logo_change_safra_transparent.png", width=450)
with col_header_text:
    st.title("ChangeSafra")
    st.subheader("Calculadora Paramétrica: Cana-de-Açúcar")
st.link_button("<< Voltar ao Portal Principal", "http://44.201.48.68:8506"); st.markdown("---")
st.markdown("Esta ferramenta utiliza dados históricos de clima e produtividade para modelar o risco de quebras de safra de cana-de-açúcar associadas a secas prolongadas (veranicos).")

with st.expander("Clique para saber mais sobre a Metodologia e as Fontes dos Dados"):
    st.subheader("Dados Meteorologicos - INMET")
    st.markdown("- **O que foi extraido?** Dados diarios de estacoes meteorologicas, principalmente a **precipitacao (chuva)**.\n- **Como sao usados??** Para encontrar e medir a duracao dos **veranicos**. A duracao do veranico mais longo de cada ano e o principal fator de risco que usamos na analise.")
    st.subheader("Dados de Produtividade - CONAB")
    st.markdown("- **O que foi extraido?** Dados anuais da safra de **cana-de-acucar**, especificamente a **produtividade**.\n- **Como sao usados?** Para definir se um ano teve **quebra de safra**. Uma queda acentuada na produtividade e classificada como um 'ano de perda'.")
    st.info("A combinacao destes dois conjuntos de dados permite encontrar a correlacao estatistica entre um parametro climatico (o gatilho) e a perda de safra (o risco segurado).")
    st.markdown("---")

    with st.container(border=True):
        st.subheader("Comparativo: Seguro Parametrico vs. Seguro Tradicional")
        st.image("../processamento_dados/imagem_comparativo_seguro.png", use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='text-align: center;'><b>Seguro Tradicional</b><br><br>- O pagamento é acionado por uma <b>perda real</b>.<br>- O reembolso é feito após <b>avaliação por perito</b>.<br>- A indenização é demorada.<br>- Apólice com redação padrão.</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align: center;'><b>Seguro Paramétrico (Este App)</b><br><br>- O pagamento é acionado por um <b>evento pré-definido</b>.<br>- O pagamento é feito com base no valor acordado.<br>- A indenização é rápida.<br>- Apólice altamente personalizável.</div>", unsafe_allow_html=True)


df_meteo_completo = carregar_dados_meteorologicos()

if df_meteo_completo is not None:
    st.sidebar.header("Passo 1: Selecione a Localização")

    if 'last_processed_location' not in st.session_state: st.session_state.last_processed_location = ""
    if 'estacao_selecionada_code' not in st.session_state: st.session_state.estacao_selecionada_code = None
    if 'df_estacoes_proximas' not in st.session_state: st.session_state.df_estacoes_proximas = pd.DataFrame()
    if 'mcmc_cache' not in st.session_state: st.session_state.mcmc_cache = {}

    estados_br = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
    estado_selecionado = st.sidebar.selectbox("Estado (Opcional):", [""] + sorted(estados_br))
    localizacao_digitada = st.sidebar.text_input("Digite uma cidade ou local:")

    if localizacao_digitada and localizacao_digitada != st.session_state.last_processed_location:
        st.session_state.last_processed_location = localizacao_digitada
        st.session_state.estacao_selecionada_code = None
        st.session_state.df_estacoes_proximas = pd.DataFrame()

        query = f"{localizacao_digitada}, {estado_selecionado}, Brasil" if estado_selecionado else f"{localizacao_digitada}, Brasil"
        with st.spinner("Buscando localização e estações próximas..."):
            endereco, lat, lon = buscar_coordenadas(query)
            if lat and lon:
                st.sidebar.success(f"Localização: {endereco.split(',')[0]}")
                coordenadas_usuario = (lat, lon)

                raio_graus = 2.0
                df_raio = df_meteo_completo[(df_meteo_completo['LATITUDE'].between(lat - raio_graus, lat + raio_graus)) & (df_meteo_completo['LONGITUDE'].between(lon - raio_graus, lon + raio_graus))].drop_duplicates(subset=['CODIGO_ESTACAO']).copy()
                if not df_raio.empty:
                    df_raio['distancia'] = df_raio.apply(lambda r: geodesic((lat, lon), (r['LATITUDE'], r['LONGITUDE'])).km, axis=1)
                    st.session_state.df_estacoes_proximas = df_raio.sort_values('distancia').head(20)
                    st.session_state.estacao_selecionada_code = st.session_state.df_estacoes_proximas['CODIGO_ESTACAO'].iloc[0]
                else: st.sidebar.warning("Nenhuma estação encontrada neste raio.")
            else: st.sidebar.warning("Localização não encontrada.")

    if not st.session_state.df_estacoes_proximas.empty:
        st.sidebar.markdown("---")
        st.sidebar.header("Passo 2: Confirme a Estação")
        df_estacoes = st.session_state.df_estacoes_proximas
        df_estacoes['display_name'] = df_estacoes.apply(lambda r: f"{r['NOME_ESTACAO'].title()} ({r['distancia']:.0f} km)", axis=1)

        lista_opcoes = df_estacoes['display_name'].tolist()
        try:
            display_name_atual = df_estacoes.loc[df_estacoes['CODIGO_ESTACAO'] == st.session_state.estacao_selecionada_code, 'display_name'].iloc[0]
            index_selecionado = lista_opcoes.index(display_name_atual)
        except (ValueError, IndexError): index_selecionado = 0

        def atualizar_estacao_selecionada():
            display_name = st.session_state.seletor_estacao_key
            codigo = df_estacoes[df_estacoes['display_name'] == display_name]['CODIGO_ESTACAO'].iloc[0]
            st.session_state.estacao_selecionada_code = codigo

        st.sidebar.selectbox("Estação meteorológica:", lista_opcoes, index=index_selecionado, key='seletor_estacao_key', on_change=atualizar_estacao_selecionada)

    if st.session_state.get('estacao_selecionada_code'):
        st.sidebar.markdown("---")
        st.sidebar.header("Passo 3: Execute a Análise")
        df_meteo_estacao = df_meteo_completo[df_meteo_completo['CODIGO_ESTACAO'] == st.session_state.estacao_selecionada_code]
        fase_selecionada = st.sidebar.selectbox("Fase da Cultura:", ["Desenvolvimento", "Brotacao", "Maturacao"])

        if st.sidebar.button("Executar Análise Completa", type="primary"):
            if df_meteo_estacao.empty:
                st.error("Erro interno: DataFrame da estação está vazio.")
            else:
                df_prod = carregar_dados_produtividade()
                if df_prod is not None:
                    st.session_state.analise_concluida = False
                    st.session_state.fig_analise_eventos, st.session_state.fig_gev_ajuste, st.session_state.fig_logistico_ajuste, st.session_state.fig_corner_gev, st.session_state.fig_corner_logistico = None, None, None, None, None
                    parametro_risco = analisar_eventos_extremos(df_meteo_estacao, fase_selecionada)
                    df_analise = relacionar_eventos_com_produtividade(parametro_risco, df_prod)
                    df_analise.dropna(inplace=True)

                    if len(df_analise) < 10:
                        st.warning(f"Dados Históricos Insuficientes: Apenas {len(df_analise)} anos. São necessários pelo menos 10 anos.")
                    else:
                        placeholder = st.empty()
                        data_tuple_gev = tuple(df_analise['parametro_risco'])
                        cache_key_gev = get_tuple_hash(data_tuple_gev)
                        if cache_key_gev in st.session_state.mcmc_cache:
                            samples_gev = st.session_state.mcmc_cache[cache_key_gev]
                        else:
                            progress_bar_gev = placeholder.progress(0, text="Iniciando simulação GEV...")
                            samples_gev = executar_amostragem_gev(data_tuple_gev, progress_bar_gev)
                            st.session_state.mcmc_cache[cache_key_gev] = samples_gev

                        st.session_state.fig_corner_gev = plotar_corner(samples_gev, labels=['µ', 's', '?'])
                        mu_gev, log_sigma_gev, xi_gev = np.median(samples_gev, axis=0)
                        df_analise['prob_excedencia_gev'] = 1 - genextreme.cdf(df_analise['parametro_risco'], c=-xi_gev, loc=mu_gev, scale=np.exp(log_sigma_gev))

                        data_tuple_log = (tuple(df_analise['prob_excedencia_gev']), tuple(df_analise['houve_perda']))
                        cache_key_log = get_tuple_hash(data_tuple_log)
                        if cache_key_log in st.session_state.mcmc_cache:
                            samples_logistico = st.session_state.mcmc_cache[cache_key_log]
                        else:
                            progress_bar_log = placeholder.progress(0, text="Iniciando simulação Logística...")
                            samples_logistico = executar_amostragem_logistica(data_tuple_log[0], data_tuple_log[1], progress_bar_log)
                            st.session_state.mcmc_cache[cache_key_log] = samples_logistico

                        placeholder.empty()
                        st.session_state.fig_corner_logistico = plotar_corner(samples_logistico, labels=['ß0', 'ß1'])
                        st.session_state.fig_analise_eventos = plotar_analise_eventos(df_analise)
                        st.session_state.fig_gev_ajuste = plotar_ajuste_gev(samples_gev, df_analise['parametro_risco'])
                        st.session_state.fig_logistico_ajuste = plotar_modelo_logistico(samples_logistico, df_analise['prob_excedencia_gev'], df_analise['houve_perda'])
                        st.session_state.df_analise = df_analise
                        st.session_state.analise_concluida = True

                    if st.session_state.get('analise_concluida'): st.success("Análise e modelagem concluídas!")

if st.session_state.get('analise_concluida', False):
    df_analise = st.session_state['df_analise']

    display_name_final = st.session_state.df_estacoes_proximas.loc[st.session_state.df_estacoes_proximas['CODIGO_ESTACAO'] == st.session_state.estacao_selecionada_code, 'display_name'].iloc[0]
    st.header(f"Resultados para Cana-de-Açúcar (Estação: {display_name_final})")

    tab1, tab2, tab3, tab4 = st.tabs(["Resumo e Simulador", "Análise Climática", "Modelo de Risco (GEV)", "Modelo de Perda"])

    with tab1:
        st.subheader("Simulador Interativo de Cobertura")
        col_sim_1, col_sim_2 = st.columns(2)
        with col_sim_1:
            valor_indenizacao_input = st.number_input("1. Defina o Valor da Indenização (R$ / ha)", min_value=500.0, max_value=10000.0, value=2000.0, step=100.0)
        with col_sim_2:
            gatilho_prob_input = st.slider("2. Selecione o Nível de Risco (Gatilho)", min_value=5, max_value=33, value=10, step=1, format="%d%%", help="Probabilidade do evento que aciona o seguro. Ex: 10% = evento de 1 em 10 anos.")

        gatilho_prob = gatilho_prob_input / 100.0
        periodo_retorno = int(1 / gatilho_prob) if gatilho_prob > 0 else 0
        premio_estimado = gatilho_prob * valor_indenizacao_input * 1.2

        st.markdown("---"); st.subheader("Resultados da sua Simulação")
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Indenização / ha", formatar_reais(valor_indenizacao_input), help="Este é o valor em dinheiro que você recebe por hectare da sua lavoura, caso o evento de seca (o gatilho) aconteça. Pense nisso como o seu 'capital de giro de emergência', que chega de forma rápida para garantir que você possa cobrir seus custos, pagar funcionários e se preparar para a próxima safra sem se endividar.")
        mcol2.metric("Prêmio Estimado / ha", formatar_reais(premio_estimado), help="Este é o valor anual que você investe por hectare para ter a sua proteção garantida. É um custo fixo e planejado no seu orçamento, que te protege contra uma perda muito maior e imprevisível. É a troca da incerteza por tranquilidade.")
        mcol3.metric("Gatilho (Retorno)", f"1 em {periodo_retorno} anos", help="Este é o 'coração' do seu seguro transparente. Um gatilho de '1 em 10 anos' significa que o seguro é acionado por um nível de seca tão raro que, em média, só acontece uma vez a cada 10 anos. Não é sobre a data, mas sobre a intensidade do evento. Você escolhe o quão severa a seca precisa ser para acionar sua proteção.")

    with tab2:
        st.subheader("Relação entre Clima e Safras Passadas")
        if st.session_state.get('fig_analise_eventos'):
            st.pyplot(st.session_state.fig_analise_eventos)
            with st.expander("Entenda este gráfico"):
                st.markdown("""
                - **O que este gráfico mostra?** A relação histórica entre a duração da seca (eixo horizontal) e a variação da produtividade da safra (eixo vertical) na sua região.
                - **O que isso significa para você?** A linha de tendência mostra que, para esta cultura e região, secas mais longas estão consistentemente associadas a maiores quedas de produtividade.
                - **Como isso afeta seu seguro?** Esta correlação é a base do seguro paramétrico. Ela prova que podemos usar um índice de seca como um gatilho confiável para prever perdas e acionar sua proteção.
                """)

    with tab3:
        st.subheader("Análise de Risco (Distribuição de Extremos Generalizada - GEV)")
        if st.session_state.get('fig_gev_ajuste'):
            st.pyplot(st.session_state.fig_gev_ajuste)
            st.caption("Ajuste do modelo matemático (vermelho) aos dados históricos (barras).")
            with st.expander("Entenda este gráfico"):
                st.markdown("""
                - **O que este gráfico mostra?** Ele funciona como uma "régua" para medir a intensidade de uma seca. A curva vermelha representa a probabilidade de ocorrerem secas de diferentes durações.
                - **O que isso significa para você?** A cauda longa da curva para a direita mostra que secas extremamente severas, embora raras, são uma possibilidade real na sua região.
                - **Como isso afeta seu seguro?** O modelo GEV nos permite calcular com precisão a probabilidade de um evento (ex: uma seca de 40 dias). É com base nesse cálculo que definimos o "Período de Retorno" (1 em 10 anos, 1 em 20 anos) que você seleciona no simulador.
                """)
        if st.session_state.get('fig_corner_gev'):
            with st.expander("Ver detalhes técnicos do ajuste do modelo GEV (Corner Plot)"):
                st.pyplot(st.session_state.fig_corner_gev)
                st.markdown("""
                - **O que este gráfico mostra?** As "silhuetas" dos possíveis valores para os parâmetros do modelo (µ, s, ?) e como eles se relacionam entre si.
                - **O que isso significa para você?** Gráficos com picos bem definidos (como na diagonal) indicam que o modelo está "confiante" sobre as características do risco climático da sua região.
                - **Como isso afeta seu seguro?** A confiança nos parâmetros garante que o cálculo de risco (e o preço do seguro) é estável e não baseado em um palpite estatístico.
                """)

    with tab4:
        st.subheader("Modelo de Perda (Regressão Logística)")
        if st.session_state.get('fig_logistico_ajuste'):
            st.pyplot(st.session_state.fig_logistico_ajuste)
            st.caption("A curva mostra como a probabilidade de quebra de safra aumenta com o risco.")
            with st.expander("Entenda este gráfico"):
                st.markdown("""
                - **O que este gráfico mostra?** Ele conecta a probabilidade de uma seca (calculada pelo modelo GEV) com a chance real de haver uma quebra de safra.
                - **O que isso significa para você?** A inclinação da curva vermelha mostra o quão sensível sua safra é a eventos de seca. Uma curva mais íngreme significa que mesmo secas moderadas já aumentam drasticamente a chance de perda.
                - **Como isso afeta seu seguro?** Este modelo é o coração do seguro. Ele garante que o gatilho da apólice está diretamente ligado ao risco real do seu negócio.
                """)
        if st.session_state.get('fig_corner_logistico'):
            with st.expander("Ver detalhes técnicos do ajuste do Modelo de Perda (Corner Plot)"):
                st.pyplot(st.session_state.fig_corner_logistico)
                st.markdown("""
                - **O que este gráfico mostra?** As "silhuetas" dos possíveis valores para os parâmetros do Modelo de Perda (ß0, ß1) que conectam o risco à quebra de safra.
                - **O que isso significa para você?** Picos bem definidos indicam que o modelo tem alta confiança na relação que ele encontrou entre a seca e a perda de produtividade.
                - **Como isso afeta seu seguro?** Garante que a "sensibilidade" da sua apólice ao risco climático foi modelada com precisão, resultando em um gatilho que reflete fielmente o risco real da sua lavoura.
                """)
else:
    st.info("Aguardando carregamento dos dados meteorológicos ou seleção de localização para iniciar a análise.")
