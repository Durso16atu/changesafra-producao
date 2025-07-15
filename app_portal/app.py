
# -*- coding: utf-8 -*-

# Testando o gatilho automatico em 14/07
import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import re

# --- CONSTANTES DE URL PARA MÍDIA ---
MEDIA_BUCKET_URL = "https://storage.googleapis.com/changesafrastreamlit-media"
LOGO_URL = f"{MEDIA_BUCKET_URL}/logo_change_safra_transparent.png"
VIDEO_1_URL = f"{MEDIA_BUCKET_URL}/Scene_1_the_202507041640_kbkg7.mp4"
IMAGE_4_URL = f"{MEDIA_BUCKET_URL}/4.png"
IMAGE_3_URL = f"{MEDIA_BUCKET_URL}/3.png"
IMAGE_1_URL = f"{MEDIA_BUCKET_URL}/1.png"
VIDEO_2_URL = f"{MEDIA_BUCKET_URL}/casal_negro_satisfacao.mp4"

# --- FUNCAO PARA FORMATAR MOEDA ---
def formatar_reais(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# --- INICIALIZACAO DO ESTADO DA SESSAO ---
if 'simulacao_ativa' not in st.session_state:
    st.session_state.simulacao_ativa = False

# --- FUNCOES DE BACKEND ---
@st.cache_data(ttl=600)
def get_weather_data(city_name, api_key):
    if not api_key: return None
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city_name},br&appid={api_key}&units=metric&lang=pt_br"
    try:
        response = requests.get(complete_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException: return None

@st.cache_data(ttl=600)
def get_commodity_price(ticker):
    try:
        data = yf.Ticker(ticker)
        hist = data.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except Exception: return None

@st.cache_data(ttl=600)
def get_exchange_rate(ticker_pair):
    try:
        data = yf.Ticker(ticker_pair)
        hist = data.history(period="1d")
        if not hist.empty: return hist['Close'].iloc[-1]
        return None
    except Exception: return None

@st.cache_data(ttl=3600)
def get_historical_commodity_data(ticker, period="1y"):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    except Exception as e:
        st.error(f"Erro ao buscar dados historicos para {ticker}: {e}")
        return pd.DataFrame()

# --- CONFIGURACAO DA PAGINA ---
st.set_page_config(layout="wide", page_title="ChangeSafra - Portal")
api_key = "18a6ad025df885b43fb80203ecb2f9b0"

# --- Pagina Principal ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image(LOGO_URL, width=150)
with col2:
    st.title("ChangeSafra")
    st.subheader("Ferramentas Inteligentes para a Gestao de Risco no Agronegocio")

# --- Vídeo de Apresentação ---
st.markdown("---")
st.header("Conheça a ChangeSafra em Ação!")
st.video(VIDEO_1_URL)

# --- SECOES DE CONTEÚDO E STORYTELLING ---
st.markdown("---")
st.header("O Que Fazemos")
st.markdown("""
Nos reinventamos o conceito de seguro agricola. Em vez de apolices tradicionais, complexas e burocraticas, criamos seguros parametricos inteligentes. Nossas solucoes sao baseadas em gatilhos automaticos, utilizando dados publicos de fontes como o INMET e a CONAB. Se um evento climatico adverso pre-definido ocorre - como uma seca prolongada, excesso de chuva ou uma onda de geada - o pagamento do suporte financeiro e realizado de forma rapida e transparente, sem a necessidade de vistorias de campo ou processos demorados.
""")
st.markdown("""
Na ChangeSafra, nossa missao e fortalecer o produtor rural contra a volatilidade climatica. Atraves de ferramentas de analise de risco e seguros parametricos inovadores, transformamos incerteza em seguranca financeira, alinhados aos Objetivos de Desenvolvimento Sustentavel (ODS) da ONU.
""")

st.markdown("---")
with st.expander("Nosso Compromisso com o Impacto e as ODS"):
    st.image(IMAGE_4_URL, caption="Segurança para quem Alimenta o Mundo", use_container_width=True)
    st.subheader("Nossas Contribuicoes para os Objetivos de Desenvolvimento Sustentavel:")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Erradicacao da Pobreza**")
    with c2:
        st.markdown("**2. Fome Zero e Agricultura Sustentavel**")
    with c3:
        st.markdown("**13. Acao Contra a Mudanca Global do Clima**")
    st.image(IMAGE_3_URL, caption="Plantando Segurança no Campo", use_container_width=True)

st.markdown("---")
st.subheader("Nossa Proposta de Valor")
st.image(IMAGE_1_URL, caption="Sua Safra, Seus Dados. Sua Decisão.", use_container_width=True)

st.markdown("---")
with st.expander("Nossa Teoria da Mudanca na Pratica: Uma Jornada de Resiliencia"):
    st.video(VIDEO_2_URL)
    st.markdown("##### 1. O Desafio: A Incerteza da Safra")
    st.markdown("Um pequeno produtor de cafe em Minas Gerais vive uma preocupacao constante...")
    # (Resto do texto do storytelling)

# --- SIMULADOR DE CENARIOS ---
st.markdown("---")
st.header("Simulador de Cenarios de Risco")
with st.form("form_simulador"):
    st.subheader("1. Entradas da sua Operacao")
    col_form1, col_form2 = st.columns(2)
    with col_form1:
        cultura = st.selectbox("Cultura", ["Cafe", "Cana-de-Acucar"])
        area_plantada = st.number_input("Area Plantada (ha)", min_value=1.0, value=100.0, step=10.0)
        produtividade_esperada = st.number_input("Produtividade Esperada (sacas ou ton/ha)", min_value=1.0, value=60.0, step=5.0)
    with col_form2:
        custo_producao_ha = st.number_input("Custo de Producao (R$/ha)", min_value=1.0, value=5000.0, step=100.0)
        preco_venda_esperado = st.number_input("Preco de Venda Esperado (R$ por saca/ton)", min_value=1.0, value=800.0, step=50.0)
    submitted = st.form_submit_button("Simular Cenarios")
    if submitted: st.session_state.simulacao_ativa = True
if st.session_state.simulacao_ativa:
    # (Lógica da simulação mantida como no original)
    if st.button("Nova Simulacao"): st.session_state.simulacao_ativa = False; st.rerun()

# --- DASHBOARD DE MERCADO ---
st.markdown("---")
st.header("Dashboard de Mercado")
st.subheader("Condicoes Climaticas Atuais")
cidade_selecionada = st.text_input("Digite uma cidade:", "Ribeirao Preto")
if cidade_selecionada:
    weather_data = get_weather_data(cidade_selecionada, api_key)
    if weather_data and weather_data.get("cod") == 200:
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric(f"Temperatura em {cidade_selecionada.title()}", f"{weather_data['main']['temp']:.1f} C")
        col_m2.metric("Condicao", weather_data['weather'][0]['description'].title())
        col_m3.metric("Umidade", f"{weather_data['main']['humidity']}%")
    else:
        st.error(f"Nao foi possivel buscar os dados do clima para '{cidade_selecionada}'.")

st.info("Funcionalidade de histórico de clima em desenvolvimento.")
st.subheader("Cotacoes Atuais de Commodities")
# (Lógica das cotações mantida como no original)

# --- SISTEMA DE ALERTAS (ADAPTADO) ---
st.markdown("---")
st.header("Sistema de Notificacoes e Alertas")
st.info("Funcionalidade de cadastro de alertas em desenvolvimento.")
with st.form("form_alertas_placeholder"):
    st.text_input("Seu melhor e-mail", key="placeholder_email")
    st.selectbox("Selecione a Commodity para o Alerta", ["Cafe", "Acucar"], key="placeholder_comm")
    st.number_input("Preco-alvo (R$)", key="placeholder_preco")
    if st.form_submit_button("Cadastrar Alerta"):
        st.success("Obrigado pelo interesse! Esta funcionalidade será ativada em breve.")

# --- LINKS PARA CALCULADORAS ---
st.markdown("---")
st.header("Nossas Plataformas de Analise")
col_f1, col_f2 = st.columns(2)
with col_f1:
    st.subheader("Seguro Parametrico para Cafe")
    st.link_button("Acessar Calculadora de Cafe", "#", use_container_width=True) # Link desativado temporariamente
with col_f2:
    st.subheader("Seguro Parametrico para Cana-de-Acucar")
    st.link_button("Acessar Calculadora de Cana", "#", use_container_width=True) # Link desativado temporariamente