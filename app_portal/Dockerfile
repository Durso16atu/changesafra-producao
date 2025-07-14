# 1. Use uma imagem base oficial do Python
FROM python:3.10-slim

# 2. Defina o diretório de trabalho
WORKDIR /app

# 3. Copie o arquivo de dependências
COPY requirements.txt ./requirements.txt

# 4. Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copie o código do aplicativo
COPY . .

# 6. Exponha a porta que será fornecida pelo Cloud Run
EXPOSE 8080

# 7. Defina o comando para executar o aplicativo (SINTAXE CORRIGIDA)
CMD exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
