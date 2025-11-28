import streamlit as st
from pymongo import MongoClient
import gridfs
from openai import OpenAI
from PIL import Image
import io
import numpy as np

# OpenAI Client
client_ai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# MongoDB
uri = st.secrets["MONGO_URI"]
client = MongoClient(uri)
db = client["midias"]
fs = gridfs.GridFS(db)

st.title("Reconhecimento Facial com Embeddings (OpenAI)")

# ----------------------------------
# Função para gerar embeddings
# ----------------------------------
def gerar_embedding(imagem_bytes):
    emb = client_ai.embeddings.create(
        model="text-embedding-3-large",
        input=imagem_bytes
    )
    return np.array(emb.data[0].embedding)

# ----------------------------------
# Carregar imagens + embeddings
# ----------------------------------
@st.cache_resource
def carregar_base():
    base = []
    for arquivo in fs.find():
        dados = arquivo.read()

        # gera embedding se nao existir no banco
        try:
            embedding = gerar_embedding(dados)
        except:
            continue

        base.append({
            "nome": arquivo.filename,
            "dados": dados,
            "embedding": embedding
        })

    return base

st.write("⏳ Carregando imagens e processando embeddings...")
base_emb = carregar_base()
st.write(f"📁 Total de imagens carregadas: {len(base_emb)}")

# ----------------------------------
# Upload da foto do usuário
# ----------------------------------
foto = st.file_uploader("Envie uma foto", type=["jpg", "jpeg", "png"])

if foto is not None:
    img = Image.open(foto)
    st.image(img, caption="Sua foto", width=300)
    img_bytes = foto.read()

    emb_user = gerar_embedding(img_bytes)

    menor_dist = 9999
    mais_parecida = None

    for pessoa in base_emb:
        dist = np.linalg.norm(pessoa["embedding"] - emb_user)
        if dist < menor_dist:
            menor_dist = dist
            mais_parecida = pessoa

    st.subheader("Pessoa mais parecida encontrada:")
    st.image(mais_parecida["dados"], caption=mais_parecida["nome"], width=300)
    st.write(f"Distância: {menor_dist:.4f}")
