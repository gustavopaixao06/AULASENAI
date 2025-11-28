import streamlit as st
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
from deepface import DeepFace
import numpy as np

# -------------------------------------------
# Conexão com o MongoDB GridFS
# -------------------------------------------
uri = "mongodb+srv://gustavopaixao086_db_user:a9AaQOXwmFVP6lb7@cluster0.qtihjlg.mongodb.net/?appName=Cluster0"
client = MongoClient(uri)
db = client['midias']
fs = gridfs.GridFS(db)

st.title("Reconhecimento Facial – Encontre a Pessoa Mais Parecida")
st.write("Base: Imagens armazenadas no MongoDB GridFS")

# -------------------------------------------
# Carrega todas as imagens e gera embeddings
# (Faz isso somente uma vez por sessão)
# -------------------------------------------
@st.cache_resource
def carregar_base_gridfs():
    base = []
    arquivos = list(fs.find())

    if not arquivos:
        return []

    for arquivo in arquivos:
        dados = arquivo.read()
        try:
            imagem = Image.open(io.BytesIO(dados))
            embedding = DeepFace.represent(
                img_path=np.array(imagem),
                model_name="Facenet512"
            )[0]["embedding"]

            base.append({
                "nome": arquivo.filename,
                "dados": dados,
                "embedding": np.array(embedding)
            })
        except Exception as e:
            print("Erro ao processar imagem:", arquivo.filename, e)

    return base

st.write("⏳ Carregando e processando imagens do banco...")
base_emb = carregar_base_gridfs()
st.write(f"📁 Total de imagens carregadas: {len(base_emb)}")


# -------------------------------------------
# Upload da foto do usuário
# -------------------------------------------
st.subheader("Envie uma foto para encontrar a pessoa mais parecida:")

foto_enviada = st.file_uploader("Selecione uma foto", type=["jpg", "jpeg", "png"])

if foto_enviada is not None:
    imagem_usuario = Image.open(foto_enviada)
    st.image(imagem_usuario, caption="Sua foto enviada", width=300)

    st.write("🔍 **Analisando rosto...**")

    # Criar embedding da foto enviada
    embedding_usuario = DeepFace.represent(
        img_path=np.array(imagem_usuario),
        model_name="Facenet512"
    )[0]["embedding"]
    embedding_usuario = np.array(embedding_usuario)

    # -------------------------------------------
    # Comparar com as imagens armazenadas
    # -------------------------------------------
    menor_dist = 9999
    mais_parecida = None

    for pessoa in base_emb:
        dist = np.linalg.norm(pessoa["embedding"] - embedding_usuario)

        if dist < menor_dist:
            menor_dist = dist
            mais_parecida = pessoa

    # -------------------------------------------
    # Mostrar resultado
    # -------------------------------------------
    st.subheader("Pessoa mais parecida encontrada no banco:")
    st.image(mais_parecida["dados"], caption=mais_parecida["nome"], width=300)
    st.write(f"📌 Distância (similaridade): **{menor_dist:.4f}**")


# -------------------------------------------
# Mostrar todas as imagens do GridFS
# -------------------------------------------
st.subheader("📸 Todas as imagens armazenadas no banco:")

arquivos = list(fs.find())

if not arquivos:
    st.warning("Nenhuma imagem encontrada no GridFS.")
else:
    cols = st.columns(3)
    for i, arquivo in enumerate(arquivos):
        dados = arquivo.read()
        imagem = Image.open(io.BytesIO(dados))

        with cols[i % 3]:
            st.image(imagem, caption=arquivo.filename, use_container_width=True)
            st.download_button(
                label="Baixar",
                data=dados,
                file_name=arquivo.filename,
                mime="image/jpeg"
            )
