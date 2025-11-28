import streamlit as st
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import numpy as np
import face_recognition

# -------------------------------------------
# Conexão com o MongoDB GridFS
# -------------------------------------------
uri = "mongodb+srv://gustavopaixao086_db_user:a9AaQOXwmFVP6lb7@cluster0.qtihjlg.mongodb.net/?appName=Cluster0"
client = MongoClient(uri)
db = client["midias"]
fs = gridfs.GridFS(db)

st.title("Reconhecimento Facial – Pessoa Mais Parecida")
st.write("Base: Imagens armazenadas no MongoDB GridFS")

# -------------------------------------------
# Carregar imagens + criar embeddings
# -------------------------------------------
@st.cache_resource
def carregar_base_gridfs():
    base = []
    arquivos = list(fs.find())

    for arquivo in arquivos:
        dados = arquivo.read()
        img = face_recognition.load_image_file(io.BytesIO(dados))
        face_locations = face_recognition.face_locations(img)

        if len(face_locations) == 0:
            continue

        encoding = face_recognition.face_encodings(img, known_face_locations=face_locations)[0]

        base.append({
            "nome": arquivo.filename,
            "dados": dados,
            "encoding": encoding
        })

    return base

st.write("⏳ Processando imagens do banco...")
base_emb = carregar_base_gridfs()
st.write(f"📁 Total de imagens com rosto encontrado: **{len(base_emb)}**")

# -------------------------------------------
# Upload da foto do usuário
# -------------------------------------------
st.subheader("Envie uma foto para encontrar a pessoa mais parecida")
foto = st.file_uploader("Selecione uma foto", type=["jpg", "jpeg", "png"])

if foto is not None:
    imagem = Image.open(foto)
    st.image(imagem, caption="Sua foto", width=300)

    # Converter para array
    img_array = np.array(imagem)

    # Localizar rosto na imagem enviada
    face_locations = face_recognition.face_locations(img_array)

    if len(face_locations) == 0:
        st.error("Nenhum rosto detectado na sua imagem.")
    else:
        usuario_encoding = face_recognition.face_encodings(img_array, known_face_locations=face_locations)[0]

        # Encontrar imagem mais parecida
        menor_dist = 9999
        mais_parecida = None

        for pessoa in base_emb:
            dist = np.linalg.norm(pessoa["encoding"] - usuario_encoding)
            if dist < menor_dist:
                menor_dist = dist
                mais_parecida = pessoa

        st.subheader("Pessoa mais parecida encontrada:")
        st.image(mais_parecida["dados"], caption=mais_parecida["nome"], width=300)
        st.write(f"Distância (similaridade): **{menor_dist:.4f}**")

# -------------------------------------------
# Mostrar a galeria
# -------------------------------------------
st.subheader("📸 Todas as imagens armazenadas:")
arquivos = list(fs.find())
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
