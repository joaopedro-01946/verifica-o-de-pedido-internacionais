
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dense

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "delivery_model.keras"
VECTORIZER_PATH = "vectorizer_vocab.txt"

st.set_page_config(
    page_title="Verificador de Entregas",
    page_icon="📦",
    layout="wide"
)

# =====================================================
# DATASET INICIAL
# =====================================================

dados = [
    ("Delivered", "Entregue"),
    ("Item Delivered", "Entregue"),
    ("Package Delivered", "Entregue"),
    ("Successfully Delivered", "Entregue"),
    ("Delivered to Recipient", "Entregue"),
    ("Signed by Recipient", "Entregue"),
    ("Delivery Completed", "Entregue"),

    ("In Transit", "Em trânsito"),
    ("Out for Delivery", "Em trânsito"),
    ("Shipment Received", "Em trânsito"),
    ("Arrived at Facility", "Em trânsito"),
    ("Departed Facility", "Em trânsito"),
    ("Customs Clearance", "Em trânsito"),
    ("Processing Center", "Em trânsito"),
    ("Forwarded", "Em trânsito"),
    ("Received by Carrier", "Em trânsito"),

    ("Returned to Sender", "Exceção"),
    ("Delivery Failed", "Exceção"),
    ("Address Problem", "Exceção"),
    ("Lost Package", "Exceção"),
    ("Held by Customs", "Exceção"),
    ("Delivery Exception", "Exceção"),
    ("Unable to Deliver", "Exceção"),
    ("Failed Attempt", "Exceção")
]

df = pd.DataFrame(dados, columns=["texto", "classe"])

# =====================================================
# LABELS
# =====================================================

classes = {
    "Entregue": 0,
    "Em trânsito": 1,
    "Exceção": 2
}

classes_inverso = {
    0: "Entregue",
    1: "Em trânsito",
    2: "Exceção"
}

# =====================================================
# TREINAMENTO
# =====================================================

def treinar_modelo():

    x = df["texto"].values
    y = np.array([classes[c] for c in df["classe"]])

    vectorizer = TextVectorization(
        max_tokens=5000,
        output_mode="int",
        output_sequence_length=20
    )

    vectorizer.adapt(x)

    vocab = vectorizer.get_vocabulary()

    with open(VECTORIZER_PATH, "w", encoding="utf-8") as f:
        for palavra in vocab:
            f.write(palavra + "\n")

    model = Sequential([
        vectorizer,
        Embedding(5000, 16),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dense(16, activation="relu"),
        Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x,
        y,
        epochs=50,
        verbose=0
    )

    model.save(MODEL_PATH)

    return model

# =====================================================
# CARREGAR MODELO
# =====================================================

@st.cache_resource
def carregar_modelo():

    if not os.path.exists(MODEL_PATH):
        return treinar_modelo()

    return load_model(MODEL_PATH)

modelo = carregar_modelo()

# =====================================================
# PREVISÃO
# =====================================================

def prever_status(texto):

    pred = modelo.predict(
        np.array([texto]),
        verbose=0
    )

    indice = np.argmax(pred)

    confianca = float(np.max(pred))

    return {
        "status": classes_inverso[indice],
        "confianca": round(confianca, 4),
        "motivo": f"Classificação TensorFlow baseada no texto informado."
    }

# =====================================================
# INTERFACE STREAMLIT
# =====================================================

st.title("📦 IA de Verificação de Entregas Internacionais")

st.markdown("""
Modelo TensorFlow para classificação automática de rastreamento.
""")

tracking = st.text_input(
    "Número de Rastreamento"
)

carrier = st.text_input(
    "Transportadora"
)

status = st.text_area(
    "Último Status de Rastreamento",
    placeholder="Exemplo: Delivered to Recipient"
)

if st.button("Analisar"):

    if not status.strip():
        st.warning("Digite um status.")
        st.stop()

    resultado = prever_status(status)

    resposta = {
        "tracking_number": tracking,
        "carrier": carrier,
        "status": resultado["status"],
        "confianca": resultado["confianca"],
        "motivo": resultado["motivo"]
    }

    st.success("Análise concluída")

    st.json(resposta)

    st.subheader("JSON")

    st.code(
        json.dumps(
            resposta,
            indent=4,
            ensure_ascii=False
        ),
        language="json"
    )

# =====================================================
# TESTES RÁPIDOS
# =====================================================

st.markdown("---")

st.subheader("Exemplos")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Delivered to Recipient")

with col2:
    st.info("Out for Delivery")

with col3:
    st.info("Returned to Sender")
