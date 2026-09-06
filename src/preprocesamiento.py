"""
PROYECTO-NATHY_IA - Paso 2: Preprocesamiento de texto
------------------------------------------------------
Lee todas las canciones de data/raw/, las limpia, las tokeniza por PALABRA
(con NLTK) y arma:
  1) un vocabulario (palabra <-> numero)
  2) secuencias de entrenamiento: [palabras de contexto] -> [palabra siguiente]

Todo queda guardado en data/processed/ listo para la siguiente etapa (la LSTM).
"""

import os
import re
import json
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize

# --- Configuracion ---
CARPETA_RAW = "data/raw"
CARPETA_PROCESSED = "data/processed"
LARGO_SECUENCIA = 4   # cuantas palabras de contexto usamos para predecir la siguiente

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def limpiar_texto(texto: str) -> str:
    """Deja el texto en minusculas y solo con letras (incluye acentos/ñ),
    numeros, signos de puntuacion basicos y signos de exclamacion/interrogacion
    (los dejamos porque en las letras aportan emocion: '¡Desafiamos!')."""
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü¡!¿?.,\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def cargar_canciones(carpeta: str) -> list:
    canciones = []
    for nombre in sorted(os.listdir(carpeta)):
        if nombre.endswith(".txt"):
            ruta = os.path.join(carpeta, nombre)
            with open(ruta, "r", encoding="utf-8") as f:
                canciones.append(f.read())
    return canciones


def main():
    print(f"Leyendo canciones desde '{CARPETA_RAW}'...")
    canciones = cargar_canciones(CARPETA_RAW)
    print(f"  -> {len(canciones)} canciones encontradas")

    canciones_tokenizadas = []
    total_palabras = 0
    for texto in canciones:
        texto_limpio = limpiar_texto(texto)
        tokens = word_tokenize(texto_limpio, language="spanish")
        canciones_tokenizadas.append(tokens)
        total_palabras += len(tokens)

    print(f"  -> {total_palabras} palabras (tokens) en total")

    contador = Counter(tok for cancion in canciones_tokenizadas for tok in cancion)
    palabras_unicas = sorted(contador.keys())

    palabra_a_indice = {"<PAD>": 0, "<UNK>": 1}
    for palabra in palabras_unicas:
        palabra_a_indice[palabra] = len(palabra_a_indice)

    print(f"  -> vocabulario de {len(palabra_a_indice)} palabras distintas")

    secuencias_x = []
    secuencias_y = []
    for tokens in canciones_tokenizadas:
        indices = [palabra_a_indice.get(t, 1) for t in tokens]
        for i in range(len(indices) - LARGO_SECUENCIA):
            secuencias_x.append(indices[i:i + LARGO_SECUENCIA])
            secuencias_y.append(indices[i + LARGO_SECUENCIA])

    print(f"  -> {len(secuencias_x)} secuencias de entrenamiento generadas "
          f"(contexto de {LARGO_SECUENCIA} palabras -> 1 palabra objetivo)")

    os.makedirs(CARPETA_PROCESSED, exist_ok=True)

    with open(os.path.join(CARPETA_PROCESSED, "vocabulario.json"), "w", encoding="utf-8") as f:
        json.dump({"palabra_a_indice": palabra_a_indice,
                   "largo_secuencia": LARGO_SECUENCIA}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(CARPETA_PROCESSED, "secuencias.pkl"), "wb") as f:
        pickle.dump({"x": secuencias_x, "y": secuencias_y}, f)

    print(f"\nListo. Archivos guardados en '{CARPETA_PROCESSED}/':")
    print("  - vocabulario.json   (palabra -> numero, y al reves)")
    print("  - secuencias.pkl     (secuencias de entrenamiento en numeros)")


if __name__ == "__main__":
    main()
