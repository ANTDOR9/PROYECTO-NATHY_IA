"""
PROYECTO-NATHY_IA - Paso 2: Preprocesamiento de texto
------------------------------------------------------
Lee todas las canciones de data/raw/ (incluye subcarpetas; ignora las que empiezan
con '_'), las limpia, las tokeniza por PALABRA y arma:
  1) un vocabulario (palabra <-> numero), construido SOLO con las canciones de entrenamiento
  2) secuencias de entrenamiento: [palabras de contexto] -> [palabra siguiente]
  3) secuencias de VALIDACION, sacadas de canciones que el modelo nunca ve al entrenar,
     para medir si de verdad generaliza o si solo memoriza.

El fin de cada verso se conserva como el token "|" (asi el modelo aprende la forma de la
letra: cuanto dura un verso, cuando termina una estrofa).

Todo queda guardado en data/processed/ listo para la siguiente etapa (la LSTM).
"""

import os
import re
import json
import pickle
from collections import Counter

# --- Configuracion ---
CARPETA_RAW = "data/raw"
CARPETA_PROCESSED = "data/processed"
LARGO_SECUENCIA = 6        # cuantas palabras de contexto usamos para predecir la siguiente
CADA_N_PARA_VALIDACION = 8  # 1 de cada 8 canciones se reserva para validar (~12 %)
TOKEN_FIN_VERSO = "|"

_PATRON_TOKEN = re.compile(r"\||\.{2,}|[¡!¿?.,]|[a-záéíóúñü]+")


def limpiar_texto(texto: str) -> str:
    """Minusculas, solo letras (con acentos/ñ), puntuacion basica y el marcador de fin de verso.
    Cada salto de linea se vuelve '|' (una linea vacia = estrofa nueva = '| |')."""
    texto = texto.lower()
    texto = texto.replace("\r", "").replace("\n", f" {TOKEN_FIN_VERSO} ")
    texto = re.sub(r"[^a-záéíóúñü¡!¿?.,|\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto: str) -> list:
    """Limpia y separa en tokens. Se usa tambien al generar, para que semilla y
    entrenamiento tokenicen exactamente igual."""
    return _PATRON_TOKEN.findall(limpiar_texto(texto))


def cargar_canciones(carpeta: str) -> list:
    """Devuelve [(nombre, texto)] de todos los .txt de 'carpeta' y sus subcarpetas
    (data/raw/romantico/reales, .../originales, ...). Las carpetas que empiezan con '_'
    (ej. _archivo) se ignoran: ahi van canciones que no entran al entrenamiento."""
    canciones = []
    for raiz, subcarpetas, archivos in os.walk(carpeta):
        subcarpetas[:] = sorted(d for d in subcarpetas if not d.startswith("_"))
        for nombre in sorted(archivos):
            if nombre.endswith(".txt"):
                with open(os.path.join(raiz, nombre), "r", encoding="utf-8") as f:
                    canciones.append((os.path.join(raiz, nombre), f.read()))
    canciones.sort(key=lambda c: c[0])
    return canciones


def armar_secuencias(indices: list):
    x, y = [], []
    for i in range(len(indices) - LARGO_SECUENCIA):
        x.append(indices[i:i + LARGO_SECUENCIA])
        y.append(indices[i + LARGO_SECUENCIA])
    return x, y


def main():
    print(f"Leyendo canciones desde '{CARPETA_RAW}'...")
    canciones = cargar_canciones(CARPETA_RAW)
    print(f"  -> {len(canciones)} canciones encontradas")

    tokenizadas = [tokenizar(texto) for _, texto in canciones]

    # Division entrenamiento / validacion POR CANCION (determinista: siempre las mismas)
    indices_val = {i for i in range(len(canciones)) if i % CADA_N_PARA_VALIDACION == CADA_N_PARA_VALIDACION - 1}
    train = [t for i, t in enumerate(tokenizadas) if i not in indices_val]
    val = [t for i, t in enumerate(tokenizadas) if i in indices_val]
    print(f"  -> {len(train)} canciones para entrenar, {len(val)} para validar")
    print(f"  -> {sum(map(len, tokenizadas))} tokens en total")

    # Vocabulario SOLO con entrenamiento: lo que aparezca solo en validacion cuenta como <UNK>
    contador = Counter(tok for cancion in train for tok in cancion)
    palabra_a_indice = {"<PAD>": 0, "<UNK>": 1}
    for palabra in sorted(contador):
        palabra_a_indice[palabra] = len(palabra_a_indice)
    print(f"  -> vocabulario de {len(palabra_a_indice)} palabras distintas")

    def a_indices(tokens):
        return [palabra_a_indice.get(t, 1) for t in tokens]

    x, y, x_val, y_val = [], [], [], []
    for tokens in train:
        a, b = armar_secuencias(a_indices(tokens)); x += a; y += b
    for tokens in val:
        a, b = armar_secuencias(a_indices(tokens)); x_val += a; y_val += b

    desconocidas = sum(1 for v in y_val if v == 1)
    print(f"  -> {len(x)} secuencias de entrenamiento y {len(x_val)} de validacion "
          f"(contexto de {LARGO_SECUENCIA} palabras -> 1 palabra objetivo)")
    print(f"  -> {desconocidas / max(1, len(y_val)):.1%} de las palabras a predecir en validacion son <UNK>")

    os.makedirs(CARPETA_PROCESSED, exist_ok=True)
    with open(os.path.join(CARPETA_PROCESSED, "vocabulario.json"), "w", encoding="utf-8") as f:
        json.dump({"palabra_a_indice": palabra_a_indice,
                   "largo_secuencia": LARGO_SECUENCIA,
                   "token_fin_verso": TOKEN_FIN_VERSO}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(CARPETA_PROCESSED, "secuencias.pkl"), "wb") as f:
        pickle.dump({"x": x, "y": y, "x_val": x_val, "y_val": y_val}, f)

    with open(os.path.join(CARPETA_PROCESSED, "canciones_validacion.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(canciones[i][0] for i in sorted(indices_val)))

    print(f"\nListo. Archivos guardados en '{CARPETA_PROCESSED}/':")
    print("  - vocabulario.json          (palabra -> numero)")
    print("  - secuencias.pkl            (secuencias de entrenamiento y validacion)")
    print("  - canciones_validacion.txt  (que canciones quedaron fuera del entrenamiento)")


if __name__ == "__main__":
    main()
