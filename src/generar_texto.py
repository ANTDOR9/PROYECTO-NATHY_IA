"""
PROYECTO-NATHY_IA - Paso 5: Generacion de texto
---------------------------------------------------
Carga el modelo ya entrenado (models/modelo_lstm.pt) y, a partir de una
semilla escrita por el usuario, genera una continuacion de letra palabra
por palabra.

Como funciona la generacion (en simple):
    1) Tomamos las ultimas N palabras conocidas (la semilla, o lo ya generado)
    2) El modelo predice la palabra mas probable como siguiente
    3) Esa palabra se agrega al texto, y se vuelve a repetir el paso 1
       (usando ahora las ultimas N palabras, que ya incluyen la nueva)
Esto se llama generacion "autoregresiva": el modelo se alimenta de su
propia salida anterior para seguir generando.
"""

import os
import re
import json
import pickle

import torch
import torch.nn.functional as F

from modelo_lstm import GeneradorDeLetras

RUTA_MODELO = "models/modelo_lstm.pt"
RUTA_VOCAB = "data/processed/vocabulario.json"
CARPETA_SALIDA = "outputs/textos_generados"

PALABRAS_A_GENERAR = 30
TEMPERATURA = 0.8   # >1 = mas "arriesgado"/creativo, <1 = mas conservador/seguro


def limpiar_semilla(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-záéíóúñü¡!¿?.,\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def elegir_siguiente_palabra(puntajes, temperatura):
    """En vez de elegir siempre la palabra con el puntaje mas alto (lo cual
    hace que el texto se repita mucho), convertimos los puntajes en
    probabilidades (softmax) y elegimos al azar segun esas probabilidades.
    La temperatura controla que tan "arriesgada" es esa eleccion."""
    probabilidades = F.softmax(puntajes / temperatura, dim=-1)
    indice_elegido = torch.multinomial(probabilidades, num_samples=1).item()
    return indice_elegido


def generar(semilla: str, num_palabras: int = PALABRAS_A_GENERAR, temperatura: float = TEMPERATURA) -> str:
    # --- Cargar vocabulario y modelo entrenado ---
    with open(RUTA_VOCAB, encoding="utf-8") as f:
        vocab = json.load(f)
    palabra_a_indice = vocab["palabra_a_indice"]
    indice_a_palabra = {i: p for p, i in palabra_a_indice.items()}
    largo_secuencia = vocab["largo_secuencia"]

    checkpoint = torch.load(RUTA_MODELO, map_location="cpu")
    modelo = GeneradorDeLetras(tamano_vocabulario=checkpoint["tamano_vocab"])
    modelo.load_state_dict(checkpoint["state_dict"])
    modelo.eval()   # modo evaluacion: apaga cosas que solo se usan en entrenamiento

    # --- Preparar la semilla ---
    semilla_limpia = limpiar_semilla(semilla)
    tokens = semilla_limpia.split()

    # si la semilla tiene menos palabras que el largo de secuencia, rellenamos
    # al inicio con <PAD> para poder alimentar al modelo igual
    if len(tokens) < largo_secuencia:
        tokens = ["<PAD>"] * (largo_secuencia - len(tokens)) + tokens

    indices = [palabra_a_indice.get(t, palabra_a_indice["<UNK>"]) for t in tokens]

    # --- Generacion palabra por palabra ---
    generadas = []
    with torch.no_grad():
        for _ in range(num_palabras):
            contexto = torch.tensor([indices[-largo_secuencia:]], dtype=torch.long)
            puntajes = modelo(contexto)[0]   # [tamano_vocab]
            siguiente_idx = elegir_siguiente_palabra(puntajes, temperatura)
            siguiente_palabra = indice_a_palabra.get(siguiente_idx, "<UNK>")

            indices.append(siguiente_idx)
            generadas.append(siguiente_palabra)

    texto_generado = " ".join(tokens[max(0, len(tokens) - largo_secuencia):]).replace("<PAD> ", "").strip()
    texto_generado += " " + " ".join(generadas)
    return texto_generado.strip()


def main():
    print("=== Generador de letras - PROYECTO-NATHY_IA ===")
    semilla = input("Escribe una linea semilla: ").strip()
    if not semilla:
        semilla = "con manos entrelazadas avanzamos"
        print(f"(no escribiste nada, uso la semilla de ejemplo: '{semilla}')")

    resultado = generar(semilla)

    print("\n--- Letra generada ---")
    print(resultado)

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    ruta_salida = os.path.join(CARPETA_SALIDA, "ultima_generacion.txt")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(resultado)
    print(f"\nGuardado en '{ruta_salida}'")


if __name__ == "__main__":
    main()
