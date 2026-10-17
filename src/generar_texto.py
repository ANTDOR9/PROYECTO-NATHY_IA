"""
PROYECTO-NATHY_IA - Paso 5: Generacion de texto
---------------------------------------------------
Carga un modelo ya entrenado y, a partir de una semilla escrita por el usuario,
genera una continuacion de letra palabra por palabra.

Como funciona la generacion (en simple):
    1) Tomamos las ultimas N palabras conocidas (la semilla, o lo ya generado)
    2) El modelo calcula una probabilidad para CADA palabra del vocabulario
    3) Se sortea la siguiente palabra segun esas probabilidades
       (la temperatura y top_k controlan que tan arriesgado es el sorteo)
    4) Esa palabra se agrega al texto y se repite desde el paso 1
Esto se llama generacion "autoregresiva": el modelo se alimenta de su propia salida.

Hay dos modelos disponibles para poder compararlos:
    "nuevo"    -> entrenado con validacion y early stopping (models/modelo_lstm.pt)
    "anterior" -> el primero, entrenado 120 epocas sin validacion (memorizaba)
"""

import os
import re
import json

import torch
import torch.nn.functional as F

from modelo_lstm import GeneradorDeLetras
from preprocesamiento import tokenizar

CARPETA_SALIDA = "outputs/textos_generados"

MODELOS = {
    "nuevo": {
        "titulo": "Modelo nuevo",
        "modelo": "models/modelo_lstm.pt",
        "vocab": "data/processed/vocabulario.json",
    },
    "anterior": {
        "titulo": "Modelo anterior",
        "modelo": "models/modelo_lstm_v1_120epocas.pt",
        "vocab": "models/vocabulario_v1.json",
    },
}
RUTA_MODELO = MODELOS["nuevo"]["modelo"]   # (compatibilidad con otros scripts)
RUTA_VOCAB = MODELOS["nuevo"]["vocab"]

PALABRAS_A_GENERAR = 30
TEMPERATURA = 0.8   # >1 = mas "arriesgado"/creativo, <1 = mas conservador/seguro
TOP_K = 25          # solo se sortea entre las K palabras mas probables (evita palabras sin sentido)
TOP_MOSTRADAS = 5   # cuantas candidatas se guardan por paso para el instructor

_cache = {}


def modelo_disponible(clave: str) -> bool:
    datos = MODELOS[clave]
    return os.path.exists(datos["modelo"]) and os.path.exists(datos["vocab"])


def cargar_modelo(clave: str = "nuevo"):
    """Devuelve (modelo, vocab, info). La arquitectura (tamano de embedding, de capa
    oculta, numero de capas) se deduce de los pesos guardados, asi que sirve para
    cualquier modelo entrenado con una configuracion distinta."""
    if clave in _cache:
        return _cache[clave]
    datos = MODELOS[clave]
    with open(datos["vocab"], encoding="utf-8") as f:
        vocab = json.load(f)
    checkpoint = torch.load(datos["modelo"], map_location="cpu")
    pesos = checkpoint["state_dict"]

    dim_embedding = pesos["embedding.weight"].shape[1]
    dim_oculta = pesos["lstm.weight_hh_l0"].shape[1]
    num_capas = sum(1 for k in pesos if k.startswith("lstm.weight_ih_l"))

    modelo = GeneradorDeLetras(checkpoint["tamano_vocab"], dim_embedding, dim_oculta, num_capas)
    modelo.load_state_dict(pesos)
    modelo.eval()   # modo evaluacion: apaga el dropout y otras cosas que solo se usan al entrenar

    info = {
        "clave": clave,
        "titulo": datos["titulo"],
        "epocas": checkpoint.get("epoca"),
        "perdida_val": checkpoint.get("perdida_val"),
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "vocabulario": checkpoint["tamano_vocab"],
        "dim_embedding": dim_embedding,
        "dim_oculta": dim_oculta,
    }
    _cache[clave] = (modelo, vocab, info)
    return _cache[clave]


def distribucion(puntajes, temperatura, top_k=TOP_K):
    """Convierte los puntajes del modelo en probabilidades (softmax), aplicando la
    temperatura y dejando solo las top_k palabras mas probables."""
    puntajes = puntajes / temperatura
    if top_k and top_k < puntajes.numel():
        umbral = torch.topk(puntajes, top_k).values[-1]
        puntajes = puntajes.masked_fill(puntajes < umbral, float("-inf"))
    return F.softmax(puntajes, dim=-1)


def formatear_letra(texto: str) -> str:
    """Convierte el marcador '|' de fin de verso en saltos de linea y pega la puntuacion."""
    texto = texto.replace(" | ", "\n").replace("|", "\n")
    texto = re.sub(r" +([,.!?])", r"\1", texto)
    texto = re.sub(r"([¡¿]) +", r"\1", texto)
    return "\n".join(linea.strip() for linea in texto.split("\n")).strip()


def generar_con_detalle(semilla: str, num_palabras: int = PALABRAS_A_GENERAR,
                        temperatura: float = TEMPERATURA, top_k: int = TOP_K,
                        modelo: str = "nuevo", semilla_azar=None) -> dict:
    """Genera la letra y ademas devuelve como 'penso' el modelo en cada paso.

    Devuelve un diccionario con:
        texto        la letra ya formateada
        pasos        lista; por cada palabra generada: el contexto que vio, las
                     candidatas mas probables con su probabilidad, y la elegida
        desconocidas palabras de la semilla que el modelo no conoce (<UNK>)
        info         datos del modelo usado (epocas, parametros, perdida de validacion...)
    """
    red, vocab, info = cargar_modelo(modelo)
    if semilla_azar is not None:
        torch.manual_seed(semilla_azar)

    palabra_a_indice = vocab["palabra_a_indice"]
    indice_a_palabra = {i: p for p, i in palabra_a_indice.items()}
    largo = vocab["largo_secuencia"]
    unk = palabra_a_indice["<UNK>"]

    tokens = tokenizar(semilla)
    desconocidas = [t for t in tokens if t not in palabra_a_indice]
    if len(tokens) < largo:   # relleno al inicio para poder alimentar al modelo igual
        tokens = ["<PAD>"] * (largo - len(tokens)) + tokens
    indices = [palabra_a_indice.get(t, unk) for t in tokens]

    generadas, pasos = [], []
    with torch.no_grad():
        for _ in range(num_palabras):
            ventana = indices[-largo:]
            contexto = [indice_a_palabra.get(i, "<UNK>") for i in ventana]
            probs = distribucion(red(torch.tensor([ventana], dtype=torch.long))[0], temperatura, top_k)
            idx = torch.multinomial(probs, num_samples=1).item()
            mejores = torch.topk(probs, TOP_MOSTRADAS)
            pasos.append({
                "contexto": contexto,
                "candidatas": [(indice_a_palabra.get(i.item(), "<UNK>"), p.item())
                               for p, i in zip(mejores.values, mejores.indices)],
                "elegida": indice_a_palabra.get(idx, "<UNK>"),
                "prob_elegida": probs[idx].item(),
            })
            indices.append(idx)
            generadas.append(indice_a_palabra.get(idx, "<UNK>"))

    ultimos = [t for t in tokens[-largo:] if t != "<PAD>"]
    texto = formatear_letra(" ".join(ultimos + generadas))
    return {"texto": texto, "pasos": pasos, "desconocidas": desconocidas, "info": info}


def generar(semilla: str, num_palabras: int = PALABRAS_A_GENERAR, temperatura: float = TEMPERATURA,
            modelo: str = "nuevo") -> str:
    """Version simple: solo devuelve la letra."""
    return generar_con_detalle(semilla, num_palabras, temperatura, modelo=modelo)["texto"]


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
