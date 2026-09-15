"""
PROYECTO-NATHY_IA - Paso 4: Entrenamiento
--------------------------------------------
Toma las secuencias de data/processed/secuencias.pkl y entrena el modelo
GeneradorDeLetras (definido en modelo_lstm.py) para que aprenda a predecir
la siguiente palabra a partir de un contexto de 4 palabras.

Al terminar, guarda los pesos entrenados en models/modelo_lstm.pt

Soporta entrenar por tandas: si ya existe un checkpoint en RUTA_MODELO,
sigue entrenando desde ahi en vez de empezar de cero (util con datasets
grandes que no alcanzan a entrenarse completos de una sola corrida).

Uso:
    python src/entrenamiento.py                # entrena EPOCAS_POR_TANDA épocas
    python src/entrenamiento.py 50              # entrena 50 épocas en esta tanda
"""

import os
import sys
import json
import pickle
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from modelo_lstm import GeneradorDeLetras

# --- Configuracion ---
EPOCAS_TOTALES = 120      # meta total de epocas a lo largo de todas las tandas
EPOCAS_POR_TANDA = 60      # cuantas epocas entrena esta corrida si no se pasa un numero
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 0.005
RUTA_MODELO = "models/modelo_lstm.pt"
RUTA_HISTORIAL = "data/processed/historial_perdida.csv"


def registrar_perdida(epoca, perdida):
    """Agrega una fila (epoca, perdida) al historial en CSV, creando el
    archivo con encabezado si todavia no existe. Esto es lo que despues
    usamos en el notebook para graficar la curva de perdida."""
    existe = os.path.exists(RUTA_HISTORIAL)
    os.makedirs(os.path.dirname(RUTA_HISTORIAL), exist_ok=True)
    with open(RUTA_HISTORIAL, "a", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        if not existe:
            escritor.writerow(["epoca", "perdida"])
        escritor.writerow([epoca, perdida])


class DatasetLetras(Dataset):
    def __init__(self, secuencias_x, secuencias_y):
        self.x = torch.tensor(secuencias_x, dtype=torch.long)
        self.y = torch.tensor(secuencias_y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    epocas_esta_tanda = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCAS_POR_TANDA

    with open("data/processed/vocabulario.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open("data/processed/secuencias.pkl", "rb") as f:
        datos = pickle.load(f)

    tamano_vocab = len(vocab["palabra_a_indice"])
    dataset = DatasetLetras(datos["x"], datos["y"])
    cargador = DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True)

    modelo = GeneradorDeLetras(tamano_vocabulario=tamano_vocab)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)
    funcion_perdida = nn.CrossEntropyLoss()

    epoca_inicial = 0
    try:
        checkpoint = torch.load(RUTA_MODELO, map_location="cpu")
        if checkpoint.get("tamano_vocab") == tamano_vocab:
            modelo.load_state_dict(checkpoint["state_dict"])
            optimizador.load_state_dict(checkpoint["optimizador"])
            epoca_inicial = checkpoint.get("epoca", 0)
            print(f"Retomando entrenamiento desde la epoca {epoca_inicial} "
                  f"(checkpoint encontrado en '{RUTA_MODELO}')")
        else:
            print("El vocabulario cambio desde el ultimo checkpoint, empezando de cero.")
    except FileNotFoundError:
        print("No hay checkpoint previo, empezando de cero.")

    print(f"Entrenando con {len(dataset)} secuencias, vocabulario de {tamano_vocab} palabras")
    print(f"Meta total: {EPOCAS_TOTALES} epocas | esta tanda: {epocas_esta_tanda} epocas")

    modelo.train()
    epoca_final = epoca_inicial
    for i in range(1, epocas_esta_tanda + 1):
        epoca_actual = epoca_inicial + i
        perdida_total = 0.0

        for lote_x, lote_y in cargador:
            optimizador.zero_grad()
            predicciones = modelo(lote_x)
            perdida = funcion_perdida(predicciones, lote_y)
            perdida.backward()
            optimizador.step()
            perdida_total += perdida.item()

        perdida_promedio = perdida_total / len(cargador)
        epoca_final = epoca_actual
        registrar_perdida(epoca_actual, perdida_promedio)

        if epoca_actual % 10 == 0 or i == 1 or i == epocas_esta_tanda:
            print(f"  Epoca {epoca_actual:>3}/{EPOCAS_TOTALES}  -  perdida promedio: {perdida_promedio:.4f}")

        # Guardado parcial cada 10 epocas, para no perder progreso si la
        # corrida se corta antes de terminar la tanda completa
        if epoca_actual % 10 == 0:
            torch.save({
                "state_dict": modelo.state_dict(),
                "optimizador": optimizador.state_dict(),
                "tamano_vocab": tamano_vocab,
                "epoca": epoca_actual,
            }, RUTA_MODELO)

    torch.save({
        "state_dict": modelo.state_dict(),
        "optimizador": optimizador.state_dict(),
        "tamano_vocab": tamano_vocab,
        "epoca": epoca_final,
    }, RUTA_MODELO)

    print(f"\nModelo guardado en '{RUTA_MODELO}' (epoca {epoca_final}/{EPOCAS_TOTALES})")
    if epoca_final < EPOCAS_TOTALES:
        print(f"Faltan {EPOCAS_TOTALES - epoca_final} epocas. Corre de nuevo este script para continuar.")
    else:
        print("Entrenamiento completo.")


if __name__ == "__main__":
    main()
