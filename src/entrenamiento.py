"""
PROYECTO-NATHY_IA - Paso 4: Entrenamiento
--------------------------------------------
Entrena el modelo GeneradorDeLetras (modelo_lstm.py) para predecir la palabra siguiente
a partir de un contexto de LARGO_SECUENCIA palabras (ver data/processed/vocabulario.json).

Cada epoca mide DOS perdidas:
  - perdida de entrenamiento: sobre las canciones con las que aprende
  - perdida de validacion:    sobre canciones que NUNCA ve al entrenar
Si la de entrenamiento baja pero la de validacion sube, el modelo esta memorizando.
Se guarda el modelo de la MEJOR epoca de validacion y se frena solo (early stopping)
cuando deja de mejorar durante PACIENCIA epocas seguidas.

Uso:
    python src/entrenamiento.py            # hasta EPOCAS_MAX epocas
    python src/entrenamiento.py 30         # maximo 30 epocas
"""

import os
import sys
import csv
import json
import math
import pickle
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from modelo_lstm import GeneradorDeLetras

# --- Configuracion ---
EPOCAS_MAX = 80
PACIENCIA = 8            # epocas seguidas sin mejorar la validacion antes de frenar
TAMANO_LOTE = 32
TASA_APRENDIZAJE = 0.003
DECAIMIENTO_PESOS = 1e-4
SEMILLA = 42
RUTA_MODELO = "models/modelo_lstm.pt"
RUTA_HISTORIAL = "data/processed/historial_perdida.csv"


class DatasetLetras(Dataset):
    def __init__(self, secuencias_x, secuencias_y):
        self.x = torch.tensor(secuencias_x, dtype=torch.long)
        self.y = torch.tensor(secuencias_y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def perdida_en(modelo, cargador, funcion_perdida):
    """Perdida promedio sobre un conjunto, sin ajustar pesos."""
    modelo.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for lote_x, lote_y in cargador:
            total += funcion_perdida(modelo(lote_x), lote_y).item() * len(lote_y)
            n += len(lote_y)
    return total / max(1, n)


def main():
    epocas_max = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCAS_MAX
    torch.manual_seed(SEMILLA)

    with open("data/processed/vocabulario.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open("data/processed/secuencias.pkl", "rb") as f:
        datos = pickle.load(f)

    tamano_vocab = len(vocab["palabra_a_indice"])
    entrenamiento = DatasetLetras(datos["x"], datos["y"])
    validacion = DatasetLetras(datos["x_val"], datos["y_val"])
    cargador = DataLoader(entrenamiento, batch_size=TAMANO_LOTE, shuffle=True)
    cargador_val = DataLoader(validacion, batch_size=256)

    modelo = GeneradorDeLetras(tamano_vocabulario=tamano_vocab)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE, weight_decay=DECAIMIENTO_PESOS)
    funcion_perdida = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)

    # No pisar un modelo anterior: se guarda como copia antes de empezar
    if os.path.exists(RUTA_MODELO):
        copia = RUTA_MODELO.replace(".pt", "_anterior.pt")
        shutil.copy(RUTA_MODELO, copia)
        print(f"Modelo previo guardado como '{copia}'")

    os.makedirs(os.path.dirname(RUTA_HISTORIAL), exist_ok=True)
    with open(RUTA_HISTORIAL, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoca", "perdida", "perdida_val"])

    azar = math.log(tamano_vocab)
    print(f"Entrenando con {len(entrenamiento)} secuencias, validando con {len(validacion)}; "
          f"vocabulario de {tamano_vocab} palabras")
    print(f"Referencia: adivinar al azar daria una perdida de {azar:.2f}\n")

    mejor_val, epoca_mejor, sin_mejorar = float("inf"), 0, 0
    for epoca in range(1, epocas_max + 1):
        modelo.train()
        total = 0.0
        for lote_x, lote_y in cargador:
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(lote_x), lote_y)
            perdida.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optimizador.step()
            total += perdida.item()
        p_train = total / len(cargador)
        p_val = perdida_en(modelo, cargador_val, funcion_perdida)

        with open(RUTA_HISTORIAL, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoca, p_train, p_val])

        marca = ""
        if p_val < mejor_val:
            mejor_val, epoca_mejor, sin_mejorar = p_val, epoca, 0
            torch.save({"state_dict": modelo.state_dict(), "tamano_vocab": tamano_vocab,
                        "epoca": epoca, "perdida_val": p_val}, RUTA_MODELO)
            marca = "  <- mejor"
        else:
            sin_mejorar += 1

        print(f"  Epoca {epoca:>3}  entrenamiento: {p_train:.3f}  validacion: {p_val:.3f}  "
              f"(perplejidad {math.exp(p_val):.0f}){marca}")

        if sin_mejorar >= PACIENCIA:
            print(f"\nLa validacion no mejora hace {PACIENCIA} epocas: se frena para no memorizar.")
            break

    print(f"\nMejor modelo: epoca {epoca_mejor}, perdida de validacion {mejor_val:.3f} "
          f"(perplejidad {math.exp(mejor_val):.0f}) -> guardado en '{RUTA_MODELO}'")


if __name__ == "__main__":
    main()
