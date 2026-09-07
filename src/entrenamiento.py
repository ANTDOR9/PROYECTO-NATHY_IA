"""
PROYECTO-NATHY_IA - Paso 4: Entrenamiento
--------------------------------------------
Toma las secuencias de data/processed/secuencias.pkl y entrena el modelo
GeneradorDeLetras (definido en modelo_lstm.py) para que aprenda a predecir
la siguiente palabra a partir de un contexto de 4 palabras.

Al terminar, guarda los pesos entrenados en models/modelo_lstm.pt
"""

import json
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from modelo_lstm import GeneradorDeLetras

# --- Configuracion ---
EPOCAS = 100          # cuantas veces recorre TODO el dataset
TAMANO_LOTE = 32      # cuantas secuencias procesa junto antes de ajustar pesos
TASA_APRENDIZAJE = 0.005
RUTA_MODELO = "models/modelo_lstm.pt"


class DatasetLetras(Dataset):
    """Envuelve las secuencias en el formato que PyTorch necesita para
    entrenar por lotes (batches)."""

    def __init__(self, secuencias_x, secuencias_y):
        self.x = torch.tensor(secuencias_x, dtype=torch.long)
        self.y = torch.tensor(secuencias_y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def main():
    # --- Cargar datos ya preparados en el Paso 2 ---
    with open("data/processed/vocabulario.json", encoding="utf-8") as f:
        vocab = json.load(f)
    with open("data/processed/secuencias.pkl", "rb") as f:
        datos = pickle.load(f)

    tamano_vocab = len(vocab["palabra_a_indice"])
    dataset = DatasetLetras(datos["x"], datos["y"])
    cargador = DataLoader(dataset, batch_size=TAMANO_LOTE, shuffle=True)

    print(f"Entrenando con {len(dataset)} secuencias, vocabulario de {tamano_vocab} palabras")

    # --- Modelo, funcion de perdida y optimizador ---
    modelo = GeneradorDeLetras(tamano_vocabulario=tamano_vocab)

    # CrossEntropyLoss: mide que tan lejos estuvo la prediccion de la
    # palabra correcta (igual que en tus clasificadores con PyTorch)
    funcion_perdida = nn.CrossEntropyLoss()

    # Adam: el mismo optimizador que ya usaste, ajusta los pesos de a poco
    # en la direccion que reduce el error
    optimizador = torch.optim.Adam(modelo.parameters(), lr=TASA_APRENDIZAJE)

    # --- Loop de entrenamiento ---
    modelo.train()
    for epoca in range(1, EPOCAS + 1):
        perdida_total = 0.0

        for lote_x, lote_y in cargador:
            optimizador.zero_grad()               # limpiar gradientes del paso anterior
            predicciones = modelo(lote_x)          # forward: el modelo predice
            perdida = funcion_perdida(predicciones, lote_y)  # que tan mal predijo
            perdida.backward()                     # backward: calcula como ajustar cada peso
            optimizador.step()                     # aplica el ajuste (regla delta, version PyTorch)

            perdida_total += perdida.item()

        perdida_promedio = perdida_total / len(cargador)

        if epoca % 10 == 0 or epoca == 1:
            print(f"  Epoca {epoca:>3}/{EPOCAS}  -  perdida promedio: {perdida_promedio:.4f}")

    # --- Guardar el modelo entrenado ---
    torch.save({
        "state_dict": modelo.state_dict(),
        "tamano_vocab": tamano_vocab,
    }, RUTA_MODELO)

    print(f"\nModelo entrenado guardado en '{RUTA_MODELO}'")


if __name__ == "__main__":
    main()
