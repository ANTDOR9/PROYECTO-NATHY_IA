"""
PROYECTO-NATHY_IA - Visualizacion: curva de perdida del entrenamiento
----------------------------------------------------------------------
Lee data/processed/historial_perdida.csv (generado por src/entrenamiento.py)
y dibuja como fue bajando (o estabilizandose) la perdida a lo largo de las
epocas. Sirve para el reporte de avance: muestra visualmente si el modelo
sigue aprendiendo o si ya "convergio" (dejo de mejorar).

Ojo: este historial solo tiene las epocas que se entrenaron DESPUES de
agregar el registro en entrenamiento.py (desde la epoca 101 en adelante).
Las primeras 100 epocas no quedaron guardadas porque en ese momento el
script todavia no guardaba el historial en un archivo.

Como correrlo:
    python notebooks/graficar_perdida.py

Genera: notebooks/curva_perdida.png
"""

import csv
import os

import matplotlib.pyplot as plt

RUTA_CSV = os.path.join("data", "processed", "historial_perdida.csv")
RUTA_IMAGEN = os.path.join("notebooks", "curva_perdida.png")


def main():
    if not os.path.exists(RUTA_CSV):
        print(f"No encontre '{RUTA_CSV}'. Corre primero src/entrenamiento.py "
              "para generar el historial.")
        return

    epocas, perdidas, perdidas_val = [], [], []
    with open(RUTA_CSV, encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            epocas.append(int(fila["epoca"]))
            perdidas.append(float(fila["perdida"]))
            if fila.get("perdida_val"):
                perdidas_val.append(float(fila["perdida_val"]))

    print(f"Leidas {len(epocas)} epocas del historial (de la {epocas[0]} a la {epocas[-1]})")

    plt.figure(figsize=(9, 5))
    plt.plot(epocas, perdidas, marker="o", color="#4a7c59", linewidth=2, markersize=4,
             label="Entrenamiento (canciones que ve)")
    if perdidas_val:
        plt.plot(epocas, perdidas_val, marker="o", color="#c0562f", linewidth=2, markersize=4,
                 label="Validacion (canciones que NO ve)")
        mejor = min(range(len(perdidas_val)), key=perdidas_val.__getitem__)
        plt.axvline(epocas[mejor], color="gray", linestyle="--", alpha=0.6)
        plt.annotate(f"mejor modelo (epoca {epocas[mejor]})", (epocas[mejor], perdidas_val[mejor]),
                     textcoords="offset points", xytext=(8, 10), fontsize=9)
    plt.title("Curva de perdida - PROYECTO-NATHY_IA (LSTM)")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida promedio (CrossEntropyLoss)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUTA_IMAGEN, dpi=150)

    print(f"Grafico guardado en '{RUTA_IMAGEN}'")


if __name__ == "__main__":
    main()
