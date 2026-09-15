"""
PROYECTO-NATHY_IA - Exploracion (Parte 3, opcional): extraer el dataset NES-MDB
--------------------------------------------------------------------------------
Este script NO es parte del pipeline principal de letras. Es un script aparte
para descomprimir el dataset de melodias NES-MDB (formato MIDI) que descargamos
para explorar si mas adelante armamos una "Parte 3" de generacion de musica.

Que hace un .tar.gz (para que quede documentado):
  Es como un .zip, pero hecho en dos pasos:
    1) "tar" empaqueta muchos archivos en uno solo (sin comprimir)
    2) "gzip" comprime ese paquete
  Por eso el nombre trae las dos partes: .tar.gz

Como usarlo:
  1) Descarga 'nesmdb_midi.tar.gz' desde el repositorio de NES-MDB
     (https://github.com/chrisdonahue/nesmdb)
  2) Colocalo dentro de: exploracion_musica/dataset_zip/
  3) Corre: python extraer_dataset.py
  4) Los archivos .mid van a quedar en: exploracion_musica/dataset_midi/
"""

import os
import tarfile

CARPETA_ZIP = os.path.join("exploracion_musica", "dataset_zip")
CARPETA_SALIDA = os.path.join("exploracion_musica", "dataset_midi")
NOMBRE_ARCHIVO = "nesmdb_midi.tar.gz"


def main():
    ruta_zip = os.path.join(CARPETA_ZIP, NOMBRE_ARCHIVO)

    if not os.path.exists(ruta_zip):
        print(f"No encontre '{ruta_zip}'.")
        print(f"Descarga 'nesmdb_midi.tar.gz' y colocalo en '{CARPETA_ZIP}/' antes de correr este script.")
        return

    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print(f"Extrayendo '{ruta_zip}'...")
    with tarfile.open(ruta_zip) as archivo:
        archivo.extractall(CARPETA_SALIDA)

    # Contamos cuantos archivos .mid quedaron (pueden estar dentro de subcarpetas
    # como train/, valid/, test/, segun como vengan empaquetados)
    total_midis = 0
    for raiz, _, archivos in os.walk(CARPETA_SALIDA):
        total_midis += sum(1 for a in archivos if a.endswith(".mid"))

    print(f"\nListo. Se extrajeron {total_midis} archivos .mid en '{CARPETA_SALIDA}/'")


if __name__ == "__main__":
    main()
