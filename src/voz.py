"""
PROYECTO-NATHY_IA - Paso 6 (Parte 2, extra): Voz robotica con pyttsx3
------------------------------------------------------------------------
Lee la ultima letra generada (outputs/textos_generados/ultima_generacion.txt)
y la "canta"/lee en voz alta con una voz robotica simple, offline.

pyttsx3 usa el motor de voz que ya trae el sistema operativo:
  - Windows -> SAPI5 (no hay que instalar nada aparte)
  - Linux   -> espeak (hay que tenerlo instalado)
  - macOS   -> NSSpeechSynthesizer

No se genera musica ni ritmo con IA: esto es solo texto-a-voz, la parte
"extra" y mas simple del proyecto (Parte 2).
"""

import os

import pyttsx3

RUTA_TEXTO = "outputs/textos_generados/ultima_generacion.txt"
CARPETA_AUDIO = "outputs/audio"
NOMBRE_AUDIO = "letra_generada.wav"

VELOCIDAD = 150   # palabras por minuto (por defecto ronda 200, mas lento sale mas claro)
VOLUMEN = 1.0     # 0.0 a 1.0


def leer_en_voz_alta(texto: str, guardar_audio: bool = True):
    motor = pyttsx3.init()
    motor.setProperty("rate", VELOCIDAD)
    motor.setProperty("volume", VOLUMEN)

    # Si el sistema tiene una voz en español instalada, la usamos.
    # (esto varia por computadora; si no encuentra ninguna, usa la voz por defecto)
    for voz in motor.getProperty("voices"):
        nombre = (voz.name or "").lower()
        idioma = str(voz.languages).lower() if voz.languages else ""
        if "spanish" in nombre or "español" in nombre or "es_" in idioma or "es-" in idioma:
            motor.setProperty("voice", voz.id)
            break

    if guardar_audio:
        os.makedirs(CARPETA_AUDIO, exist_ok=True)
        ruta_audio = os.path.join(CARPETA_AUDIO, NOMBRE_AUDIO)
        motor.save_to_file(texto, ruta_audio)
        motor.runAndWait()
        print(f"Audio guardado en '{ruta_audio}'")
    else:
        motor.say(texto)
        motor.runAndWait()


def main():
    if not os.path.exists(RUTA_TEXTO):
        print(f"No encontre '{RUTA_TEXTO}'. Corre primero src/generar_texto.py")
        return

    with open(RUTA_TEXTO, encoding="utf-8") as f:
        texto = f.read().strip()

    print("=== Voz robotica - PROYECTO-NATHY_IA ===")
    print(f"Texto a leer:\n  {texto}\n")

    leer_en_voz_alta(texto, guardar_audio=True)

    # tambien la reproduce en vivo, ademas de guardarla
    leer_en_voz_alta(texto, guardar_audio=False)


if __name__ == "__main__":
    main()
