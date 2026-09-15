"""
PROYECTO-NATHY_IA - Paso 6 (Parte 2, extra): Voz neuronal con Silero TTS
------------------------------------------------------------------------
Lee la ultima letra generada (outputs/textos_generados/ultima_generacion.txt)
y la lee en voz alta con Silero TTS: un modelo de voz neuronal (no un motor
"robotico" tipo Loquendo como pyttsx3), pero que sigue funcionando 100%
offline una vez descargado.

Que es Silero TTS y como se instala (importante, no es "pip install silero"):
  Silero NO se instala como una libreria de Python tradicional. Es un modelo
  de PyTorch que se descarga automaticamente la PRIMERA vez que corres este
  script, usando torch.hub (una funcion que trae PyTorch para descargar
  modelos ya entrenados desde un repositorio de GitHub). Despues de la
  primera descarga, el modelo queda guardado en cache en tu computadora
  (carpeta ~/.cache/torch/hub) y las siguientes veces carga al instante,
  sin necesitar internet.

  Requisitos (van en requirements.txt): torch (ya lo tenias) y omegaconf
  (una libreria chiquita que Silero usa para leer su archivo de configuracion).

No se genera musica ni ritmo con IA: esto sigue siendo solo texto-a-voz, la
parte "extra" y mas simple del proyecto (Parte 2).
"""

import os

import torch

RUTA_TEXTO = "outputs/textos_generados/ultima_generacion.txt"
CARPETA_AUDIO = "outputs/audio"
NOMBRE_AUDIO = "letra_generada.wav"

IDIOMA = "es"
SPEAKER = "v3_es"       # voz/modelo de espanol que trae Silero
SAMPLE_RATE = 48000     # calidad del audio (48000 = alta calidad)
LOCUTOR = "es_1"        # voz especifica dentro del modelo v3_es (probar es_0, es_1, es_2...)

# Cache del modelo a nivel de modulo: si generamos varias veces en la misma
# ejecucion (ej. desde la interfaz grafica), no lo volvemos a cargar cada vez.
_modelo_cache = None


def _cargar_modelo():
    global _modelo_cache
    if _modelo_cache is None:
        print("Cargando modelo de voz Silero (la primera vez puede tardar, se descarga)...")
        modelo, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=IDIOMA,
            speaker=SPEAKER,
        )
        _modelo_cache = modelo
    return _modelo_cache


def leer_en_voz_alta(texto: str, guardar_audio: bool = True):
    modelo = _cargar_modelo()

    if guardar_audio:
        os.makedirs(CARPETA_AUDIO, exist_ok=True)
        ruta_audio = os.path.join(CARPETA_AUDIO, NOMBRE_AUDIO)
        modelo.save_wav(
            text=texto,
            speaker=LOCUTOR,
            sample_rate=SAMPLE_RATE,
            audio_path=ruta_audio,
        )
        print(f"Audio guardado en '{ruta_audio}'")
    else:
        # Silero no tiene un "hablar en vivo" incorporado como pyttsx3.
        # Generamos el audio en memoria y lo reproducimos con simpleaudio/sounddevice
        # si esta disponible; si no, avisamos que use el archivo guardado.
        audio = modelo.apply_tts(
            text=texto,
            speaker=LOCUTOR,
            sample_rate=SAMPLE_RATE,
        )
        try:
            import sounddevice as sd
            sd.play(audio.numpy(), SAMPLE_RATE)
            sd.wait()
        except ImportError:
            print("(No esta instalado 'sounddevice' para reproducir en vivo; "
                  "usa el audio ya guardado en outputs/audio/)")


def main():
    if not os.path.exists(RUTA_TEXTO):
        print(f"No encontre '{RUTA_TEXTO}'. Corre primero src/generar_texto.py")
        return

    with open(RUTA_TEXTO, encoding="utf-8") as f:
        texto = f.read().strip()

    print("=== Voz neuronal (Silero TTS) - PROYECTO-NATHY_IA ===")
    print(f"Texto a leer:\n  {texto}\n")

    leer_en_voz_alta(texto, guardar_audio=True)

    # tambien intenta reproducirla en vivo, ademas de guardarla
    leer_en_voz_alta(texto, guardar_audio=False)


if __name__ == "__main__":
    main()
