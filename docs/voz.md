# `src/voz.py` — Paso 6 (Parte 2, extra): Voz neuronal (Silero TTS)

## ¿Qué hace?

Lee la última letra generada (`outputs/textos_generados/ultima_generacion.txt`) y la lee en voz alta usando **Silero TTS**, un modelo de voz neuronal (no un motor "robótico" tipo Loquendo), pero que funciona **offline** una vez descargado.

No se genera música ni ritmo con IA: esto sigue siendo solo texto-a-voz, la parte extra y más simple del proyecto (Parte 2, ~10%).

> Antes usábamos `pyttsx3` (voz robótica del sistema operativo). Se reemplazó por Silero porque suena más natural y con mejor entonación en español, sin perder la ventaja de funcionar sin internet.

## Cómo funciona Silero TTS (y por qué no es `pip install silero`)

Silero **no** se instala como una librería normal de Python. Es un modelo de PyTorch que se descarga automáticamente la **primera vez** que corres el script, usando `torch.hub.load(...)` — una función que trae PyTorch para descargar modelos ya entrenados desde un repositorio de GitHub (`snakers4/silero-models`).

Después de esa primera descarga, el modelo queda guardado en caché en tu computadora (carpeta `~/.cache/torch/hub`) y las siguientes veces carga al instante, sin necesitar internet.

Dependencias reales (en `requirements.txt`):
- `torch` (ya la teníamos, por la LSTM)
- `omegaconf` — librería chiquita que Silero usa para leer su configuración

## Salida que genera

- **`outputs/audio/letra_generada.wav`** — el audio guardado (48000 Hz).
- Además intenta reproducir el audio en vivo al correr el script (si tienes `sounddevice` instalado; si no, solo avisa que uses el `.wav` guardado).

## Cómo correrlo

```bash
python src/voz.py
```

Requiere haber corrido antes `src/generar_texto.py` (necesita que exista `outputs/textos_generados/ultima_generacion.txt`).

La primera ejecución va a tardar un poco más porque descarga el modelo (~unos MB). Las siguientes son casi instantáneas.

## Configuración

- `SPEAKER = "v3_es"` — el modelo/voz de español que trae Silero.
- `LOCUTOR = "es_1"` — locutor específico dentro de ese modelo (se puede probar `es_0`, `es_2`, etc. para comparar voces).
- `SAMPLE_RATE = 48000` — calidad del audio.

## Posible mejora futura

Probar los distintos locutores (`es_0`, `es_1`, `es_2`...) para elegir el que suene mejor para las letras generadas, o instalar `sounddevice` para que la reproducción en vivo funcione siempre.
