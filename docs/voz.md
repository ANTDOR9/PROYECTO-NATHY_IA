# `src/voz.py` — Paso 6 (Parte 2, extra): Voz robótica

## ¿Qué hace?

Lee la última letra generada (`outputs/textos_generados/ultima_generacion.txt`) y la lee en voz alta con `pyttsx3` — una voz robótica simple tipo "Loquendo", que funciona **offline**.

No se genera música ni ritmo con IA: esto es solo texto-a-voz, la parte extra y más simple del proyecto (Parte 2, ~10%).

## Cómo funciona `pyttsx3`

Usa el motor de voz que ya trae el sistema operativo — no descarga nada de internet:

- **Windows** → SAPI5 (ya viene instalado)
- **Linux** → espeak (hay que instalarlo aparte: `sudo apt install espeak`)
- **macOS** → NSSpeechSynthesizer

El script intenta buscar automáticamente una voz en español entre las instaladas en tu sistema; si no encuentra ninguna, usa la voz por defecto (probablemente en inglés).

## Salida que genera

- **`outputs/audio/letra_generada.wav`** — el audio guardado.
- Además reproduce el audio en vivo al correr el script.

## Cómo correrlo

```bash
python src/voz.py
```

Requiere haber corrido antes `src/generar_texto.py` (necesita que exista `outputs/textos_generados/ultima_generacion.txt`).

## Configuración

- `VELOCIDAD = 150` — palabras por minuto (más bajo = más claro/lento).
- `VOLUMEN = 1.0` — de 0.0 a 1.0.

## Posible mejora futura

Ya lo dejaste anotado en el prompt original del proyecto: una voz más elaborada o con ritmo. Por ahora el objetivo es que simplemente funcione de forma básica.
