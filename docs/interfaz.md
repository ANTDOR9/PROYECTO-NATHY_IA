# `src/interfaz.py` — Interfaz gráfica simple (beta)

## ¿Qué hace?

Une `generar_texto.py` + `voz.py` en una ventanita con `tkinter` (incluido con Python, no requiere instalar nada más), para no tener que usar la consola.

## Flujo de uso

1. Escribes una línea semilla en el campo de texto.
2. Presionas **"🎵 Generar letra"** → se muestra el texto generado en el cuadro de abajo (y se guarda en `outputs/textos_generados/ultima_generacion.txt`, igual que antes).
3. Se activa **"🔊 Escuchar"** → al presionarlo, lee la letra en voz alta con `pyttsx3` y guarda el audio en `outputs/audio/letra_generada.wav`.

## Detalles técnicos

- La generación y la voz corren en un **hilo aparte** (`threading`) para que la ventana no se "congele" mientras el modelo trabaja.
- Reutiliza directamente las funciones `generar()` de `generar_texto.py` y `leer_en_voz_alta()` de `voz.py` — no duplica lógica, solo las conecta a botones.

## Cómo correrlo

```bash
python src/interfaz.py
```

Requiere que ya exista `models/modelo_lstm.pt` (o sea, haber corrido antes `entrenamiento.py` al menos una vez).

## Estado

Marcado como **"beta"**: cumple el flujo básico (generar + escuchar) pero es simple a propósito — sin manejo de configuración avanzada (temperatura, largo de generación, etc.) desde la ventana. Se puede mejorar más adelante si da tiempo.
