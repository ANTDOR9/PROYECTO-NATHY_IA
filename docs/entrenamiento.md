# `src/entrenamiento.py` — Paso 4: Entrenamiento

## ¿Qué hace?

Toma las secuencias de `data/processed/secuencias.pkl` y entrena el modelo `GeneradorDeLetras` (de `modelo_lstm.py`) para que aprenda a predecir la siguiente palabra a partir de un contexto de 4 palabras.

## El loop, en simple

Por cada lote de secuencias, en cada época:

1. El modelo **predice** la palabra siguiente (`forward`).
2. Se compara la predicción con la palabra correcta usando `CrossEntropyLoss` — es el mismo concepto que `error = Target − f(Z)` del perceptrón, pero para las 614 palabras del vocabulario a la vez.
3. `.backward()` calcula cómo ajustar cada peso de la red para reducir ese error.
4. `optimizador.step()` (Adam) aplica el ajuste — la versión más sofisticada de la regla delta (`W_nuevo = W_viejo + ajuste`).

## Qué mirar mientras entrena

La **pérdida promedio** por época debería ir bajando. Con las 9 canciones de prueba bajó de ~5.18 (adivinando al azar) a ~0.07 en 100 épocas.

⚠️ Con un dataset tan chico, una pérdida muy baja puede significar que el modelo **memorizó** las letras casi literales en vez de generalizar un estilo. Eso se nota en el siguiente paso, al generar texto con una semilla nueva.

## Salida que genera

- **`models/modelo_lstm.pt`** — los pesos ya entrenados del modelo, listos para usarse en `generar_texto.py`.

## Cómo correrlo

```bash
python src/entrenamiento.py
```

## Configuración

- `EPOCAS = 100` — vueltas completas al dataset.
- `TAMANO_LOTE = 32` — secuencias procesadas juntas antes de ajustar pesos.
- `TASA_APRENDIZAJE = 0.005` — qué tan grande es cada ajuste de pesos.
