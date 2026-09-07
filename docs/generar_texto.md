# `src/generar_texto.py` — Paso 5: Generación de texto

## ¿Qué hace?

Carga el modelo ya entrenado (`models/modelo_lstm.pt`) y, a partir de una **semilla** (línea inicial que tú escribes), genera una continuación palabra por palabra.

## Cómo genera (generación "autoregresiva")

1. Toma las últimas 4 palabras conocidas (la semilla, o lo ya generado).
2. El modelo calcula un puntaje para cada palabra del vocabulario.
3. En vez de elegir siempre la de mayor puntaje (eso repite mucho el texto), se convierten los puntajes en probabilidades (`softmax`) y se elige una palabra al azar según esas probabilidades — la `TEMPERATURA` controla qué tan "arriesgada" es esa elección.
4. Esa palabra se agrega al texto, y se repite el proceso usando las últimas 4 palabras (que ya incluyen la nueva).

## Resultado real con el dataset de prueba (9 canciones)

```
Semilla: "con manos entrelazadas"
-> con manos entrelazadas avanzamos no me rendiré, no importa cuántas veces caiga...

Semilla: "nunca ha sido mi"
-> nunca ha sido mi fascinación me siento ícaro volando muy cerca del sol...
```

⚠️ **Importante:** el modelo **memorizó** casi literalmente los versos originales en vez de inventar algo nuevo — se nota porque reproduce fragmentos exactos de "Desafiando el futuro" y "Cuando eras tú". Esto es esperable con un dataset tan chico (9 canciones, ~2800 palabras): la red no tiene suficiente variedad para generalizar un "estilo" en vez de copiar. Para que genere letras realmente nuevas hace falta sumar más canciones al dataset (`data/raw/`) y volver a correr `preprocesamiento.py` + `entrenamiento.py`.

## Salida que genera

- **`outputs/textos_generados/ultima_generacion.txt`** — la última letra generada.

## Cómo correrlo

```bash
python src/generar_texto.py
```

Te pide escribir una línea semilla por consola.

## Configuración

- `PALABRAS_A_GENERAR = 30` — cuántas palabras nuevas genera.
- `TEMPERATURA = 0.8` — más alto (>1) = más "creativo"/impredecible; más bajo (<1) = más conservador, pegado a lo aprendido.
