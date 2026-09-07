# PROYECTO-NATHY_IA

Generador de letras de canciones con una red neuronal LSTM (PyTorch), con una demo extra de voz robótica (pyttsx3).


## Idea del proyecto

Le doy al sistema una o varias líneas de una canción (la "semilla") y el modelo genera una continuación de letra nueva, coherente con ese estilo.

- **Parte 1 (foco principal, ~90%):** red neuronal LSTM en PyTorch que aprende a predecir la siguiente palabra/carácter a partir del contexto anterior, entrenada con un corpus de letras de canciones.
- **Parte 2 (extra, ~10%):** leer en voz alta la letra generada con `pyttsx3` (voz robótica simple, offline). No se genera música ni ritmo con IA.

## Estructura de carpetas

```
PROYECTO-NATHY_IA/
├── data/
│   ├── raw/            # dataset original de letras, tal cual se consigue (no se edita a mano)
│   └── processed/       # texto ya limpio y tokenizado, listo para entrenar
├── notebooks/            # cuadernos .ipynb de exploración, pruebas y visualización
├── src/                  # código fuente en .py
│   ├── preprocesamiento.py   # limpieza, tokenización, vocabulario, secuencias
│   ├── modelo_lstm.py        # arquitectura de la red (Embedding -> LSTM -> salida)
│   ├── entrenamiento.py      # loop de entrenamiento (loss, optimizador, épocas)
│   ├── generar_texto.py      # genera letra nueva a partir de una semilla
│   └── voz.py                 # lee en voz alta la letra generada (pyttsx3)
├── models/                # pesos del modelo ya entrenado (.pt) — no se sube a git
├── outputs/
│   ├── textos_generados/  # letras generadas por el modelo (.txt)
│   └── audio/              # salidas de audio de la voz robótica
├── planos/                 # diagramas del proyecto (plano nathy.svg, planificación en draw.io)
├── docs/                    # informe final, notas, capturas para la entrega
├── requirements.txt         # librerías del proyecto
└── .gitignore
```

### Por qué esta separación

- **`data/raw` vs `data/processed`**: nunca se modifica el dataset original a mano; todo lo que se limpia/transforma se guarda aparte, así siempre se puede volver a empezar desde cero si algo sale mal.
- **`src/` numerado por etapa**: cada archivo corresponde a un paso del flujo (preprocesar → armar modelo → entrenar → generar → dar voz), en el mismo orden en que se explica en el informe.
- **`models/` fuera de git**: los pesos entrenados pesan mucho y no aportan como código, por eso están en `.gitignore`.
- **`outputs/` separado de `data/`**: aquí solo caen resultados generados por el modelo, nunca datos de entrada.
- **`planos/` y `docs/`**: todo lo visual y lo escrito para la sustentación queda junto, separado del código.

## Documentación por archivo

Cada script importante de `src/` tiene su propia explicación simple en `docs/`, de qué hace y cómo correrlo:

| Script | Documentación |
|---|---|
| `src/preprocesamiento.py` | [📄 docs/preprocesamiento.md](docs/preprocesamiento.md) |
| `src/modelo_lstm.py` | [📄 docs/modelo_lstm.md](docs/modelo_lstm.md) |

(se va completando a medida que se agregan `entrenamiento.py`, `generar_texto.py` y `voz.py`)

## Instalación

```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm   # o el modelo del idioma del dataset
```

## Flujo de trabajo

1. Conseguir/armar el corpus de letras → `data/raw/`
2. Preprocesar el texto → `src/preprocesamiento.py` → `data/processed/`
3. Definir la arquitectura LSTM → `src/modelo_lstm.py`
4. Entrenar → `src/entrenamiento.py` → pesos en `models/`
5. Generar letra nueva desde una semilla → `src/generar_texto.py` → `outputs/textos_generados/`
6. Leer la letra en voz alta → `src/voz.py` → `outputs/audio/`
7. Documentar y presentar → `docs/`, `planos/`
