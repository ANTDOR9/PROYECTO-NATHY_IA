# `src/preprocesamiento.py` — Paso 2: Preprocesamiento de texto

## ¿Qué hace?

Lee todas las canciones de `data/raw/`, las limpia, las tokeniza por **palabra** (con NLTK) y arma:

1. Un **vocabulario**: cada palabra distinta que aparece en tus canciones recibe un número único (ID). La red neuronal no entiende texto, solo números.
2. **Secuencias de entrenamiento**: cada canción se parte en ventanas de 4 palabras seguidas + la palabra que viene después.

## Ejemplo real (de tus propias letras)

```
entrada (contexto): nunca ha sido mi
objetivo (a predecir): fascinación
```

Es el mismo patrón "entrada → salida esperada" que ya usaste con el perceptrón, solo que aquí la entrada son 4 palabras en vez de una fila de tabla de verdad.

## Salidas que genera

Guardadas en `data/processed/`:

- **`vocabulario.json`** — el diccionario palabra ↔ número, más el largo de secuencia usado.
- **`secuencias.pkl`** — todas las secuencias de entrenamiento, ya convertidas a números.

## Formato de nombres de archivo en `data/raw/`

El script no depende del nombre del archivo (lee cualquier `.txt` que
encuentre en la carpeta), pero para mantener orden se usan estos dos
patrones:

- **Canciones reales**: `Artista_-_Titulo.txt`
  (ejemplo: `Aleks Syntek_-_Duele el amor.txt`)
- **Letras originales escritas para este proyecto (sin artista real)**:
  `IA_Original_-_Titulo.txt`
  (ejemplo: `IA_Original_-_Somos Campeones.txt`)

Esta segunda variante identifica las letras que no pertenecen a ningun
artista existente (por ejemplo las de tematica de victoria/triunfo
agregadas como dataset adicional), para diferenciarlas de un vistazo de
las canciones reales con derechos de autor.

## Cómo correrlo

```bash
python src/preprocesamiento.py
```

## Configuración

- `LARGO_SECUENCIA = 4` → cuántas palabras de contexto usa para predecir la siguiente. Se puede subir/bajar si el resultado final no es coherente.
