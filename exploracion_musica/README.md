# Exploración de música (Parte 3, opcional) — PROYECTO-NATHY_IA

Esta carpeta es una exploración aparte del proyecto principal (letras + voz). La idea: ver si el mismo enfoque de red neuronal (LSTM) que se usó para generar letras se puede aplicar a generar melodías simples, estilo "8-bit" / chiptune.

No es un compromiso de que esto se vaya a terminar — es material de estudio para decidir si se convierte en una Parte 3 formal del proyecto.

## Dataset usado

**NES-MDB (NES Music Database)**
Repositorio: https://github.com/chrisdonahue/nesmdb

Es un dataset armado por investigadores (Chris Donahue et al.) específicamente para entrenar modelos de generación de música, a partir de la música de juegos de la Nintendo Entertainment System (NES).

- Se descargó el archivo `nesmdb_midi.tar.gz` (formato MIDI) y `nesmdb_meta.pkl` (metadatos: nombre de canción/juego/compositor) desde los links de descarga oficiales del repositorio.
- Se extrajo localmente con `extraer_dataset.py`.

## Nota sobre derechos de autor

La música original de estas canciones pertenece a Nintendo y a los compositores de cada juego. El dataset NES-MDB es una transcripción/extracción hecha por investigadores para fines académicos y de investigación en generación de música. Por esa razón:

- Los archivos del dataset (`dataset_midi/`, `dataset_zip/`) **no se suben a este repositorio** (están en `.gitignore`) — se usan solo de forma local para entrenar/experimentar.
- Este proyecto no redistribuye la música ni las transcripciones, y no las utiliza con fines comerciales.
- Cualquier melodía generada a partir de este dataset sería un ejercicio de aprendizaje personal, no contenido para publicar o distribuir.

## Estado

Dataset descargado y extraído (5278 archivos `.mid`). Pendiente: decidir si se avanza con la tokenización de notas y el entrenamiento de una LSTM sobre estos datos.

---
[⬅ Volver al README principal](../README.md)
