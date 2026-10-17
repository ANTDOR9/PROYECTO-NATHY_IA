"""
PROYECTO-NATHY_IA - Diagnostico de una letra generada
--------------------------------------------------------
Mide cosas que se pueden VER en el texto para saber si el modelo esta creando o copiando:

  - % copiado:  que parte de la letra son secuencias de 4 palabras seguidas que existen
                TAL CUAL en alguna cancion del corpus. Cerca de 100 % = memoriza.
  - variedad:   palabras distintas / palabras totales (baja = se repite mucho).
  - seguridad:  probabilidad promedio con la que eligio cada palabra.
"""

import os

from preprocesamiento import tokenizar, CARPETA_RAW

N = 4
_ngramas = None


def _ngramas_del_corpus():
    """Todas las secuencias de N palabras de TODAS las canciones de data/raw,
    incluidas las archivadas (el modelo anterior tambien aprendio de ellas)."""
    global _ngramas
    if _ngramas is None:
        _ngramas = set()
        for raiz, _, archivos in os.walk(CARPETA_RAW):
            for nombre in archivos:
                if nombre.endswith(".txt"):
                    with open(os.path.join(raiz, nombre), encoding="utf-8") as f:
                        tokens = [t for t in tokenizar(f.read()) if t != "|"]
                    _ngramas.update(tuple(tokens[i:i + N]) for i in range(len(tokens) - N + 1))
    return _ngramas


def diagnosticar(texto: str, pasos: list) -> dict:
    tokens = [t for t in tokenizar(texto) if t != "|"]
    grupos = [tuple(tokens[i:i + N]) for i in range(len(tokens) - N + 1)]
    corpus = _ngramas_del_corpus()
    copiados = sum(1 for g in grupos if g in corpus)
    return {
        "pct_copiado": 100.0 * copiados / len(grupos) if grupos else 0.0,
        "variedad": len(set(tokens)) / len(tokens) if tokens else 0.0,
        "seguridad": sum(p["prob_elegida"] for p in pasos) / len(pasos) if pasos else 0.0,
        "n_palabras": len(tokens),
    }
