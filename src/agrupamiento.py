"""
PROYECTO-NATHY_IA - Aprendizaje NO supervisado: agrupar las canciones
-----------------------------------------------------------------------
Aqui nadie le dice al algoritmo cual es la "respuesta correcta": recibe las canciones,
las convierte en numeros y trata de encontrar grupos por su cuenta (K-Means).
Es lo opuesto a la LSTM, que SI aprende con respuestas (la palabra siguiente de cada verso).

Pasos (los mismos del ejercicio de clustering):
  1) Texto -> numeros con TF-IDF (cuanto "pesa" cada palabra en cada cancion)
  2) Normalizar con StandardScaler
  3) Reducir dimensiones con PCA (para poder graficar en 2D y estabilizar el calculo)
  4) K-Means con random_state fijo (asi el resultado se puede repetir)
  5) Elegir k: metodo del codo + Silhouette
  6) Comparar con la "realidad" que si conocemos (reales vs originales) con
     Rand ajustado e informacion mutua. OJO: ese origen NO es un estilo; solo sirve
     para ver si los grupos que el algoritmo encuentra tienen algo que ver con el.

Se puede usar desde la interfaz o por consola:  python src/agrupamiento.py
"""

import os
import re

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from preprocesamiento import cargar_canciones, tokenizar, CARPETA_RAW

SEMILLA = 42
K_MAX = 6
PALABRAS_VACIAS = set("""
a al algo aunque como con cuando de del el ella en es esa ese eso esta este esto ha han he
la las le les lo los me mi mis muy ni no nos o para pero por porque que se si sin sobre su sus
te ti tu tus un una uno y ya yo tú él más qué cómo ay oh yeh eh hey va ve vas voy vez
""".split())


def cargar_corpus():
    """Devuelve (nombres, textos_limpios, origen) de las canciones que usa el entrenamiento."""
    nombres, textos, origen = [], [], []
    for ruta, texto in cargar_canciones(CARPETA_RAW):
        partes = os.path.normpath(os.path.relpath(ruta, CARPETA_RAW)).split(os.sep)
        origen.append(partes[-2] if len(partes) >= 2 else "raw")
        nombres.append(os.path.splitext(partes[-1])[0])
        palabras = [t for t in tokenizar(texto) if re.fullmatch(r"[a-záéíóúñü]+", t)]
        textos.append(" ".join(palabras))
    return nombres, textos, origen


def analizar(k: int = 2, k_max: int = K_MAX) -> dict:
    nombres, textos, origen = cargar_corpus()
    vectorizador = TfidfVectorizer(stop_words=list(PALABRAS_VACIAS), min_df=2, max_features=1500, sublinear_tf=True)
    tfidf = vectorizador.fit_transform(textos).toarray()
    palabras = np.array(vectorizador.get_feature_names_out())

    X = StandardScaler().fit_transform(tfidf)
    n_comp = max(2, min(10, X.shape[0] - 1))
    Z = PCA(n_components=n_comp, random_state=SEMILLA).fit_transform(X)
    plano = PCA(n_components=2, random_state=SEMILLA).fit_transform(X)

    ks = list(range(2, min(k_max, len(nombres) - 1) + 1))
    inercias, siluetas = [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, n_init=10, random_state=SEMILLA).fit(Z)
        inercias.append(float(km.inertia_))
        siluetas.append(float(silhouette_score(Z, km.labels_)))

    k = max(2, min(k, ks[-1]))
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(Z)
    etiquetas = km.labels_
    clases_reales = [0 if o == "reales" else 1 for o in origen]

    grupos = []
    media_global = tfidf.mean(axis=0)
    for g in range(k):
        miembros = np.where(etiquetas == g)[0]
        pesos = tfidf[miembros].mean(axis=0) - media_global
        top = palabras[np.argsort(pesos)[::-1][:8]].tolist()
        composicion = {o: int(sum(1 for i in miembros if origen[i] == o)) for o in sorted(set(origen))}
        grupos.append({"id": g, "tamano": int(len(miembros)), "palabras": top,
                       "composicion": composicion, "ejemplos": [nombres[i] for i in miembros[:4]]})

    return {
        "k": k, "ks": ks, "inercias": inercias, "siluetas": siluetas,
        "silueta": float(silhouette_score(Z, etiquetas)),
        "rand_ajustado": float(adjusted_rand_score(clases_reales, etiquetas)),
        "info_mutua": float(normalized_mutual_info_score(clases_reales, etiquetas)),
        "plano": plano, "etiquetas": etiquetas, "origen": origen, "nombres": nombres,
        "grupos": grupos, "n_canciones": len(nombres),
    }


def main():
    r = analizar()
    print(f"{r['n_canciones']} canciones, k = {r['k']}")
    print("k:", r["ks"]); print("silhouette por k:", [round(s, 3) for s in r["siluetas"]])
    print("inercia por k:", [round(i) for i in r["inercias"]])
    print(f"Silhouette {r['silueta']:.3f} | Rand ajustado {r['rand_ajustado']:.3f} | Info. mutua {r['info_mutua']:.3f}")
    for g in r["grupos"]:
        print(f"  Grupo {g['id']}: {g['tamano']} canciones {g['composicion']} -> {', '.join(g['palabras'])}")


if __name__ == "__main__":
    main()
