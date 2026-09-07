"""
PROYECTO-NATHY_IA - Paso 3: Arquitectura de la red neuronal (LSTM)
--------------------------------------------------------------------
Define el modelo que va a aprender el patron de las letras de canciones.

Flujo dentro del modelo, para una secuencia de palabras (ya convertidas a numeros):

    [numeros de palabras] --Embedding--> [vectores] --LSTM--> [memoria] --Linear--> [puntajes por palabra del vocabulario]

Se entrena en entrenamiento.py y se usa para generar en generar_texto.py.
"""

import torch
import torch.nn as nn


class GeneradorDeLetras(nn.Module):
    def __init__(self, tamano_vocabulario, dim_embedding=64, dim_oculta=128, num_capas=1):
        super().__init__()

        # 1) Embedding: convierte cada numero-de-palabra en un vector de
        #    'dim_embedding' numeros que la red va APRENDIENDO a acomodar,
        #    de forma que palabras que se usan en contextos parecidos
        #    terminen con vectores parecidos.
        self.embedding = nn.Embedding(
            num_embeddings=tamano_vocabulario,
            embedding_dim=dim_embedding,
            padding_idx=0,   # el indice 0 es <PAD>, no aporta informacion
        )

        # 2) LSTM: recorre la secuencia de vectores palabra por palabra y va
        #    manteniendo una "memoria" (estado oculto) de lo que ha visto,
        #    para poder relacionar palabras que aparecieron antes.
        self.lstm = nn.LSTM(
            input_size=dim_embedding,
            hidden_size=dim_oculta,
            num_layers=num_capas,
            batch_first=True,   # nuestros lotes vienen como [lote, secuencia, ...]
        )

        # 3) Capa de salida: convierte la memoria final en un puntaje para
        #    CADA palabra del vocabulario ("que tan probable es que sea la
        #    siguiente"). La palabra con el puntaje mas alto es la prediccion.
        self.salida = nn.Linear(dim_oculta, tamano_vocabulario)

    def forward(self, x):
        # x: tensor de forma [lote, largo_secuencia] con numeros de palabras
        emb = self.embedding(x)                  # -> [lote, largo_secuencia, dim_embedding]
        salida_lstm, _ = self.lstm(emb)           # -> [lote, largo_secuencia, dim_oculta]
        ultima_memoria = salida_lstm[:, -1, :]    # nos quedamos con el ultimo paso de tiempo
        puntajes = self.salida(ultima_memoria)    # -> [lote, tamano_vocabulario]
        return puntajes


if __name__ == "__main__":
    # Prueba rapida con datos inventados, solo para confirmar que las
    # dimensiones cuadran (esto NO entrena nada todavia)
    import json

    with open("data/processed/vocabulario.json", encoding="utf-8") as f:
        vocab = json.load(f)

    tamano_vocab = len(vocab["palabra_a_indice"])
    largo_secuencia = vocab["largo_secuencia"]

    modelo = GeneradorDeLetras(tamano_vocabulario=tamano_vocab)
    print(modelo)

    lote_de_prueba = torch.randint(0, tamano_vocab, (2, largo_secuencia))  # 2 secuencias de ejemplo
    salida = modelo(lote_de_prueba)
    print(f"\nEntrada:  {lote_de_prueba.shape}  (2 secuencias de {largo_secuencia} palabras)")
    print(f"Salida:   {salida.shape}  (2 predicciones, una por cada palabra del vocabulario de {tamano_vocab})")
