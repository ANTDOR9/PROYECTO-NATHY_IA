"""
PROYECTO-NATHY_IA - Interfaz grafica simple (beta)
------------------------------------------------------
Une generar_texto.py + voz.py en una ventanita con tkinter (viene incluido
con Python, no requiere instalar nada mas).

Flujo:
    1) Escribes una linea semilla
    2) Presionas "Generar letra" -> se muestra el texto generado
    3) Presionas "Escuchar" -> se lee en voz alta (pyttsx3) y se guarda el audio

Se corre desde la raiz del proyecto:
    python src/interfaz.py
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

from generar_texto import generar
from voz import leer_en_voz_alta

CARPETA_SALIDA_TEXTO = "outputs/textos_generados"
RUTA_ULTIMA_GENERACION = os.path.join(CARPETA_SALIDA_TEXTO, "ultima_generacion.txt")


class AppGeneradorDeLetras:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("PROYECTO-NATHY_IA - Generador de letras")
        self.ventana.geometry("560x420")
        self.ventana.resizable(False, False)

        self.texto_generado = ""

        # --- Semilla ---
        tk.Label(ventana, text="Escribe una línea semilla:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=12, pady=(12, 2)
        )
        self.entrada_semilla = tk.Entry(ventana, font=("Segoe UI", 11), width=60)
        self.entrada_semilla.pack(padx=12, pady=(0, 8), fill="x")
        self.entrada_semilla.insert(0, "con manos entrelazadas")

        # --- Botones principales ---
        frame_botones = tk.Frame(ventana)
        frame_botones.pack(padx=12, pady=4, fill="x")

        self.boton_generar = tk.Button(
            frame_botones, text="🎵 Generar letra", command=self.al_presionar_generar,
            bg="#4a6fa5", fg="white", font=("Segoe UI", 10, "bold"), padx=10, pady=6
        )
        self.boton_generar.pack(side="left", padx=(0, 8))

        self.boton_escuchar = tk.Button(
            frame_botones, text="🔊 Escuchar", command=self.al_presionar_escuchar,
            bg="#5a9b6b", fg="white", font=("Segoe UI", 10, "bold"), padx=10, pady=6,
            state="disabled"
        )
        self.boton_escuchar.pack(side="left")

        # --- Resultado ---
        tk.Label(ventana, text="Letra generada:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", padx=12, pady=(12, 2)
        )
        self.caja_resultado = scrolledtext.ScrolledText(
            ventana, wrap="word", height=10, font=("Segoe UI", 10)
        )
        self.caja_resultado.pack(padx=12, pady=(0, 8), fill="both", expand=True)
        self.caja_resultado.config(state="disabled")

        # --- Barra de estado ---
        self.etiqueta_estado = tk.Label(ventana, text="Listo.", fg="gray", anchor="w")
        self.etiqueta_estado.pack(padx=12, pady=(0, 10), fill="x")

    # ------------------------------------------------------------------
    def al_presionar_generar(self):
        semilla = self.entrada_semilla.get().strip()
        if not semilla:
            messagebox.showwarning("Falta la semilla", "Escribe una línea semilla primero.")
            return

        self._bloquear_botones(True)
        self.etiqueta_estado.config(text="Generando letra...")

        # Corremos la generacion en un hilo aparte para que la ventana no se congele
        hilo = threading.Thread(target=self._generar_en_segundo_plano, args=(semilla,))
        hilo.start()

    def _generar_en_segundo_plano(self, semilla):
        try:
            resultado = generar(semilla)
            os.makedirs(CARPETA_SALIDA_TEXTO, exist_ok=True)
            with open(RUTA_ULTIMA_GENERACION, "w", encoding="utf-8") as f:
                f.write(resultado)
            self.texto_generado = resultado
            self.ventana.after(0, self._mostrar_resultado, resultado)
        except Exception as e:
            self.ventana.after(0, self._mostrar_error, str(e))

    def _mostrar_resultado(self, resultado):
        self.caja_resultado.config(state="normal")
        self.caja_resultado.delete("1.0", tk.END)
        self.caja_resultado.insert(tk.END, resultado)
        self.caja_resultado.config(state="disabled")
        self.etiqueta_estado.config(text="Letra generada. Ya puedes escucharla.")
        self._bloquear_botones(False)
        self.boton_escuchar.config(state="normal")

    # ------------------------------------------------------------------
    def al_presionar_escuchar(self):
        if not self.texto_generado:
            return
        self._bloquear_botones(True)
        self.etiqueta_estado.config(text="Leyendo en voz alta...")
        hilo = threading.Thread(target=self._escuchar_en_segundo_plano)
        hilo.start()

    def _escuchar_en_segundo_plano(self):
        try:
            leer_en_voz_alta(self.texto_generado, guardar_audio=True)
            leer_en_voz_alta(self.texto_generado, guardar_audio=False)
            self.ventana.after(0, lambda: self.etiqueta_estado.config(
                text="Listo. Audio guardado en outputs/audio/letra_generada.wav"))
        except Exception as e:
            self.ventana.after(0, self._mostrar_error, str(e))
        finally:
            self.ventana.after(0, lambda: self._bloquear_botones(False))

    # ------------------------------------------------------------------
    def _mostrar_error(self, mensaje):
        messagebox.showerror("Ocurrió un error", mensaje)
        self.etiqueta_estado.config(text="Error. Revisa la consola.")
        self._bloquear_botones(False)

    def _bloquear_botones(self, bloquear: bool):
        estado = "disabled" if bloquear else "normal"
        self.boton_generar.config(state=estado)
        if not bloquear and self.texto_generado:
            self.boton_escuchar.config(state="normal")
        elif bloquear:
            self.boton_escuchar.config(state="disabled")


def main():
    ventana = tk.Tk()
    AppGeneradorDeLetras(ventana)
    ventana.mainloop()


if __name__ == "__main__":
    main()
