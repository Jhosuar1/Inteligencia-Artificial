# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify, session
import joblib
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import datetime
import os
from voz import hablar
import re
import json
from nlp import extraer_sintomas_nlp
from fpdf.enums import XPos, YPos
from utils.reglas_generadas_es import reglas
from utils.gravedad import gravedades

app = Flask(__name__)
app.secret_key = 'clave-segura'

modelo = joblib.load("modelo_entrenado.pkl")
df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
columnas_sintomas = df.drop(columns=["diseases"]).columns.tolist()
df_enfermedades = pd.read_csv("data/enfermedades.csv")

with open("data/diccionario_sintomas_en_es.json", encoding="utf-8") as f:
    diccionario = json.load(f)

diccionario_invertido = {v: k for k, v in diccionario.items()}

with open("data/recomendaciones.json") as f:
    recomendaciones = json.load(f)


def evaluar_reglas(sintomas):
    """
    Evalúa si los síntomas coinciden exactamente con alguna regla del sistema experto.
    
    Args:
        sintomas (list): Lista de síntomas detectados.

    Returns:
        str or None: Diagnóstico basado en reglas si hay coincidencia, o None.
    """
    sintomas_set = set(sintomas)
    for clave, valor in reglas.items():
        clave_set = set([s.strip() for s in clave.split(",")])
        if clave_set.issubset(sintomas_set):
            return valor
    return None


def buscar_por_coincidencia(sintomas_usuario):
    """
    Busca enfermedades que compartan al menos un síntoma con los síntomas del usuario.

    Args:
        sintomas_usuario (list): Lista de síntomas detectados.

    Returns:
        str or None: Enfermedad con síntomas coincidentes o None.
    """
    sintomas_usuario_set = set(sintomas_usuario)
    for _, fila in df_enfermedades.iterrows():
        sintomas_bd = set([s.strip().lower() for s in fila["sintomas"].split(",")])
        if sintomas_bd.intersection(sintomas_usuario_set):
            return fila["enfermedad"]
    return None


def exportar_pdf(sintomas, diagnostico, confianza, recomendacion):
    """
    Genera un informe PDF del diagnóstico médico.

    Args:
        sintomas (str): Texto original con los síntomas.
        diagnostico (str): Diagnóstico determinado.
        confianza (float): Porcentaje de confianza del diagnóstico.
        recomendacion (str): Recomendación médica.
    """
    ruta_fuentes = "utils/fonts"

    pdf = FPDF()
    pdf.add_page()

    pdf.add_font('DejaVu', '', os.path.join(ruta_fuentes, 'DejaVuSans.ttf'))
    pdf.add_font('DejaVu', 'B', os.path.join(ruta_fuentes, 'DejaVuSans-Bold.ttf'))
    pdf.add_font('DejaVu', 'I', os.path.join(ruta_fuentes, 'DejaVuSans-Oblique.ttf'))

    pdf.set_font('DejaVu', '', 16)
    pdf.set_text_color(0, 102, 204)
    pdf.cell(0, 10, "Informe de Diagnóstico Médico", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    pdf.set_font('DejaVu', '', 12)
    pdf.set_text_color(0)
    fecha = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 10, f"Fecha: {fecha}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font('DejaVu', 'B', 13)
    pdf.cell(0, 10, "Diagnóstico:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, f"- Enfermedad estimada: {diagnostico}\n- Nivel de confianza: {confianza:.2f}%")
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 13)
    pdf.cell(0, 10, "Síntomas reportados:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, sintomas)
    pdf.ln(5)

    pdf.set_font('DejaVu', 'B', 13)
    pdf.set_text_color(220, 50, 50)
    pdf.cell(0, 10, "Recomendación médica:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0)
    pdf.set_font('DejaVu', '', 12)
    pdf.multi_cell(0, 10, recomendacion, border=1)
    pdf.ln(5)

    pdf.set_font('DejaVu', 'I', 10)
    pdf.set_text_color(100)
    pdf.multi_cell(0, 10, "Este informe ha sido generado automáticamente por el Asistente Médico Inteligente. Para un diagnóstico definitivo, consulte a un profesional de la salud.")

    pdf.output("static/diagnostico.pdf")


@app.route("/")
def index():
    """
    Renderiza la página principal del asistente.
    """
    return render_template("index.html")


@app.route("/diagnosticar", methods=["POST"])
def diagnosticar():
    """
    Procesa los síntomas recibidos, aplica el modelo, reglas y coincidencias para generar un diagnóstico.
    Retorna una respuesta JSON con el diagnóstico, confianza, recomendación y PDF generado.
    """
    data = request.get_json()
    entrada = data.get("mensaje", "")
    sintomas_es = extraer_sintomas_nlp(entrada)

    if not sintomas_es:
        return jsonify({"respuesta": "No se detectaron síntomas válidos.", "pdf": ""})

    sintomas_ingles = [diccionario_invertido.get(s, None) for s in sintomas_es]
    sintomas_ingles = [s for s in sintomas_ingles if s is not None]

    entrada_binaria = pd.DataFrame([[0]*len(columnas_sintomas)], columns=columnas_sintomas)
    sintomas_reconocidos = [s for s in sintomas_ingles if s in columnas_sintomas]
    for s in sintomas_reconocidos:
        entrada_binaria[s] = 1

    diagnostico = None
    confianza = 0.0

    if sintomas_reconocidos:
        probabilidades = modelo.predict_proba(entrada_binaria)[0]
        confianza = max(probabilidades) * 100
        if confianza >= 40:
            diagnostico = modelo.predict(entrada_binaria)[0]

    if confianza < 30.0:
        diagnostico = "Diagnóstico no concluyente. Se recomienda una consulta médica presencial."

    if not diagnostico or diagnostico.startswith("Diagnóstico no concluyente"):
        diag_reglas = evaluar_reglas(sintomas_es)
        if diag_reglas:
            diagnostico = diag_reglas
            confianza = 100.0

    if (not diagnostico or diagnostico.startswith("Diagnóstico no concluyente")) and not diagnostico in reglas.values():
        diag_coinc = buscar_por_coincidencia(sintomas_es)
        if diag_coinc:
            diagnostico = diag_coinc
            confianza = 100.0

    if not diagnostico or diagnostico.startswith("Diagnóstico no concluyente"):
        return jsonify({"respuesta": "No se pudo determinar un diagnóstico confiable. Por favor, consulte a un médico.", "pdf": ""})

    recomendacion = recomendaciones.get(diagnostico, "Consulte a un profesional médico para una valoración completa.")
    gravedad = gravedades.get(diagnostico, "Desconocida")

    respuesta = (
        f"Diagnóstico probable: {diagnostico} (confianza: {confianza:.2f}%)\n"
        f"Gravedad estimada: {gravedad}\n"
        f"Recomendación: {recomendacion}"
    )

    exportar_pdf(entrada, diagnostico, confianza, recomendacion)
    hablar(f"Tu diagnóstico probable es: {diagnostico}")

    session.setdefault("historial", []).append({
        "fecha": datetime.datetime.now().isoformat(),
        "entrada": entrada,
        "diagnostico": diagnostico,
        "confianza": confianza,
        "recomendacion": recomendacion
    })

    return jsonify({"respuesta": respuesta, "pdf": "/static/diagnostico.pdf"})

if __name__ == "__main__":
    app.run(debug=True)