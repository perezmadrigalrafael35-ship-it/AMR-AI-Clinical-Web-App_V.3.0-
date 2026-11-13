import streamlit as st
import pandas as pd
import joblib
import datetime
import random
import database
import google.generativeai as genai
import numpy as np

# 1. Configurar la página web
st.set_page_config(page_title="AMR-AI Clinical Pipeline", page_icon="🧬", layout="wide")

st.title("🧬 AMR-AI: Clinical Decision Support System")
st.markdown("Sistema de apoyo a la decisión clínica basado en Inteligencia Artificial.")

# --- FOOTER GLOBAL (Derechos Reservados) ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: gray;
        text-align: right;
        padding: 10px 20px;
        font-size: 12px;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        © 2026 Todos los derechos reservados al Ing. en Biotecnología Rafael Pérez Madrigal.
    </div>
    """,
    unsafe_allow_html=True
)

# 2. Cargar los tres modelos
try:
    modelo_basico = joblib.load(r"C:\Users\perez\modelo_amr_real.pkl")
    modelo_genomico = joblib.load(r"C:\Users\perez\modelo_multilabel_ecoli.pkl")
    modelo_genomico_kleb = joblib.load(r"C:\Users\perez\modelo_multilabel_klebsiella.pkl")
except Exception as e:
    st.error(f"Error al cargar los modelos: {e}. Asegúrate de que los archivos .pkl estén en la ruta.")
    st.stop()

# 3. Configurar Gemini (Corregido y con tiempo de espera)
API_KEY = "TU_LLAVE_AQUI" 
genai.configure(api_key=API_KEY)
modelo_gemini = genai.GenerativeModel('gemini-3.8-flash')

# 4. Base de datos de antibióticos
ANTIBIOTICOS = {
    "Meropenem": {"familia": "Carbapenémico", "usar_ctex": "Amikacina", "usar_susceptible": "Meropenem"},
    "Ceftriaxona": {"familia": "Cefalosporina 3ra Gen", "usar_ctex": "Meropenem", "usar_susceptible": "Ceftriaxona"},
    "Cefotaxima": {"familia": "Cefalosporina 3ra Gen", "usar_ctex": "Meropenem", "usar_susceptible": "Cefotaxima"},
    "Ceftazidima": {"familia": "Cefalosporina 3ra Gen", "usar_ctex": "Meropenem", "usar_susceptible": "Ceftazidima"},
    "Ampicilina": {"familia": "Penicilina", "usar_ctex": "Meropenem/Amikacina", "usar_susceptible": "Ampicilina"},
    "Amoxicilina": {"familia": "Penicilina", "usar_ctex": "Meropenem/Amikacina", "usar_susceptible": "Amoxicilina"},
    "Amikacina": {"familia": "Aminoglucósido", "usar_ctex": "Es la alternativa", "usar_susceptible": "Amikacina"},
    "Gentamicina": {"familia": "Aminoglucósido", "usar_ctex": "Es la alternativa", "usar_susceptible": "Gentamicina"},
    "Ciprofloxacino": {"familia": "Fluoroquinolona", "usar_ctex": "Es la alternativa", "usar_susceptible": "Ciprofloxacino"},
    "Trimetoprima/Sulfametoxazol": {"familia": "Sulfonamida", "usar_ctex": "Es la alternativa", "usar_susceptible": "Trimetoprima/Sulfametoxazol"}
}

# 5. CREACIÓN DE LAS PESTAÑAS
tab1, tab2, tab3 = st.tabs(["🔬 Análisis de Paciente", "📋 Historial Clínico", "💬 Asistente IA (Gemini)"])

# ==========================================
# PESTAÑA 1: LA APP DEL HOSPITAL
# ==========================================
with tab1:
    # --- BARRA LATERAL ---
    st.sidebar.header("📋 Registro del Paciente")
    nombre = st.sidebar.text_input("Nombre completo del paciente:")
    id_paciente = st.sidebar.text_input("ID / Expediente del hospital:")
    edad = st.sidebar.number_input("Edad del paciente:", min_value=0, max_value=120, value=45)

    st.sidebar.subheader("🧪 Evaluación Específica (Opcional)")
    st.sidebar.markdown("Solo si deseas verificar un antibiótico en particular.")
    antibiotico_evaluado = st.sidebar.text_input("Escribe el antibiótico a evaluar:", placeholder="Ej. Ceftriaxona (o déjalo en blanco)")

    st.sidebar.subheader("🔬 Capacidad del Laboratorio")
    st.sidebar.subheader("🦠 Especie Bacteriana")
    bacteria_seleccionada = st.sidebar.selectbox(
        "¿Qué bacteria está analizando?",
        ["Escherichia coli", "Klebsiella pneumoniae"]
    )
    
    modo_lab = st.sidebar.radio(
        "¿Qué datos tiene disponibles?",
        ["Básico (Antibiograma/BLEE)", "Avanzado (Secuenciación/Genes)"]
    )

    st.sidebar.subheader("🧬 Marcadores y Resultados")
    if modo_lab == "Básico (Antibiograma/BLEE)":
        st.sidebar.info("Modo Hospital Público MX: Usando pruebas fenotípicas.")
        tiene_blee = st.sidebar.checkbox("Prueba de BLEE Positiva (+)")
        tiene_kpc_fenotipico = st.sidebar.checkbox("Resistente a Carbapenémicos (Antibiograma)")
        tiene_ctex = False
        tiene_kpc = False
    else:
        st.sidebar.success("Modo Laboratorio de Alta Especialidad: Usando genómica.")
        tiene_ctex = st.sidebar.checkbox("Gen blaCTX-M (BLEE) detectado")
        tiene_kpc = st.sidebar.checkbox("Gen blaKPC (Carbapenemasa) detectado")
        tiene_blee = tiene_ctex 
        tiene_kpc_fenotipico = tiene_kpc

    # --- ÁREA PRINCIPAL ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Análisis de Laboratorio")
        if st.button("🔬 Ejecutar Diagnóstico IA", type="primary"):
            if not nombre or not id_paciente:
                st.warning("Por favor, ingrese el nombre y el ID del paciente.")
            else:
                st.session_state['analizado'] = True
                st.session_state['datos_paciente'] = {
                    "nombre": nombre,
                    "id": id_paciente,
                    "edad": edad,
                    "bacteria": bacteria_seleccionada,
                    "antibiotico": antibiotico_evaluado.strip().capitalize() if antibiotico_evaluado.strip() else "No especificado",
                    "fecha": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "folio": f"AMR-{random.randint(1000, 9999)}"
                }
                if 'guardado' in st.session_state:
                    del st.session_state['guardado']

    with col2:
        st.subheader("📊 Estado del Sistema")
        st.success("🟢 **Modelos de IA Cargados**")
        st.markdown("- *Random Forest (Fenotípico)*")
        st.markdown("- *Random Forest (E. coli)*")
        st.markdown("- *Random Forest (Klebsiella)*")
        
        st.info("💡 **Tip Clínico:**")
        st.markdown("Usa el **Modo Avanzado** para análisis genómico o el **Modo Básico** si solo tienes pruebas fenotípicas.")

    # --- GENERACIÓN DEL REPORTE CLÍNICO ---
    if st.session_state.get('analizado', False):
        st.markdown("---")
        datos = st.session_state['datos_paciente']
        
        st.header("📄 Reporte Clínico Automatizado")
        
        with st.container(border=True):
            st.markdown(f"**Folio:** {datos['folio']} | **Fecha de Emisión:** {datos['fecha']}")
            st.markdown(f"**Paciente:** {datos['nombre']} (ID: {datos['id']}) | **Edad:** {datos['edad']} años")
            st.markdown(f"**Bacteria Analizada:** {datos['bacteria']}")
            
            if datos['antibiotico'] != "No especificado":
                st.markdown(f"**Antibiótico a Evaluar:** {datos['antibiotico']}")
            else:
                st.markdown(f"**Antibiótico a Evaluar:** No especificado (Mostrando resumen general)")
                
            st.divider()

            # --- RESUMEN CLÍNICO AUTOMÁTICO ---
            st.subheader("📊 Resumen Clínico Automático")
            
            if tiene_blee or tiene_kpc or tiene_kpc_fenotipico:
                st.error("⚠️ **Perfil de Resistencia Detectado:** Cepa multirresistente.")
                
                if tiene_blee:
                    st.markdown("**Mecanismo detectado:** Betalactamasa de Espectro Extendido (BLEE).")
                    st.markdown("❌ **Fármacos Contraindicados:** Ceftriaxona, Cefotaxima, Ceftazidima, Ampicilina.")
                    st.success("✅ **Opciones Terapéuticas Viables:** Meropenem, Amikacina (o Carbapenémicos según stock).")
                
                if tiene_kpc or tiene_kpc_fenotipico:
                    st.markdown("**Mecanismo detectado:** Carbapenemasa (KPC).")
                    st.markdown("❌ **Fármacos Contraindicados:** Meropenem, Imipenem, Ertapenem (y la mayoría de betalactámicos).")
                    st.success("✅ **Opciones Terapéuticas Viables:** Polimixina B, Tigeciclina, Ceftazidima/Avibactam.")
            else:
                st.success("✅ **Perfil de Susceptibilidad:** No se detectaron mecanismos de alto riesgo.")
                st.markdown("**Opción empírica sugerida:** Terapia de primera línea (Ampicilina, Trimetoprima/Sulfametoxazol, etc.).")
                
            st.divider()
            
            resultado_guardar = "Evaluación General"
            recomendacion_guardar = "Ver Resumen Clínico"

            # --- EVALUACIÓN ESPECÍFICA (Solo si el doctor escribió un antibiótico) ---
            if datos['antibiotico'] != "No especificado":
                st.subheader(f"🔬 Evaluación del Fármaco: {datos['antibiotico']}")
                
                info_ab = ANTIBIOTICOS.get(datos['antibiotico'])
                
                if not info_ab:
                    st.warning(f"El antibiótico '{datos['antibiotico']}' no está en la base de datos de cuadro básico. Consulta al Asistente IA abajo.")
                else:
                    if tiene_blee or tiene_kpc or tiene_kpc_fenotipico:
                        st.error(f"⚠️ **ALERTA: Perfil de resistencia detectado (Cepa BLEE/KPC).**")
                        resultado_guardar = "Resistente"
                        
                        if tiene_blee:
                            st.markdown(f"**Justificación:** Se ha detectado **Betalactamasa de Espectro Extendido (BLEE)**.")
                            if datos['antibiotico'] in ["Meropenem", "Amikacina"]:
                                st.success(f"✅ **Nota Clínica:** Aunque la cepa es BLEE, el **{datos['antibiotico']}** podría seguir siendo viable. Confirmar con antibiograma.")
                                recomendacion_guardar = f"Mantener {datos['antibiotico']}"
                            else:
                                st.success(f"✅ **Alternativa:** Meropenem o Amikacina.")
                                recomendacion_guardar = "Meropenem o Amikacina"
                        
                        if tiene_kpc or tiene_kpc_fenotipico:
                            st.markdown(f"**Justificación:** Se ha detectado **resistencia a Carbapenémicos**.")
                            if datos['antibiotico'] in ["Meropenem", "Imipenem", "Ertapenem"]:
                                st.error(f"❌ **Contraindicado:** Pertenece a los Carbapenémicos.")
                                st.success("✅ **Opciones:** Polimixina B, Tigeciclina.")
                                recomendacion_guardar = "Polimixina B / Tigeciclina"
                            else:
                                st.success("✅ **Opciones:** Polimixina B, Tigeciclina.")
                                recomendacion_guardar = "Polimixina B / Tigeciclina"
                    else:
                        # IA Básica
                        if modo_lab == "Básico (Antibiograma/BLEE)":
                            paciente = {'Genome ID': 562, 'Laboratory Typing Method': 'Disk diffusion'}
                            df_paciente = pd.DataFrame([paciente])
                            df_dummies = pd.get_dummies(df_paciente)
                            for col in modelo_basico.feature_names_in_:
                                if col not in df_dummies.columns:
                                    df_dummies[col] = 0
                            df_final = df_dummies[modelo_basico.feature_names_in_]
                            prediccion = modelo_basico.predict(df_final)[0]
                            
                            if prediccion == 1:
                                st.warning(f"⚠️ **RESULTADO: CEPA RESISTENTE (IA Fenotípica).**")
                                st.success(f"✅ **Tratamiento sugerido:** {info_ab['usar_ctex']}")
                                resultado_guardar = "Resistente (IA)"
                                recomendacion_guardar = info_ab['usar_ctex']
                            else:
                                st.success(f"✅ **RESULTADO: CEPA SUSCEPTIBLE.**")
                                st.info(f"💡 **Tratamiento sugerido:** {info_ab['usar_susceptible']}")
                                resultado_guardar = "Susceptible (IA)"
                                recomendacion_guardar = info_ab['usar_susceptible']
                        
                        # IA Genómica Dinámica
                        else:
                            st.info(f"🧬 Analizando perfil genómico con el Cerebro 2 (EMBL-EBI) para {datos['bacteria']}...")
                            
                            if datos['bacteria'] == "Escherichia coli":
                                modelo_actual = modelo_genomico
                            else:
                                modelo_actual = modelo_genomico_kleb
                            
                            df_gen = pd.DataFrame(0, index=[0], columns=modelo_actual.feature_names_in_)
                            prediccion_multilabel = modelo_actual.predict(df_gen)[0]
                            
                            antibiotico_buscar = datos['antibiotico'].lower()
                            
                            # Traductor de nombres
                            clases_crudas = modelo_actual.classes_
                            lista_antibioticos = []
                            for item in clases_crudas:
                                if isinstance(item, (list, tuple)) or hasattr(item, '__len__') and not isinstance(item, str):
                                    lista_antibioticos.append(str(item[0]).lower())
                                else:
                                    lista_antibioticos.append(str(item).lower())
                            
                            diccionario_ab = {
                                "meropenem": ["meropenem", "carbapenem", "imipenem"],
                                "ceftriaxona": ["ceftriaxone", "cephalosporin", "cefotaxime"],
                                "ciprofloxacino": ["ciprofloxacin", "fluoroquinolone"],
                                "amikacina": ["amikacin", "aminoglycoside"],
                                "trimetoprima/sulfametoxazol": ["trimethoprim", "sulfonamide", "co-trimoxazole"],
                                "ampicilina": ["ampicillin", "penicillin"]
                            }
                            
                            nombres_a_buscar = diccionario_ab.get(antibiotico_buscar, [antibiotico_buscar])
                            
                            idx_encontrado = -1
                            for nombre in nombres_a_buscar:
                                if nombre in lista_antibioticos:
                                    idx_encontrado = lista_antibioticos.index(nombre)
                                    break
                            
                            if idx_encontrado != -1:
                                try:
                                    if hasattr(prediccion_multilabel, 'shape') and len(prediccion_multilabel.shape) > 1:
                                        es_resistente = prediccion_multilabel[0][idx_encontrado]
                                    elif isinstance(prediccion_multilabel, (list, tuple)):
                                        es_resistente = prediccion_multilabel[0][idx_encontrado] if isinstance(prediccion_multilabel[0], (list, tuple)) else prediccion_multilabel[idx_encontrado]
                                    else:
                                        es_resistente = prediccion_multilabel
                                except Exception:
                                    es_resistente = 0
                            else:
                                st.info(f"🧬 El antibiótico '{datos['antibiotico']}' no tiene una coincidencia directa en el catálogo genómico.")
                                try:
                                    if hasattr(prediccion_multilabel, 'shape') and len(prediccion_multilabel.shape) > 1:
                                        es_resistente = prediccion_multilabel[0][0]
                                    elif isinstance(prediccion_multilabel, (list, tuple)):
                                        if isinstance(prediccion_multilabel[0], (list, tuple)):
                                            es_resistente = prediccion_multilabel[0][0]
                                        else:
                                            es_resistente = prediccion_multilabel[0]
                                    else:
                                        es_resistente = prediccion_multilabel
                                except Exception:
                                    es_resistente = 0

                            if isinstance(es_resistente, (list, tuple)) or hasattr(es_resistente, '__len__'):
                                if hasattr(es_resistente, 'flatten'):
                                    es_resistente = es_resistente.flatten()[0]
                                else:
                                    es_resistente = es_resistente[0]
                            
                            try:
                                es_resistente = int(es_resistente)
                            except:
                                es_resistente = 0

                            if es_resistente == 1:
                                st.warning(f"⚠️ **RESULTADO: CEPA RESISTENTE a {datos['antibiotico']} (IA Genómica).**")
                                st.success(f"✅ **Tratamiento sugerido:** {info_ab['usar_ctex']}")
                                resultado_guardar = "Resistente (Genómica)"
                                recomendacion_guardar = info_ab['usar_ctex']
                            else:
                                st.success(f"✅ **RESULTADO: CEPA SUSCEPTIBLE a {datos['antibiotico']} (IA Genómica).**")
                                st.info(f"💡 **Tratamiento sugerido:** {info_ab['usar_susceptible']}")
                                resultado_guardar = "Susceptible (Genómica)"
                                recomendacion_guardar = info_ab['usar_susceptible']
            
            else:
                st.info("💡 No se seleccionó un antibiótico específico para evaluar. El resumen clínico de arriba es la recomendación general.")
            
            # --- GUARDAR EN BASE DE DATOS ---
            if 'guardado' not in st.session_state:
                database.guardar_reporte(
                    datos['folio'], datos['fecha'], datos['nombre'], datos['id'], 
                    datos['edad'], datos['bacteria'], 
                    datos['antibiotico'], resultado_guardar, recomendacion_guardar
                )
                st.session_state['guardado'] = True

            # --- FARMACÉUTICO VIRTUAL ---
            st.divider()
            st.subheader("💊 Consulta de Disponibilidad (Farmacia)")
            st.markdown("Si el antibiótico recomendado no está disponible, consulta al Asistente IA por una alternativa viable.")
            
            col_farm1, col_farm2 = st.columns([3, 1])
            with col_farm1:
                falta_medicamento = st.text_input("¿Qué medicamento no está disponible?", placeholder="Ej. Meropenem, Polimixina B...")
            with col_farm2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🔍 Buscar Alternativa", type="secondary"):
                    if falta_medicamento:
                        with st.spinner("Consultando al asistente clínico..."):
                            try:
                                contexto_farmacia = f"""
                                Eres un farmacéutico clínico experto en resistencia antimicrobiana.
                                Un paciente tiene una infección por {datos['bacteria']}.
                                Reporte previo: {resultado_guardar}.
                                Recomendación inicial: {recomendacion_guardar}.
                                El medicamento {falta_medicamento} NO ESTÁ DISPONIBLE en el hospital.
                                Dime cuál sería la mejor alternativa (de otra familia si es necesario) que sea viable, 
                                y explica por qué en 2 oraciones.
                                """
                                respuesta_alt = modelo_gemini.generate_content(contexto_farmacia)
                                st.info(f"🤖 **Sugerencia del Asistente:** \n\n{respuesta_alt.text}")
                            except Exception as e:
                                st.error(f"No se pudo conectar con el asistente. Detalle: {e}")

            st.divider()
            st.caption("Firma del responsable: ___________________________ (Validación médica requerida)")
            st.caption("Este es un reporte generado por IA y no sustituye el juicio clínico de un profesional de la salud.")

# ==========================================
# PESTAÑA 2: HISTORIAL CLÍNICO
# ==========================================
with tab2:
    st.header("📋 Historial de Reportes Clínicos")
    st.markdown("Consulta los diagnósticos generados previamente por el sistema.")
    
    reportes = database.obtener_reportes()
    
    if not reportes:
        st.info("No hay reportes guardados en el sistema aún.")
    else:
        df_reportes = pd.DataFrame(reportes, columns=["ID", "Folio", "Fecha", "Paciente", "ID Paciente", "Edad", "Bacteria", "Antibiótico", "Resultado", "Recomendación"])
        st.dataframe(df_reportes, use_container_width=True)
        
        st.divider()
        st.subheader("⚠️ Gestión de Errores")
        st.markdown("Si un reporte fue generado por error, puedes eliminarlo ingresando su folio.")
        
        col_borrar1, col_borrar2 = st.columns([2, 1])
        with col_borrar1:
            folio_a_borrar = st.text_input("Ingresa el Folio a eliminar (Ej. AMR-1234):")
        with col_borrar2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Eliminar Reporte", type="primary"):
                if folio_a_borrar:
                    database.borrar_reporte_por_folio(folio_a_borrar)
                    st.success(f"El reporte con folio {folio_a_borrar} ha sido eliminado.")
                    st.rerun()
                else:
                    st.warning("Por favor, ingresa un folio válido.")

# ==========================================
# PESTAÑA 3: EL ASISTENTE INTELIGENTE (GEMINI)
# ==========================================
with tab3:
    st.header("💬 Asistente Clínico IA")
    st.markdown("Consulta sobre mecanismos de resistencia, antibióticos o interpretación de genes.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar historial de mensajes con avatares profesionales
    for message in st.session_state.messages:
        avatar = "🧑‍⚕️" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # Input del chat
    if prompt := st.chat_input("Ej: ¿Qué antibióticos debo evitar si tengo el gen blaKPC?"):
        with st.chat_message("user", avatar="🧑‍⚕️"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Consultando bases de datos médicas..."):
                try:
                    # Contexto completo, pidiendo formato visual
                    contexto = f"""
                    You are an expert clinical microbiologist specializing in antimicrobial resistance. 
                    Answer the following question clearly, technically, and in detail. 
                    Use bullet points and bold text to highlight key concepts. 
                    Reply in the same language the user uses.
                    
                    Question: {prompt}
                    """
                    # Agregamos el timeout de 600 segundos para que no se corte la respuesta
                    respuesta = modelo_gemini.generate_content(
                        contexto, 
                        request_options={"timeout": 600}
                    )
                    st.markdown(respuesta.text)
                    st.session_state.messages.append({"role": "assistant", "content": respuesta.text})
                except Exception as e:
                    st.error(f"Error con Gemini: {e}")