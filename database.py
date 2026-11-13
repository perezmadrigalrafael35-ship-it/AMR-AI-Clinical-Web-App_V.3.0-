import sqlite3

def inicializar_db():
    # Conectar (si no existe la base de datos, la crea)
    conn = sqlite3.connect('hospital_amr.db')
    cursor = conn.cursor()
    
    # Crear la tabla de reportes si no existe
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reportes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folio TEXT UNIQUE,
            fecha TEXT,
            nombre_paciente TEXT,
            id_paciente TEXT,
            edad INTEGER,
            hospital TEXT,
            antibiotico_evaluado TEXT,
            resultado TEXT,
            recomendacion TEXT
        )
    ''')
    conn.commit()
    conn.close()

def guardar_reporte(folio, fecha, nombre, id_paciente, edad, hospital, antibiotico, resultado, recomendacion):
    conn = sqlite3.connect('hospital_amr.db')
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO reportes (folio, fecha, nombre_paciente, id_paciente, edad, hospital, antibiotico_evaluado, resultado, recomendacion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (folio, fecha, nombre, id_paciente, edad, hospital, antibiotico, resultado, recomendacion))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"El folio {folio} ya existe.")
    finally:
        conn.close()

def obtener_reportes():
    conn = sqlite3.connect('hospital_amr.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reportes ORDER BY id DESC')
    reportes = cursor.fetchall()
    conn.close()
    return reportes

def borrar_reporte_por_folio(folio):
    conn = sqlite3.connect('hospital_amr.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM reportes WHERE folio = ?', (folio,))
    conn.commit()
    conn.close()

# Inicializamos la base de datos al importar el archivo
inicializar_db()