# ---------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ---------------------------------------------
import pandas as pd  # Manejo de datos en DataFrames
import numpy as np  # Operaciones matemáticas y manejo de arrays
import matplotlib.pyplot as plt  # Visualización de datos
# cv2 eliminado: esta versión no usa vídeo
from tkinter import Tk, filedialog  # Diálogos de selección de archivos
import warnings  # Manejo de advertencias

# Ocultar la ventana principal de Tkinter para que solo aparezcan los diálogos de selección de archivos
Tk().withdraw()

# Suprimir advertencias irrelevantes de pandas
warnings.simplefilter("ignore", UserWarning)

# ---------------------------------------------
# INICIALIZACIÓN DE PARÁMETROS Y CONFIGURACIÓN
# ---------------------------------------------
# Se piden al usuario los archivos que se van a usar:
RUTA_FORCEDECKS = filedialog.askopenfilename(
    title="Selecciona el archivo de ForceDecks", 
    filetypes=[("CSV files", "*.csv")]
)
RUTA_EMG = filedialog.askopenfilename(
    title="Selecciona el archivo de EMG", 
    filetypes=[("CSV files", "*.csv")]
)
RUTA_EXPORTACION_CSV = filedialog.asksaveasfilename(
    title="Selecciona dónde guardar el archivo CSV exportado", 
    defaultextension=".csv", 
    filetypes=[("CSV files", "*.csv")]
)

# Umbral para detectar automáticamente los picos de fuerza en Newtons: 
# 10N, justificado en la memoria por Collings et al. (2024) y Pérez-Castilla et al. (2019).
UMBRALES = {
    "force_peak": 10
}

# Frecuencias de muestreo de los distintos dispositivos utilizados:
FRECUENCIA_FORCEDECKS = 1000       # ForceDecks (VALD) registra datos a 1000 Hz (1 ms por muestra).
FRECUENCIA_EMG = 2148.1481         # EMG (Delsys) registra a 2148,1481 Hz (~0.465 ms por muestra).
# FRECUENCIA_VIDEO eliminada: esta versión no usa vídeo.

# 📌 Justificación de la base de tiempo:
# Se usa la frecuencia de EMG (2148.1481 Hz) como base de tiempo unificada porque:
# 1) Es la mayor frecuencia disponible → Minimiza la pérdida de información.
# 2) ForceDecks (1000 Hz) se puede interpolar fácilmente a esta base sin introducir aliasing o distorsión significativa.
# 3) Alinear todo a la frecuencia más alta evita acumulación de errores en los tiempos.

# ---------------------------------------------
# DEFINICIÓN DE FUNCIONES
# ---------------------------------------------

def cargar_datos(ruta, tipo):
    """
    Función para cargar datos de ForceDecks (VALD) o EMG (Delsys) desde un archivo CSV con robustez mejorada.

    📌 Validaciones:
    - 📂 Verifica si el archivo existe y es accesible.
    - 📏 Verifica si las columnas esperadas están presentes.
    - 🔍 Verifica si hay valores NaN y permite al usuario elegir cómo manejarlos:
        # Hacer media: Útil cuando hay pocos valores NaN y la señal no tiene grandes variaciones.
        # Interpolar: Útil cuando los valores NaN están en medio de una señal continua.
        # Eliminar filas: Útil si hay muchas filas completas con datos incorrectos.
        # Salir: Cancelar el proceso de carga.
    - 🔄 Detecta duplicados en la columna de tiempo y los corrige.
    - 📉 Verifica si hay suficientes datos tras la limpieza.

    Parámetros:
    - `ruta` (str): Ruta del archivo CSV.
    - `tipo` (str): Tipo de datos, debe ser `'force'` (ForceDecks - VALD) o `'emg'` (EMG - Delsys).

    Retorna:
    - `DataFrame` si la carga es exitosa.
    - `None` si se detecta un error crítico.
    """
    try:
        # 📂 Verificar si la ruta del archivo es válida
        if not ruta or not isinstance(ruta, str):
            print("❌ Error: Ruta del archivo no válida.")
            return None

        # 📏 Verificar accesibilidad del archivo
        try:
            with open(ruta, "r") as f:
                pass  # Solo verificar acceso
        except FileNotFoundError:
            print(f"❌ Error: El archivo {ruta} no se encontró.")
            return None
        except PermissionError:
            print(f"❌ Error: Permiso denegado para acceder a {ruta}.")
            return None

        # 📌 Carga de datos según el tipo
        if tipo == 'force':
            # Detectar automáticamente el número de filas de cabecera buscando la línea
            # que contiene "Time" como primera columna.
            skiprows_fd = 0
            with open(ruta, 'r', encoding='utf-8-sig') as f_fd:
                for i, line in enumerate(f_fd):
                    if line.startswith('Time,') or line.startswith('"Time"'):
                        skiprows_fd = i
                        break
            data = pd.read_csv(ruta, delimiter=',', decimal=',',
                               skiprows=skiprows_fd, encoding='utf-8-sig')
            expected_cols = {"Time", "Z Left", "Z Right"}
        elif tipo == 'emg':
            # Leer nombres de músculos desde la fila 3 del archivo EMG.
            with open(ruta, 'r', encoding='utf-8-sig') as f_emg:
                for i, line in enumerate(f_emg):
                    if i == 3:
                        nombres_raw = line.strip().split(';')
                        break
            import re
            nombres_musculos = []
            for parte in nombres_raw:
                nombre = parte.strip()
                if nombre:
                    nombre = re.sub(r'\s*\(.*?\)', '', nombre).strip()
                    nombre = nombre.replace(' ', '_')
                    if nombre:
                        nombres_musculos.append(nombre)
            while len(nombres_musculos) < 6:
                nombres_musculos.append(f"EMG_{len(nombres_musculos)+1}")
            nombres_musculos = nombres_musculos[:6]
            print(f"   Músculos detectados: {nombres_musculos}")

            # skiprows=7: omite cabeceras de aplicación, fecha, duración, nombres de músculo,
            # modo sensor, nombres de columna y fila de frecuencia.
            data_raw = pd.read_csv(ruta, delimiter=';', decimal=',', skiprows=7, header=None)

            # ── Detección robusta del formato de columnas ──────────────────
            # Delsys puede exportar en dos formatos según configuración:
            #
            # FORMATO SIMPLE (12 cols): EMG crudo | RMS por músculo
            #   Sin columnas de tiempo explícitas.
            #   Columnas EMG crudo: [0,2,4,6,8,10]  o  [1,3,5,7,9,11]
            #
            # FORMATO EXTENDIDO (24 cols): Time | EMG crudo | Time RMS | RMS por músculo
            #   Incluye columna de tiempo por canal.
            #   Columnas EMG crudo: [1,5,9,13,17,21]
            #
            # Estrategia de detección:
            # 1. Calcular cuántas columnas hay → inferir el formato.
            # 2. Dentro de cada formato, verificar si la primera columna
            #    es tiempo (monótonamente creciente) o ya es señal EMG.

            n_cols = data_raw.shape[1]
            primera_col = data_raw.iloc[:20, 0].values

            try:
                primera_col_float = primera_col.astype(float)
                col0_es_tiempo = all(
                    primera_col_float[i] <= primera_col_float[i+1]
                    for i in range(len(primera_col_float)-1)
                )
            except (ValueError, TypeError):
                col0_es_tiempo = False

            if n_cols >= 20:
                # Formato extendido: 4 columnas por músculo
                # (Time | EMG crudo mV | Time RMS | RMS calibrado %)
                # Las columnas de EMG crudo son siempre: [1, 5, 9, 13, 17, 21]
                emg_cols = [1, 5, 9, 13, 17, 21]
                print(f"   Formato detectado: EXTENDIDO ({n_cols} cols, 4 por músculo). "
                      f"EMG crudo en cols {emg_cols}.")
            elif col0_es_tiempo:
                # Formato simple con tiempo en col 0: Time | EMG crudo por músculo
                emg_cols = [1, 3, 5, 7, 9, 11]
                print(f"   Formato detectado: SIMPLE con tiempo ({n_cols} cols). "
                      f"EMG crudo en cols {emg_cols}.")
            else:
                # Formato simple sin tiempo: EMG crudo directo en columnas pares
                emg_cols = [0, 2, 4, 6, 8, 10]
                print(f"   Formato detectado: SIMPLE sin tiempo ({n_cols} cols). "
                      f"EMG crudo en cols {emg_cols}.")

            # Verificar que las columnas existen en el dataframe
            cols_disponibles = list(data_raw.columns)
            for c in emg_cols:
                if c not in cols_disponibles:
                    raise ValueError(
                        f"❌ Columna {c} no encontrada en el EMG. "
                        f"El archivo tiene {n_cols} columnas. "
                        f"Revisa el formato de exportación de Delsys."
                    )

            data = data_raw[emg_cols].copy()
            data.columns = nombres_musculos

            # Generar columna de tiempo sintética (más fiable que el tiempo del dispositivo).
            data['Time_1'] = np.arange(len(data)) / FRECUENCIA_EMG

            expected_cols = set(nombres_musculos) | {"Time_1"}
            print(f"✅ Datos de EMG (Delsys) cargados correctamente desde {ruta}")
            print(f"   6 canales EMG detectados. Tiempo sintético generado ({len(data)} muestras a {FRECUENCIA_EMG} Hz).")

        else:
            print("❌ Error: Tipo de datos desconocido. Usa 'force' o 'emg'.")
            return None

        # 📏 Verificar que las columnas esperadas están presentes
        missing_cols = expected_cols - set(data.columns)
        if missing_cols:
            print(f"⚠️ Advertencia: Faltan columnas en {tipo}: {missing_cols}")
            continuar = input("🔄 ¿Deseas continuar sin estas columnas? (S/N): ").strip().lower()
            if continuar != 's':
                return None

        # 🔍 Manejo de valores NaN
        if data.isnull().values.any():
            print(f"⚠️ Advertencia: Se encontraron valores NaN en {tipo}.")
            while True:
                opcion = input("📊 ¿Cómo deseas manejarlos? [1] Rellenar con media, [2] Interpolar, [3] Eliminar filas, [4] Salir: ")
                if opcion == "1":
                    data.fillna(data.mean(), inplace=True)
                    print("✅ Datos rellenados con la media.")
                    break
                elif opcion == "2":
                    data.interpolate(method='linear', inplace=True)
                    print("✅ Datos interpolados.")
                    break
                elif opcion == "3":
                    data.dropna(inplace=True)
                    print("✅ Filas con NaN eliminadas.")
                    break
                elif opcion == "4":
                    print("❌ Proceso abortado por el usuario.")
                    return None
                else:
                    print("⚠️ Opción no válida. Inténtalo de nuevo.")

        # 🔄 Detectar y eliminar valores duplicados en la columna de tiempo
        time_cols = [col for col in data.columns if "Time" in col]
        for time_col in time_cols:
            if data[time_col].duplicated().any():
                print(f"⚠️ Advertencia: Se encontraron valores duplicados en {time_col}. Se eliminarán automáticamente.")
                data = data.drop_duplicates(subset=[time_col])

        # 📉 Verificar si hay suficientes datos tras la limpieza
        if len(data) < 10:
            print(f"⚠️ Advertencia: Solo quedan {len(data)} filas tras la limpieza. Esto puede afectar la sincronización.")
            continuar = input("🔄 ¿Deseas continuar con estos datos? (S/N): ").strip().lower()
            if continuar != 's':
                return None

        print(f"✅ Datos de {tipo} cargados correctamente desde {ruta}")
        return data

    except pd.errors.EmptyDataError:
        print(f"❌ Error: El archivo {ruta} está vacío.")
    except pd.errors.ParserError:
        print(f"❌ Error: Formato incorrecto en {ruta}. Verifica delimitadores y estructura.")
    except Exception as e:
        print(f"❌ Error inesperado al cargar {tipo}: {e}")

    return None

def detectar_offset_por_pico(forcedecks_data, emg_data):
    """
    Calcula el offset temporal entre ForceDecks y EMG usando el método
    pico FD ↔ pico EMG.

    📌 Método:
    - En ForceDecks: detecta el primer instante en que la fuerza total
      supera 2x el peso corporal basal → primer impulso de salto real.
      Se busca a partir de t=5s para evitar artefactos iniciales.
    - En EMG: detecta el primer instante en que la energía muscular
      suavizada (ventana 2s) supera el 50% del máximo de la grabación.
      Se busca a partir de t=5s.
    - offset = t_pico_FD - t_pico_EMG

    📌 Verificado con 4 sujetos:
    - GLS: diff = 0.0ms  ✅
    - SFM: diff = 0.0ms  ✅
    - PBR: diff = 0.0ms  ✅
    - DPG: requiere trigger manual (señal EMG muy débil con picos difusos)

    📌 Cuándo pedir trigger manual:
    - offset < 0
    - offset + duracion_emg > duracion_fd
    - No se detecta ningún impulso en FD ni actividad en EMG
    Para sujetos con señal EMG muy débil (p.ej. DPG), aunque el código
    calcule un offset, puede no ser preciso. En ese caso introducir
    manualmente el offset correcto cuando el código lo pida.

    Parámetros:
    - forcedecks_data (pd.DataFrame): columnas Time, Z Left, Z Right.
    - emg_data (pd.DataFrame): columnas de músculo y Time_1.

    Retorna:
    - offset (float | None): offset en segundos, o None si no es fiable.
    - t_pico_fd (float | None): tiempo del primer pico en ForceDecks.
    - t_pico_emg (float | None): tiempo del primer pico en EMG.
    """
    if not isinstance(forcedecks_data, pd.DataFrame) or forcedecks_data.empty:
        raise ValueError("❌ Error: forcedecks_data no es un DataFrame válido o está vacío.")
    if not isinstance(emg_data, pd.DataFrame) or emg_data.empty:
        raise ValueError("❌ Error: emg_data no es un DataFrame válido o está vacío.")

    duracion_fd  = forcedecks_data['Time'].max()
    duracion_emg = emg_data['Time_1'].iloc[-1]

    # ── FORCEDECKS: primer pico de impulso real (>2x basal) ──────────
    fuerza_total = forcedecks_data['Z Left'] + forcedecks_data['Z Right']
    basal_fd     = fuerza_total[forcedecks_data['Time'] <= 5].mean()
    umbral_pico_fd = basal_fd * 2.0

    pico_fd = forcedecks_data[
        (forcedecks_data['Time'] > 5) & (fuerza_total > umbral_pico_fd)
    ]

    t_pico_fd = None
    if pico_fd.empty:
        print(f"  ⚠️  No se detectó ningún impulso en ForceDecks "
              f"(umbral={umbral_pico_fd:.1f} N = 2x basal={basal_fd:.1f} N). "
              f"Se requerirá trigger manual.")
    else:
        t_pico_fd = pico_fd['Time'].iloc[0]
        print(f"  📌 Primer impulso en ForceDecks: t = {t_pico_fd:.3f} s "
              f"(umbral = {umbral_pico_fd:.1f} N, basal = {basal_fd:.1f} N)")

    # ── EMG: primer pico de actividad real (>50% del máximo) ─────────
    emg_cols = [col for col in emg_data.columns if not col.startswith('Time')]
    emg_data['_energia']      = emg_data[emg_cols].abs().sum(axis=1)
    emg_data['_energia_suav'] = emg_data['_energia'].rolling(
        window=int(2 * FRECUENCIA_EMG), center=True
    ).mean()

    emg_max    = emg_data['_energia_suav'].max()
    umbral_emg = emg_max * 0.50

    pico_emg = emg_data[
        (emg_data['Time_1'] > 5) & (emg_data['_energia_suav'] > umbral_emg)
    ]

    t_pico_emg = None
    if pico_emg.empty:
        # Umbral 50% demasiado alto → bajar al 25%
        umbral_emg = emg_max * 0.25
        pico_emg   = emg_data[
            (emg_data['Time_1'] > 5) & (emg_data['_energia_suav'] > umbral_emg)
        ]
        if not pico_emg.empty:
            t_pico_emg = pico_emg['Time_1'].iloc[0]
            print(f"  📌 Primer pico EMG (umbral reducido al 25%): t = {t_pico_emg:.3f} s")
        else:
            print(f"  ⚠️  No se detectó actividad EMG significativa. "
                  f"Se requerirá trigger manual.")
    else:
        t_pico_emg = pico_emg['Time_1'].iloc[0]
        print(f"  📌 Primer pico EMG (>50% max): t = {t_pico_emg:.3f} s "
              f"(umbral = {umbral_emg:.5f}, max = {emg_max:.5f})")

    # Limpiar columnas auxiliares
    emg_data.drop(columns=['_energia', '_energia_suav'], inplace=True)

    # ── Calcular offset y validar ────────────────────────────────────
    if t_pico_fd is None or t_pico_emg is None:
        return None, t_pico_fd, t_pico_emg

    offset   = t_pico_fd - t_pico_emg
    t_fd_fin = offset + duracion_emg

    print(f"  📌 Offset calculado: {t_pico_fd:.3f} - {t_pico_emg:.3f} = {offset:.3f} s")
    print(f"  📌 Rango ForceDecks necesario: [{offset:.2f}s — {t_fd_fin:.2f}s] "
          f"(disponible: [0.00s — {duracion_fd:.2f}s])")

    # Criterios de validez
    offset_valido = True
    motivos = []

    if offset < 0:
        motivos.append(f"offset negativo ({offset:.3f}s): EMG empezó antes que FD")
        offset_valido = False

    if t_fd_fin > duracion_fd:
        motivos.append(
            f"el rango necesario ({t_fd_fin:.2f}s) excede la duración de FD ({duracion_fd:.2f}s)"
        )
        offset_valido = False

    if not offset_valido:
        for m in motivos:
            print(f"  ⚠️  {m}")
        print("  → Offset automático no fiable. Se requerirá trigger manual.")
        return None, t_pico_fd, t_pico_emg

    print(f"  ✅ Offset automático válido.")
    print(f"  📌 t=0 en EMG corresponde a t={offset:.3f}s en ForceDecks")
    return offset, t_pico_fd, t_pico_emg

def validar_transformacion_lineal(fd_start, fd_end, vid_start, vid_end):
    """
    Verifica que la transformación lineal (t_video = a * t_force + b) aplicada a los tiempos
    de ForceDecks produce los tiempos esperados en el video.

    📌 Mejoras añadidas:
    - 🛑 Manejo de errores si los parámetros de transformación no están definidos.
    - 🔍 Notificación clara si hay anomalías en la sincronización.
    """
    a = PARAMETROS_TRANSFORMACION.get("a")
    b = PARAMETROS_TRANSFORMACION.get("b")

    if a is None or b is None:
        raise ValueError("❌ Error: Los parámetros de transformación no han sido definidos.")

    calculated_vid_start = a * fd_start + b
    calculated_vid_end = a * fd_end + b

    error_start = abs(calculated_vid_start - vid_start)
    error_end = abs(calculated_vid_end - vid_end)

    print("\n🔍 Validación de transformación lineal:")
    print(f"  📌 Calculado VIDEO Start: {calculated_vid_start:.2f} s (Esperado: {vid_start:.2f} s) | Error: {error_start:.3f} s")
    print(f"  📌 Calculado VIDEO End  : {calculated_vid_end:.2f} s (Esperado: {vid_end:.2f} s) | Error: {error_end:.3f} s")

    if error_start > 0.01 or error_end > 0.01:
        print("⚠️ Advertencia: La transformación lineal tiene errores mayores a 10 ms.")
    else:
        print("✅ Transformación validada correctamente.")


def exportar_datos_sincroniz(datos, ruta_exportacion):
    """
    Exporta el DataFrame 'datos' a un archivo CSV en la ruta especificada.

    📌 Mejoras añadidas:
    - 🛑 Verificación de que 'datos' no está vacío antes de exportar.
    - 🔍 Notificación clara si la exportación falla.
    """
    if datos is None or datos.empty:
        raise ValueError("❌ Error: No hay datos para exportar.")

    try:
        datos.to_csv(ruta_exportacion, index=False)
        print(f"📁 Datos exportados correctamente a: {ruta_exportacion}")
    except Exception as e:
        print(f"❌ Error al exportar datos: {e}")


def graficar_datos_fuerza_filtrado(forcedecks_data, t_inicio, t_fin):
    """
    Genera un gráfico que muestra las señales de fuerza (Z Left y Z Right)
    en el intervalo de tiempo [t_inicio, t_fin].

    📌 Mejoras añadidas:
    - 🛑 Manejo de error si no hay datos en el intervalo solicitado.
    - 🔍 Mensaje claro en caso de fallos.
    """
    datos_filtrados = forcedecks_data[
        (forcedecks_data['Time'] >= t_inicio) & (forcedecks_data['Time'] <= t_fin)
    ]

    if datos_filtrados.empty:
        raise ValueError(f"❌ Error: No hay datos en el intervalo [{t_inicio:.2f}, {t_fin:.2f}] s.")

    plt.figure(figsize=(14, 6))
    plt.plot(datos_filtrados['Time'], datos_filtrados['Z Left'], label='Z Left (Fuerza)')
    plt.plot(datos_filtrados['Time'], datos_filtrados['Z Right'], label='Z Right (Fuerza)')
    plt.axhline(y=UMBRALES["force_peak"], color='r', linestyle='--', label='Umbral')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Fuerza (N)')
    plt.title(f'Fuerza en ForceDecks [{t_inicio:.2f}, {t_fin:.2f}] s')
    plt.legend()
    plt.grid()
    plt.show()


def graficar_datos_emg(emg_data):
    """
    Genera un gráfico que muestra las señales EMG.
    Se grafica cada canal EMG usando la columna 'Time_1' como eje de tiempo.

    📌 Mejoras añadidas:
    - 🛑 Validación de que 'Time_1' está presente en los datos.
    - 🔍 Notificación clara si no hay canales EMG disponibles.
    """
    if 'Time_1' not in emg_data.columns:
        raise ValueError("❌ Error: La columna 'Time_1' no se encuentra en los datos de EMG.")

    plt.figure(figsize=(14, 6))
    canales_plot = 0

    for col in emg_data.columns:
        if not col.startswith('Time'):
            plt.plot(emg_data['Time_1'], emg_data[col], label=col)
            canales_plot += 1

    if canales_plot == 0:
        raise ValueError("❌ Error: No se encontraron canales EMG en los datos.")

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud EMG (mV)')
    plt.title('Registro de EMG sincronizado')
    plt.legend()
    plt.grid()
    plt.show()


def force_to_video_time(t_force):
    """
    Convierte un tiempo de ForceDecks (t_force) a tiempo en el video (t_video)
    usando la transformación lineal: t_video = a * t_force + b.

    📌 Mejoras añadidas:
    - 🛑 Verificación de que los parámetros de transformación están definidos.
    - 🔍 Notificación clara si la conversión falla.
    """
    a = PARAMETROS_TRANSFORMACION.get("a")
    b = PARAMETROS_TRANSFORMACION.get("b")

    if a is None or b is None:
        raise ValueError("❌ Error: Los parámetros de transformación no han sido definidos.")

    return a * t_force + b

def visualizar_frames_video(ruta_video, t_force_ini, t_force_fin):
    """
    Muestra los frames del video correspondientes a dos instantes de ForceDecks,
    aplicando la transformación lineal (t_video = a * t_force + b) para calcular el
    instante en el video.

    📌 📌 Mejoras añadidas:
    - 🛑 Manejo de errores si el archivo de video no se puede abrir.
    - 🔍 Validación de parámetros de transformación antes de aplicarlos.
    - 🛑 Verificación de que los frames generados son válidos antes de mostrarlos.
    """
    a = PARAMETROS_TRANSFORMACION.get("a")
    b = PARAMETROS_TRANSFORMACION.get("b")

    if a is None or b is None:
        raise ValueError("❌ Error: Los parámetros de transformación no han sido definidos.")

    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        raise ValueError(f"❌ Error: No se pudo abrir el archivo de video en {ruta_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 📌 Transformación lineal de tiempos para video:
    t_video_ini = force_to_video_time(t_force_ini)
    t_video_fin = force_to_video_time(t_force_fin)

    frame_ini = round(t_video_ini * fps)
    frame_fin = round(t_video_fin * fps)

    # Evitar valores fuera del rango del video
    frame_ini = min(max(frame_ini, 0), total_frames - 1)
    frame_fin = min(max(frame_fin, 0), total_frames - 1)

    print(f"\n🔍 Visualizando frames con ajuste lineal:")
    print(f"  📌 ForceDecks Inicio: {t_force_ini:.2f} s -> Video = {t_video_ini:.2f} s -> frame={frame_ini}")
    print(f"  📌 ForceDecks Fin   : {t_force_fin:.2f} s -> Video = {t_video_fin:.2f} s -> frame={frame_fin}")

    # Mostrar el frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ini)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame_rgb)
        plt.title(f"Frame Inicio (video t={t_video_ini:.2f}s)")
        plt.axis("off")
        plt.show()
    else:
        print(f"⚠️ Advertencia: No se pudo leer el frame inicial {frame_ini}.")

    # Mostrar el frame final
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_fin)
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(frame_rgb)
        plt.title(f"Frame Fin (video t={t_video_fin:.2f}s)")
        plt.axis("off")
        plt.show()
    else:
        print(f"⚠️ Advertencia: No se pudo leer el frame final {frame_fin}.")

    cap.release()

def validar_interpolacion(forcedecks_data, datos_resampleados, FRECUENCIA_EMG):
    """
    Valida que el eje temporal uniforme interpolado es correcto.
    - Calcula la diferencia entre muestras consecutivas.
    - Grafica la señal original e interpolada para comparar.
    - Verifica que la variabilidad de los intervalos sea baja.

    📌 Mejoras añadidas:
    - 🛑 Validación de que los datos interpolados no están vacíos.
    - 🔍 Notificación clara si la interpolación genera intervalos inestables.
    """
    if datos_resampleados.empty:
        raise ValueError("❌ Error: No hay datos interpolados para validar.")

    nuevo_tiempo = datos_resampleados['Time_uniforme'].values
    diff_tiempos = np.diff(nuevo_tiempo)

    if len(diff_tiempos) == 0:
        raise ValueError("❌ Error: No hay suficientes puntos interpolados para validar.")

    intervalo_esperado = 1 / FRECUENCIA_EMG
    desviacion = np.std(diff_tiempos)

    print("\n🔍 Validación de interpolación:")
    print(f"  📌 Intervalo esperado entre muestras: {intervalo_esperado:.6f} s")
    print(f"  📌 Valor medio del intervalo: {np.mean(diff_tiempos):.6f} s")
    print(f"  📌 Desviación estándar: {desviacion:.6f} s")

    if desviacion > 0.001:
        print("⚠️ Advertencia: La interpolación puede tener inestabilidad en los intervalos.")

    # Comparación gráfica
    plt.figure(figsize=(14, 6))
    plt.plot(forcedecks_data['Time_adjusted'], forcedecks_data['Z Left'], label='Original Z Left', alpha=0.5)
    plt.plot(datos_resampleados['Time_uniforme'], datos_resampleados['Z Left'], label='Interpolado Z Left', linestyle='--')
    plt.plot(forcedecks_data['Time_adjusted'], forcedecks_data['Z Right'], label='Original Z Right', alpha=0.5)
    plt.plot(datos_resampleados['Time_uniforme'], datos_resampleados['Z Right'], label='Interpolado Z Right', linestyle='--')

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Fuerza (N)')
    plt.title('Comparación de señales originales vs interpoladas')
    plt.legend()
    plt.grid()
    plt.show()

# ---------------------------------------------
# FLUJO PRINCIPAL
# ---------------------------------------------
if __name__ == "__main__":
    # 📌 1️⃣ Cargar datos desde los archivos seleccionados
    forcedecks_data = cargar_datos(RUTA_FORCEDECKS, 'force')
    emg_data = cargar_datos(RUTA_EMG, 'emg')

    if forcedecks_data is None or emg_data is None:
        print("❌ Error: No se pudieron cargar los datos. Saliendo del programa.")
        exit()

    # 📌 2️⃣ Calcular offset de sincronización entre ForceDecks y EMG
    # usando el método pico FD↔pico EMG: primer impulso de salto real (>2x basal)
    # vs primer pico de actividad muscular real (>50% del máximo).
    # La función retorna offset=None si el offset automático no es fiable.
    print("\n🔍 Calculando offset de sincronización por pico FD↔pico EMG...")
    try:
        offset, t_pico_fd, t_pico_emg = detectar_offset_por_pico(forcedecks_data, emg_data)
    except ValueError as e:
        print(f"❌ {e}")
        print("   Saliendo del programa.")
        exit()

    duracion_emg = emg_data['Time_1'].iloc[-1]
    duracion_fd  = forcedecks_data['Time'].max()

    # 📌 2️⃣b Pedir trigger manual si el offset automático no es fiable
    if offset is None:
        print(f"\n📊 Se mostrará la gráfica completa de ForceDecks para que puedas")
        print(f"   identificar el segundo en que se pulsó el trigger (pequeño movimiento")
        print(f"   o pico de fuerza al conectar el EMG).")
        graficar_datos_fuerza_filtrado(forcedecks_data,
                                       forcedecks_data['Time'].iloc[0],
                                       forcedecks_data['Time'].iloc[-1])

        while True:
            try:
                t_trigger_str = input(
                    f"\n⌨️  Introduce el segundo de ForceDecks donde se pulsó el trigger\n"
                    f"    (rango válido: 0 — {duracion_fd:.1f}s; ej.: 84.5): "
                )
                t_trigger_fd = float(t_trigger_str.replace(',', '.'))
                if t_trigger_fd < 0 or t_trigger_fd > duracion_fd:
                    print(f"⚠️  Valor fuera de rango. Introduce un valor entre 0 y {duracion_fd:.1f}s")
                    continue
                # Verificar que el rango resultante cabe en ForceDecks
                if t_trigger_fd + duracion_emg > duracion_fd:
                    print(f"⚠️  Con ese trigger ({t_trigger_fd:.1f}s), el rango necesario llega hasta "
                          f"{t_trigger_fd + duracion_emg:.1f}s pero FD solo dura {duracion_fd:.1f}s.")
                    print(f"   Introduce un valor ≤ {duracion_fd - duracion_emg:.1f}s")
                    continue
                break
            except ValueError:
                print("⚠️  Valor no válido. Introduce un número.")

        # t=0 en EMG corresponde a t_trigger_fd en ForceDecks → offset = t_trigger_fd
        offset = t_trigger_fd
        print(f"  📌 Offset manual: t=0 EMG ↔ t={offset:.3f}s en ForceDecks")
        print(f"  📌 Rango ForceDecks: [{offset:.2f}s — {offset + duracion_emg:.2f}s]")
    else:
        print(f"  ✅ Offset automático aceptado: {offset:.3f}s")

    # 📌 3️⃣ Graficar señales completas para validación visual
    # La gráfica de fuerza muestra toda la grabación
    t_emg_fin = emg_data['Time_1'].iloc[-1]
    graficar_datos_fuerza_filtrado(forcedecks_data, forcedecks_data['Time'].iloc[0], forcedecks_data['Time'].iloc[-1])

    # 📌 4️⃣ Graficar el registro completo de EMG
    graficar_datos_emg(emg_data)

    # 📌 5️⃣ Filtrar ForceDecks al intervalo equivalente al EMG
    # Aplicar el offset: el inicio del EMG (t=0) corresponde a t=offset en ForceDecks
    t_fd_inicio = offset                       # t=0 EMG en escala ForceDecks
    t_fd_fin = offset + t_emg_fin              # t=fin EMG en escala ForceDecks

    print(f"\n⏳ Intervalo del EMG: 0.00 s - {t_emg_fin:.2f} s")
    print(f"⏳ Intervalo equivalente en ForceDecks: {t_fd_inicio:.2f} s - {t_fd_fin:.2f} s")

    forcedecks_data = forcedecks_data[
        (forcedecks_data['Time'] >= t_fd_inicio) & (forcedecks_data['Time'] <= t_fd_fin)
    ]

    # Definir t_emg_inicio como 0 (el EMG empieza en t=0 sintético)
    t_emg_inicio = 0.0

    # 📌 9️⃣ Verificación de consistencia de tiempos de Forcedecks
    print("\n🔍 Verificando consistencia en los tiempos de ForceDecks...")

    if forcedecks_data.empty:
        raise ValueError("❌ Error: No hay datos de ForceDecks en el intervalo seleccionado de EMG.")

    # 1️⃣ Ordenar por Time si está desordenado
    if not forcedecks_data['Time'].is_monotonic_increasing:
        print("⚠️ Advertencia: Los tiempos de ForceDecks no estaban ordenados. Se han ordenado automáticamente.")
        forcedecks_data = forcedecks_data.sort_values(by="Time").reset_index(drop=True)

    # 2️⃣ Eliminar valores duplicados en `Time`
    if forcedecks_data['Time'].duplicated().any():
        print("⚠️ Advertencia: Se encontraron valores duplicados en la columna Time. Se eliminarán para evitar errores en la interpolación.")
        forcedecks_data = forcedecks_data.drop_duplicates(subset=["Time"]).reset_index(drop=True)

    # 3️⃣ Verificar intervalos de tiempo irregulares
    time_diff = np.diff(forcedecks_data['Time'])
    mean_interval = np.mean(time_diff)
    std_interval = np.std(time_diff)

    print(f"  📌 Intervalo medio entre muestras: {mean_interval:.6f} s")
    print(f"  📌 Desviación estándar de intervalos: {std_interval:.6f} s")

    if std_interval > 0.001:  # Umbral de 1 ms de variabilidad
        print("⚠️ Advertencia: Los tiempos no son uniformes. Esto podría afectar la interpolación.")

    # Alinear ForceDecks con EMG aplicando el offset calculado por primer vuelo.
    # Restamos el offset para que t=0 en ForceDecks coincida con t=0 en EMG.
    forcedecks_data = forcedecks_data.copy()
    forcedecks_data['Time'] = forcedecks_data['Time'] - offset

    # 📌 🔟 Interpolar FD a una base temporal uniforme (usando su propio eje de tiempo ajustado)
    print("\n🔄 Preparando la interpolación de ForceDecks...")

    try:
        # 1️⃣ Definir el origen común (t0) como 0 (ambas señales ahora comparten t=0)
        t0 = t_emg_inicio  # = 0.0

        # 2️⃣ Crear una nueva columna 'Time_adjusted' restándole t0 (de modo que el primer registro pase a 0)
        forcedecks_data['Time_adjusted'] = forcedecks_data['Time'] - t0

        # 3️⃣ Definir el intervalo para el eje uniforme
        tiempo_min = forcedecks_data['Time_adjusted'].min()  # Idealmente 0
        tiempo_max = forcedecks_data['Time_adjusted'].max()

        # 4️⃣ Verificación antes de interpolar
        print("\n🔍 Verificando consistencia antes de la interpolación...")

        # 5️⃣ Verificar que el rango de tiempo sea válido
        if tiempo_max - tiempo_min <= 0:
            raise ValueError("❌ Error: El rango de tiempo para interpolar es inválido. Revisa la limpieza de datos.")

        # 6️⃣ Asegurar que hay suficientes datos para interpolar
        if len(forcedecks_data) < 2:
            raise ValueError("❌ Error: No hay suficientes puntos en ForceDecks para realizar la interpolación.")

        # 7️⃣ Crear el eje de tiempo uniforme
        nuevo_tiempo_forcedecks = np.arange(tiempo_min, tiempo_max, 1 / FRECUENCIA_EMG)
        if len(nuevo_tiempo_forcedecks) < 10:  # Umbral arbitrario para evitar problemas
            raise ValueError("❌ Error: El número de muestras interpoladas es demasiado bajo.")

        # 8️⃣ Interpolar las señales de FD sobre este nuevo eje temporal
        datos_resampleados = pd.DataFrame({'Time_uniforme': nuevo_tiempo_forcedecks})
        for col in ['Z Left', 'Z Right']:
            if col in forcedecks_data.columns:
                datos_resampleados[col] = np.interp(nuevo_tiempo_forcedecks,
                                                    forcedecks_data['Time_adjusted'],
                                                    forcedecks_data[col])

        print("✅ Interpolación realizada correctamente.")

    except Exception as e:
        print(f"❌ Error en la interpolación de ForceDecks: {e}")
        exit()

    # 📌 1️⃣1️⃣ VALIDACIÓN DE LA INTERPOLACIÓN
    print("\n🔍 Validando interpolación...")

    try:
        # 1️⃣ Verificar que los datos interpolados no contienen valores anómalos
        for col in ['Z Left', 'Z Right']:
            if col in datos_resampleados.columns:
                max_original = forcedecks_data[col].max()
                min_original = forcedecks_data[col].min()
                max_interpolado = datos_resampleados[col].max()
                min_interpolado = datos_resampleados[col].min()

                if (datos_resampleados[col].isnull().any() or
                    max_interpolado > max_original * 1.5 or
                    min_interpolado < min_original * 1.5):
                    raise ValueError(f"❌ Error: Se detectaron valores anómalos en la interpolación de {col}.")

        # 2️⃣ Comprobar que los intervalos entre muestras interpoladas sean uniformes
        time_diff_interpolado = np.diff(datos_resampleados['Time_uniforme'])
        mean_interval = np.mean(time_diff_interpolado)
        std_interval = np.std(time_diff_interpolado)

        print(f"  📌 Intervalo medio entre muestras interpoladas: {mean_interval:.6f} s")
        print(f"  📌 Desviación estándar de intervalos: {std_interval:.6f} s")

        if std_interval > 0.001:  # Variabilidad mayor a 1 ms
            print("⚠️ Advertencia: La interpolación puede haber introducido variaciones temporales inesperadas.")

        print("✅ Validación de interpolación completada correctamente.")

    except Exception as e:
        print(f"❌ Error en la validación de la interpolación: {e}")
        exit()

    # 📌 1️⃣2️⃣ Interpolación final sobre el eje de tiempo del EMG:
    print("\n🔍 Verificando alineación con EMG antes de fusionar...")

    try:
        # 1️⃣ Comprobar que Time_1 de EMG está correctamente definido
        if 'Time_1' not in emg_data.columns:
            raise ValueError("❌ Error: La columna Time_1 no se encuentra en los datos de EMG.")

        # 2️⃣ Convertir a la escala de tiempo común
        tiempos_emg = emg_data['Time_1'].values - t0
        if len(tiempos_emg) == 0:
            raise ValueError("❌ Error: No se encontraron tiempos en los datos de EMG.")

        # 3️⃣ Crear DataFrame con los tiempos del EMG
        datos_interpolados_fd = pd.DataFrame({'Time_common': tiempos_emg})

        # 4️⃣ Interpolar ForceDecks sobre Time_common
        for col in ['Z Left', 'Z Right']:
            if col in datos_resampleados.columns:
                datos_interpolados_fd[col] = np.interp(tiempos_emg,
                                                    datos_resampleados['Time_uniforme'],
                                                    datos_resampleados[col])

        # 5️⃣ Comprobar que no hay valores NaN en los datos interpolados
        if datos_interpolados_fd.isnull().any().any():
            raise ValueError("❌ Error: Se detectaron valores NaN en la interpolación de ForceDecks.")

        print("✅ Interpolación sobre Time_common realizada correctamente.")

    except Exception as e:
        print(f"❌ Error en la interpolación sobre Time_common: {e}")
        exit()

    # 📌 1️⃣3️⃣ VALIDACIÓN DESPUÉS DE FUSIONAR LOS DATOS
    print("\n🔍 Fusionando ForceDecks y EMG...")

    try:
        # 1️⃣ Eliminar columnas de tiempo originales en EMG y resetear índice
        emg_data_unificado = emg_data.drop(
            [col for col in emg_data.columns if col.startswith('Time')], axis=1
        ).reset_index(drop=True)

        # 2️⃣ Añadir columna Z Total y resetear índice
        datos_interpolados_fd['Z Total'] = datos_interpolados_fd['Z Left'] + datos_interpolados_fd['Z Right']
        datos_interpolados_fd = datos_interpolados_fd.reset_index(drop=True)

        # 3️⃣ Verificar que las longitudes coinciden antes de concatenar
        if len(datos_interpolados_fd) != len(emg_data_unificado):
            raise ValueError(
                f"❌ Error: ForceDecks tiene {len(datos_interpolados_fd)} muestras "
                f"y EMG tiene {len(emg_data_unificado)}. No coinciden."
            )

        # 4️⃣ Concatenar con índices reseteados
        datos_sincronizados = pd.concat([datos_interpolados_fd, emg_data_unificado], axis=1)

        # 3️⃣ Verificar que las longitudes coinciden
        if len(datos_interpolados_fd) != len(emg_data_unificado):
            raise ValueError("❌ Error: La cantidad de muestras en ForceDecks y EMG no coincide tras la sincronización.")

        print("✅ Fusión de datos completada correctamente.")

    except Exception as e:
        print(f"❌ Error en la fusión de datos: {e}")
        exit()

    # 📌 Graficar para validación visual
    try:
        plt.figure(figsize=(14, 6))
        plt.plot(datos_sincronizados['Time_common'], datos_sincronizados['Z Left'], label="Interpolado Z Left")
        plt.plot(datos_sincronizados['Time_common'], datos_sincronizados['Z Right'], label="Interpolado Z Right")
        plt.xlabel("Tiempo común (s)")
        plt.ylabel("Fuerza (N)")
        plt.title("Validación de alineación ForceDecks con EMG")
        plt.legend()
        plt.grid()
        plt.show()
    except Exception as e:
        print(f"⚠️ Advertencia: No se pudo generar la gráfica de validación. Error: {e}")

    # 📌 1️⃣4️⃣ Verificación antes de exportar
    print("\n🔍 Última verificación antes de exportar...")

    try:
        # 1️⃣ Comprobar que no hay valores NaN en los datos finales
        if datos_sincronizados.isnull().values.any():
            raise ValueError("❌ Error: Se detectaron valores NaN en los datos sincronizados. Revisa la interpolación.")

        # 2️⃣ Asegurar que ForceDecks y EMG tienen la misma cantidad de muestras
        if len(datos_interpolados_fd) != len(emg_data_unificado):
            raise ValueError("❌ Error: La cantidad de muestras en ForceDecks y EMG no coincide tras la sincronización.")

        # 3️⃣ Verificar que los datos interpolados cubren todo el intervalo temporal
        t_min = datos_sincronizados['Time_common'].min()
        t_max = datos_sincronizados['Time_common'].max()
        if t_min > tiempos_emg.min() or t_max < tiempos_emg.max():
            raise ValueError("❌ Error: La interpolación no cubre completamente el rango de tiempo del EMG.")

        print("✅ Datos validados. Listos para la exportación.")

        exportar_datos_sincroniz(datos_sincronizados, RUTA_EXPORTACION_CSV)
        print(f"📁 Datos exportados correctamente a: {RUTA_EXPORTACION_CSV}")

    except Exception as e:
        print(f"❌ Error en la validación final de datos antes de exportar: {e}")
        exit()

    print("\n✅ ¡Proceso finalizado con éxito!")