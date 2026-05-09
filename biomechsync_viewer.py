# ---------------------------------------------
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ---------------------------------------------

import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QSlider, QWidget, QHBoxLayout, QComboBox, QInputDialog, QGroupBox,
    QCheckBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector


# ---------------------------------------------
# CLASE PRINCIPAL: SignalViewer
# ---------------------------------------------

class SignalViewer(QMainWindow):
    """
    Visor principal para datos biomecánicos sincronizados.

    Lee el CSV exportado por el código de interpolación con formato:
        Time_common | Z Left | Z Right | Z Total | <músculo_1> ... <músculo_N>

    Los nombres de los canales EMG se detectan automáticamente al cargar el CSV:
    cualquier columna que no sea Time_common ni empiece por Z se trata como canal EMG.

    Novedades respecto a la versión anterior:
    - ✅ Lee el CSV interpolado (columnas como VL_D, BF_D, GM_D, etc.)
    - ✅ Selector de canales EMG: checkboxes para elegir cuáles se visualizan
    - ✅ Selector de ForceDecks: Z Left / Z Right / Z Total (combo)
    - ✅ Sincronización vídeo-señales intacta
    - ✅ Selección y exportación de fases conservada
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visor Señales Biomecánicas — ForceDecks + EMG + Vídeo")
        self.setGeometry(100, 100, 1700, 860)

        # ------------------------------------------------------------------
        # VARIABLES DE ESTADO
        # ------------------------------------------------------------------
        self.data = None
        self.selected_phases = []
        self.selection_mode = False
        self.phase_action = "Añadir fase"

        # Paleta de colores fija para canales EMG (mismo orden que matplotlib)
        self.EMG_COLORS = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        ]
        # Mapeo color por nombre de canal (se rellena al cargar CSV)
        self._emg_color_map = {}

        # Canales detectados en el CSV
        self.emg_cols_available = []      # Todos los canales EMG presentes en el CSV
        self.emg_cols_selected  = []      # Canales EMG actualmente visibles
        self.fd_cols_available  = []      # Canales FD presentes en el CSV
        self.fd_cols_selected   = ["Z Total", "Z Left", "Z Right"]  # FD visibles al inicio

        # Widgets de los checkboxes (se crean dinámicamente al cargar CSV)
        self._emg_checkboxes = {}         # {nombre_col: QCheckBox}
        self._fd_checkboxes  = {}         # {nombre_col: QCheckBox}

        # Vídeo
        self.VID_START      = 0.0
        self.VID_END        = 9999.0
        self.video_cap      = None
        self.video_path     = None
        self.video_fps      = 29.97
        self.video_playing  = False
        self.t_primer_vuelo = 0.0

        self.video_timer = QTimer()
        self.video_timer.setInterval(33)
        self.video_timer.timeout.connect(self._next_frame)

        # Referencias a las líneas rojas de posición actual
        self.vertical_line_emg = None
        self.vertical_line_fd  = None

        # Modo oscuro/claro
        self._dark_mode = False
        self.original_ax2_xlim = None
        self.original_ax1_ylim = None
        self.original_ax2_ylim = None

        # ------------------------------------------------------------------
        # CONSTRUCCIÓN DE LA INTERFAZ
        # ------------------------------------------------------------------
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # ── COLUMNA IZQUIERDA: gráficos + controles ──────────────────────
        self.left_widget = QWidget()
        self.left_layout = QVBoxLayout(self.left_widget)
        self.main_layout.addWidget(self.left_widget, stretch=3)

        # Figura matplotlib (2 subplots compartiendo eje X)
        # stretch=1 en el canvas para que se expanda con la ventana
        self.figure = Figure()
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212, sharex=self.ax1)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.left_layout.addWidget(self.nav_toolbar)
        self.left_layout.addWidget(self.canvas, stretch=1)

        # ── CONTROLES DE RATÓN ────────────────────────────────────────────
        # - Rueda del ratón:        zoom horizontal centrado en el cursor
        # - Clic izquierdo + drag:  pan (desplazarse)
        # - Doble clic izquierdo:   restaurar zoom completo
        self.canvas.mpl_connect('scroll_event',        self._on_scroll)
        self.canvas.mpl_connect('button_press_event',  self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event',self._on_mouse_release)

        # Estado del pan
        self._pan_active  = False
        self._pan_x_start = None
        self._pan_xlim_start = None

        # ── BARRA DE CONTROLES SUPERIOR ───────────────────────────────────
        controls_layout = QHBoxLayout()

        # Grupo Archivos
        group_archivos = QGroupBox("Archivos")
        archivos_layout = QVBoxLayout()
        self.btn_load_csv = QPushButton("Cargar CSV")
        self.btn_load_csv.clicked.connect(self.load_file)
        archivos_layout.addWidget(self.btn_load_csv)
        self.btn_load_video = QPushButton("Cargar Video")
        self.btn_load_video.clicked.connect(self.load_video)
        archivos_layout.addWidget(self.btn_load_video)
        group_archivos.setLayout(archivos_layout)
        controls_layout.addWidget(group_archivos)

        # Grupo Reproducción
        group_repro = QGroupBox("Reproducción")
        repro_layout = QVBoxLayout()
        self.btn_play = QPushButton("Play/Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        repro_layout.addWidget(self.btn_play)
        self.btn_reset_video = QPushButton("Reiniciar Video")
        self.btn_reset_video.clicked.connect(self.reset_video_position)
        repro_layout.addWidget(self.btn_reset_video)
        group_repro.setLayout(repro_layout)
        controls_layout.addWidget(group_repro)

        # Grupo Análisis
        group_analisis = QGroupBox("Análisis")
        analisis_layout = QVBoxLayout()
        self.btn_select_phase = QPushButton("Seleccionar fase")
        self.btn_select_phase.setCheckable(True)
        self.btn_select_phase.toggled.connect(self.toggle_selection_mode)
        analisis_layout.addWidget(self.btn_select_phase)
        self.phase_action_selector = QComboBox()
        self.phase_action_selector.addItems(["Añadir fase", "Borrar fase"])
        self.phase_action_selector.currentTextChanged.connect(self.update_phase_action)
        analisis_layout.addWidget(self.phase_action_selector)
        self.btn_export = QPushButton("Exportar Selección")
        self.btn_export.clicked.connect(self.export_selected_phases)
        analisis_layout.addWidget(self.btn_export)
        group_analisis.setLayout(analisis_layout)
        controls_layout.addWidget(group_analisis)

        # Grupo Vista
        group_vista = QGroupBox("Vista")
        vista_layout = QVBoxLayout()

        self.btn_reestablecer_zoom = QPushButton("Restaurar Zoom")
        self.btn_reestablecer_zoom.clicked.connect(self.restore_zoom)
        vista_layout.addWidget(self.btn_reestablecer_zoom)

        self.btn_dark_mode = QPushButton("Modo oscuro")
        self.btn_dark_mode.setCheckable(True)
        self.btn_dark_mode.toggled.connect(self.toggle_dark_mode)
        vista_layout.addWidget(self.btn_dark_mode)

        # Slider grosor de línea
        grosor_layout = QHBoxLayout()
        grosor_layout.addWidget(QLabel("Grosor:"))
        self.slider_grosor = QSlider(Qt.Horizontal)
        self.slider_grosor.setMinimum(1)
        self.slider_grosor.setMaximum(5)
        self.slider_grosor.setValue(1)
        self.slider_grosor.setTickInterval(1)
        self.slider_grosor.setFixedWidth(80)
        self.slider_grosor.sliderReleased.connect(self.initial_plot)
        grosor_layout.addWidget(self.slider_grosor)
        vista_layout.addLayout(grosor_layout)

        group_vista.setLayout(vista_layout)
        controls_layout.addWidget(group_vista)

        self.left_layout.addLayout(controls_layout)

        # ── SLIDER ────────────────────────────────────────────────────────
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.slider_changed)
        self.left_layout.addWidget(self.slider)

        # ── ETIQUETAS DE ESTADO ───────────────────────────────────────────
        self.label_status   = QLabel("No se ha cargado ningún archivo.")
        self.left_layout.addWidget(self.label_status)
        self.label_timeinfo = QLabel("t_Video: 0.00 s | t_CSV: 0.00 s")
        self.left_layout.addWidget(self.label_timeinfo)

        # ── COLUMNA DERECHA: vídeo + selector EMG ─────────────────────────
        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.main_layout.addWidget(self.right_widget, stretch=2)

        # Visor de vídeo
        self.video_label = QLabel("Sin vídeo cargado")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.right_layout.addWidget(self.video_label, stretch=3)

        # ── PANEL DE CHECKBOXES EMG (scroll) ─────────────────────────────
        group_emg_sel = QGroupBox("Canales EMG visibles")
        emg_sel_layout = QVBoxLayout()

        # Botones Todos / Ninguno
        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Todos")
        btn_none = QPushButton("Ninguno")
        btn_all.clicked.connect(self._select_all_emg)
        btn_none.clicked.connect(self._deselect_all_emg)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        emg_sel_layout.addLayout(btn_row)

        # Área con scroll para los checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setFixedHeight(180)
        self._emg_checkbox_container = QWidget()
        self._emg_checkbox_layout    = QVBoxLayout(self._emg_checkbox_container)
        self._emg_checkbox_layout.setContentsMargins(4, 4, 4, 4)
        self._emg_checkbox_layout.setSpacing(2)
        scroll.setWidget(self._emg_checkbox_container)
        emg_sel_layout.addWidget(scroll)

        group_emg_sel.setLayout(emg_sel_layout)
        self.right_layout.addWidget(group_emg_sel, stretch=0)

        # ── PANEL DE CHECKBOXES FORCEDECKS ───────────────────────────────
        group_fd_sel = QGroupBox("Canales ForceDecks visibles")
        fd_sel_layout = QVBoxLayout()

        # Botones Todos / Ninguno
        btn_row_fd = QHBoxLayout()
        btn_all_fd  = QPushButton("Todos")
        btn_none_fd = QPushButton("Ninguno")
        btn_all_fd.clicked.connect(self._select_all_fd)
        btn_none_fd.clicked.connect(self._deselect_all_fd)
        btn_row_fd.addWidget(btn_all_fd)
        btn_row_fd.addWidget(btn_none_fd)
        fd_sel_layout.addLayout(btn_row_fd)

        # Área con scroll para los checkboxes FD
        scroll_fd = QScrollArea()
        scroll_fd.setWidgetResizable(True)
        scroll_fd.setFrameShape(QFrame.NoFrame)
        scroll_fd.setFixedHeight(90)
        self._fd_checkbox_container = QWidget()
        self._fd_checkbox_layout    = QVBoxLayout(self._fd_checkbox_container)
        self._fd_checkbox_layout.setContentsMargins(4, 4, 4, 4)
        self._fd_checkbox_layout.setSpacing(2)
        scroll_fd.setWidget(self._fd_checkbox_container)
        fd_sel_layout.addWidget(scroll_fd)

        group_fd_sel.setLayout(fd_sel_layout)
        self.right_layout.addWidget(group_fd_sel, stretch=0)

    # ==================================================================
    # CONTROLES DE RATÓN — zoom y pan
    # ==================================================================

    def _on_scroll(self, event):
        """
        Rueda del ratón → zoom horizontal centrado en la posición del cursor.
        Hacia arriba = acercar, hacia abajo = alejar.
        """
        if self.data is None or event.xdata is None:
            return
        if self.original_ax1_xlim is None:
            return

        x_min, x_max = self.ax1.get_xlim()
        x_range = x_max - x_min
        x_cursor = event.xdata

        FACTOR = 0.85  # 15% de zoom por paso de rueda
        if event.button == 'up':
            new_range = x_range * FACTOR
        elif event.button == 'down':
            new_range = x_range / FACTOR
        else:
            return

        # Clamp: no alejarse más que el rango total original
        x_total = self.original_ax1_xlim[1] - self.original_ax1_xlim[0]
        new_range = min(new_range, x_total)

        # Centrar el zoom en la posición del cursor
        ratio = (x_cursor - x_min) / x_range if x_range > 0 else 0.5
        new_min = x_cursor - ratio * new_range
        new_max = x_cursor + (1 - ratio) * new_range

        # Clamp a los límites originales
        if new_min < self.original_ax1_xlim[0]:
            new_min = self.original_ax1_xlim[0]
            new_max = new_min + new_range
        if new_max > self.original_ax1_xlim[1]:
            new_max = self.original_ax1_xlim[1]
            new_min = new_max - new_range

        self.ax1.set_xlim(new_min, new_max)
        self.ax2.set_xlim(new_min, new_max)
        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        """
        Botón derecho: inicia el pan.
        Doble clic izquierdo: restaura el zoom completo.
        """
        # Doble clic izquierdo → restaurar zoom
        if event.button == 1 and event.dblclick:
            self.restore_zoom()
            return

        # Botón derecho → iniciar pan
        if event.button == 3 and event.xdata is not None:
            # Compatibilidad con barra de herramientas
            if self.nav_toolbar.mode != '':
                return
            self._pan_active     = True
            self._pan_x_start    = event.xdata
            self._pan_xlim_start = self.ax1.get_xlim()

    def _on_mouse_move(self, event):
        """
        Arrastra con botón derecho → desplaza la vista (pan).
        """
        if not self._pan_active or event.xdata is None:
            return
        if self._pan_x_start is None or self._pan_xlim_start is None:
            return

        dx = self._pan_x_start - event.xdata
        new_min = self._pan_xlim_start[0] + dx
        new_max = self._pan_xlim_start[1] + dx

        # Clamp a los límites originales
        if self.original_ax1_xlim is not None:
            x_total_min = self.original_ax1_xlim[0]
            x_total_max = self.original_ax1_xlim[1]
            x_vis = new_max - new_min
            if new_min < x_total_min:
                new_min = x_total_min
                new_max = new_min + x_vis
            if new_max > x_total_max:
                new_max = x_total_max
                new_min = new_max - x_vis

        self.ax1.set_xlim(new_min, new_max)
        self.ax2.set_xlim(new_min, new_max)
        self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        """Finaliza el pan al soltar el botón derecho."""
        if event.button == 3:
            self._pan_active     = False
            self._pan_x_start    = None
            self._pan_xlim_start = None

    def toggle_dark_mode(self, enabled):
        """Alterna entre modo oscuro y claro en los gráficos."""
        self._dark_mode = enabled
        self.btn_dark_mode.setText("Modo claro" if enabled else "Modo oscuro")

        if enabled:
            self.figure.patch.set_facecolor('#1e1e1e')
            for ax in [self.ax1, self.ax2]:
                ax.set_facecolor('#2d2d2d')
                ax.tick_params(colors='#cccccc')
                ax.xaxis.label.set_color('#cccccc')
                ax.yaxis.label.set_color('#cccccc')
                ax.title.set_color('#ffffff')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#555555')
        else:
            self.figure.patch.set_facecolor('white')
            for ax in [self.ax1, self.ax2]:
                ax.set_facecolor('white')
                ax.tick_params(colors='black')
                ax.xaxis.label.set_color('black')
                ax.yaxis.label.set_color('black')
                ax.title.set_color('black')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')

        self.canvas.draw_idle()

    # ==================================================================
    # CARGA DE ARCHIVOS
    # ==================================================================

    def load_file(self):
        """
        Carga el CSV interpolado exportado por el código de interpolación.

        Formato esperado:
            Time_common | Z Left | Z Right | Z Total | <EMG_cols...>

        Los canales EMG se detectan automáticamente: cualquier columna
        que no sea Time_common y no empiece por 'Z' es tratada como EMG.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar CSV interpolado", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path, delimiter=',')

            # ── Validar columnas mínimas ──────────────────────────────
            required = {'Time_common', 'Z Left', 'Z Right'}
            missing = required - set(self.data.columns)
            if missing:
                self.label_status.setText(
                    f"❌ Columnas faltantes en el CSV: {missing}. "
                    f"¿Es el archivo correcto?"
                )
                self.data = None
                return

            # Añadir Z Total si no está (compatibilidad con CSVs antiguos)
            if 'Z Total' not in self.data.columns:
                self.data['Z Total'] = self.data['Z Left'] + self.data['Z Right']

            # ── Corrección de escala de fuerza (kN → N) ──────────────
            for col in ['Z Left', 'Z Right', 'Z Total']:
                if col in self.data.columns and self.data[col].abs().max() < 10:
                    self.data[col] *= 1000

            # ── Detectar canales EMG ──────────────────────────────────
            fd_cols = {'Time_common', 'Z Left', 'Z Right', 'Z Total'}
            self.emg_cols_available = [
                c for c in self.data.columns if c not in fd_cols
            ]
            self.emg_cols_selected = list(self.emg_cols_available)  # Todos visibles al inicio

            # ── Detectar canales ForceDecks disponibles ───────────────
            self.fd_cols_available = [
                c for c in ['Z Total', 'Z Left', 'Z Right']
                if c in self.data.columns
            ]
            self.fd_cols_selected = list(self.fd_cols_available)  # Todos visibles al inicio

            # ── Reconstruir checkboxes EMG y FD ──────────────────────
            self._rebuild_emg_checkboxes()
            self._rebuild_fd_checkboxes()

            # ── Slider ───────────────────────────────────────────────
            self.slider.setEnabled(True)
            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.data) - 1)

            # Resetear límites originales para que initial_plot los recalcule
            self.original_ax1_xlim = None
            self.original_ax2_xlim = None
            self.original_ax1_ylim = None
            self.original_ax2_ylim = None

            # ── Detectar primer vuelo real ────────────────────────────
            # Estrategia robusta en 2 pasos:
            # 1. Encontrar el primer impulso real (>2x basal) → marca el
            #    inicio de los saltos, descartando actividad previa.
            # 2. Encontrar el primer vuelo (fuerza<30% basal) con duración
            #    >0.1s que ocurra después del primer impulso.
            # Esto elimina falsos positivos de sujetos como SFM y GLS.
            try:
                fuerza_total = self.data['Z Left'] + self.data['Z Right']
                basal        = fuerza_total[self.data['Time_common'] <= 5].mean()

                # Paso 1: primer impulso real (>2x basal)
                impulsos = self.data[
                    (self.data['Time_common'] > 5) & (fuerza_total > basal * 2.0)
                ]
                t_primer_impulso = 0.0
                if not impulsos.empty:
                    idx_imp    = impulsos.index.tolist()
                    grupos_imp = [idx_imp[0]]
                    for i in range(1, len(idx_imp)):
                        if idx_imp[i] - idx_imp[i-1] > 500:
                            grupos_imp.append(idx_imp[i])
                    t_primer_impulso = self.data.loc[grupos_imp[0], 'Time_common']

                # Paso 2: primer vuelo real (>0.1s de duración, post-impulso)
                vuelos = self.data[
                    (self.data['Time_common'] > 5) & (fuerza_total < basal * 0.30)
                ]
                t_vuelo_real = 0.0
                if not vuelos.empty:
                    idx_v    = vuelos.index.tolist()
                    grupos_v = []
                    g = [idx_v[0]]
                    for i in range(1, len(idx_v)):
                        if idx_v[i] - idx_v[i-1] > 50:
                            grupos_v.append((
                                self.data.loc[g[0],  'Time_common'],
                                self.data.loc[g[-1], 'Time_common']
                            ))
                            g = []
                        g.append(idx_v[i])
                    if g:
                        grupos_v.append((
                            self.data.loc[g[0],  'Time_common'],
                            self.data.loc[g[-1], 'Time_common']
                        ))

                    for t0, t1 in grupos_v:
                        dur = t1 - t0
                        if t0 >= t_primer_impulso - 2.0 and dur > 0.1:
                            t_vuelo_real = t0
                            break

                self.t_primer_vuelo = t_vuelo_real

            except Exception:
                self.t_primer_vuelo = 0.0

            # Actualizar título con nombre del archivo
            nombre_archivo = file_path.replace('\\', '/').split('/')[-1]
            self.setWindowTitle(f"Visor Biomecánico — {nombre_archivo}")

            self.label_status.setText(
                f"✅ CSV cargado: {file_path} | "
                f"{len(self.emg_cols_available)} canales EMG: "
                f"{', '.join(self.emg_cols_available)} | "
                f"Primer vuelo real: t={self.t_primer_vuelo:.3f}s"
            )

            self.initial_plot()

        except Exception as e:
            self.label_status.setText(f"❌ Error al cargar el archivo: {e}")

    # ------------------------------------------------------------------
    # GESTIÓN DE CHECKBOXES EMG
    # ------------------------------------------------------------------

    def _rebuild_emg_checkboxes(self):
        """
        Elimina los checkboxes anteriores y crea uno por cada canal EMG
        detectado en el CSV. Todos empiezan marcados (visibles).
        El color del texto del checkbox coincide con el color de la línea en el gráfico.
        """
        # Vaciar layout anterior
        for cb in self._emg_checkboxes.values():
            self._emg_checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._emg_checkboxes.clear()

        # Asignar colores fijos por canal
        self._emg_color_map = {
            col: self.EMG_COLORS[i % len(self.EMG_COLORS)]
            for i, col in enumerate(self.emg_cols_available)
        }

        for col in self.emg_cols_available:
            cb = QCheckBox(col)
            cb.setChecked(True)
            color = self._emg_color_map.get(col, '#333333')
            cb.setStyleSheet(f"color: {color}; font-weight: bold;")
            cb.stateChanged.connect(self._on_emg_checkbox_changed)
            self._emg_checkbox_layout.addWidget(cb)
            self._emg_checkboxes[col] = cb

        # Empujar checkboxes al inicio del layout
        self._emg_checkbox_layout.addStretch()

    def _on_emg_checkbox_changed(self):
        """Actualiza la lista de canales visibles y redibuja los gráficos."""
        self.emg_cols_selected = [
            col for col, cb in self._emg_checkboxes.items() if cb.isChecked()
        ]
        self.initial_plot()

    def _select_all_emg(self):
        """Marca todos los checkboxes EMG."""
        for cb in self._emg_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.emg_cols_selected = list(self.emg_cols_available)
        self.initial_plot()

    def _deselect_all_emg(self):
        """Desmarca todos los checkboxes EMG."""
        for cb in self._emg_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.emg_cols_selected = []
        self.initial_plot()

    # ------------------------------------------------------------------
    # GESTIÓN DE CHECKBOXES FORCEDECKS
    # ------------------------------------------------------------------

    def _rebuild_fd_checkboxes(self):
        """Elimina los checkboxes FD anteriores y crea uno por canal disponible."""
        for cb in self._fd_checkboxes.values():
            self._fd_checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self._fd_checkboxes.clear()

        color_map = {'Z Total': '#2ca02c', 'Z Left': '#1f77b4', 'Z Right': '#ff7f0e'}
        for col in self.fd_cols_available:
            cb = QCheckBox(col)
            cb.setChecked(True)
            # Color del texto igual al color de la línea en el gráfico
            color = color_map.get(col, '#333333')
            cb.setStyleSheet(f"color: {color}; font-weight: bold;")
            cb.stateChanged.connect(self._on_fd_checkbox_changed)
            self._fd_checkbox_layout.addWidget(cb)
            self._fd_checkboxes[col] = cb

        self._fd_checkbox_layout.addStretch()

    def _on_fd_checkbox_changed(self):
        """Actualiza la lista de canales FD visibles y redibuja."""
        self.fd_cols_selected = [
            col for col, cb in self._fd_checkboxes.items() if cb.isChecked()
        ]
        self.initial_plot()

    def _select_all_fd(self):
        """Marca todos los checkboxes FD."""
        for cb in self._fd_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.fd_cols_selected = list(self.fd_cols_available)
        self.initial_plot()

    def _deselect_all_fd(self):
        """Desmarca todos los checkboxes FD."""
        for cb in self._fd_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.fd_cols_selected = []
        self.initial_plot()

    # ==================================================================
    # CARGA DE VÍDEO
    # ==================================================================

    def load_video(self):
        """
        Carga un vídeo y lo sincroniza con el CSV mediante un evento de referencia
        común: el usuario introduce el segundo del mismo evento tanto en el vídeo
        como en el CSV (puede ser cualquier despegue, pico de fuerza, etc.).

        VID_START = t_evento_video - t_evento_csv
        """
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Cargar Video", "", "Video Files (*.mp4 *.avi *.mts *.MP4)"
        )
        if not video_path:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.label_status.setText("❌ Error: No se pudo abrir el vídeo.")
            return

        self.video_cap  = cap
        self.video_path = video_path
        self.video_fps  = cap.get(cv2.CAP_PROP_FPS) or 29.97
        self.video_timer.setInterval(int(1000 / self.video_fps))

        # ── Paso 1: segundo del evento en el CSV ─────────────────────
        t_evento_csv, ok = QInputDialog.getDouble(
            self,
            "Sincronización — Evento en CSV",
            f"Segundo del CSV donde ocurre el evento de referencia\n"
            f"(míralo en el gráfico; primer vuelo real detectado: t={self.t_primer_vuelo:.3f}s):",
            self.t_primer_vuelo,
            0, 9999, 3
        )
        if not ok:
            self.label_status.setText("⚠️ Carga de vídeo cancelada.")
            return

        # ── Paso 2: segundo del mismo evento en el vídeo ─────────────
        t_evento_video, ok = QInputDialog.getDouble(
            self,
            "Sincronización — Evento en vídeo",
            f"Segundo del vídeo donde ocurre el mismo evento de referencia\n"
            f"(el segundo exacto que identificaste en Clipchamp):",
            0.0, 0, 9999, 3
        )
        if not ok:
            self.label_status.setText("⚠️ Carga de vídeo cancelada.")
            return

        # ── Calcular VID_START ────────────────────────────────────────
        self.VID_START = t_evento_video - t_evento_csv
        if self.data is not None:
            duracion_csv = self.data['Time_common'].iloc[-1]
            self.VID_END = self.VID_START + duracion_csv
        else:
            self.VID_END = self.VID_START + 9999.0

        cap.set(cv2.CAP_PROP_POS_MSEC, self.VID_START * 1000)
        ret, frame = cap.read()
        if ret:
            self._show_frame(frame)

        self.label_status.setText(
            f"✅ Vídeo cargado: {video_path} | "
            f"Evento CSV={t_evento_csv:.3f}s | "
            f"Evento vídeo={t_evento_video:.3f}s | "
            f"VID_START={self.VID_START:.2f}s"
        )

    def _show_frame(self, frame):
        """Convierte un frame OpenCV (BGR) → QPixmap y lo muestra."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch  = frame_rgb.shape
        qt_image  = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap    = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _next_frame(self):
        """Lee y muestra el siguiente frame durante la reproducción."""
        if self.video_cap is None:
            return
        ret, frame = self.video_cap.read()
        if not ret:
            self.video_timer.stop()
            self.video_playing = False
            return

        t_video_s = self.video_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if t_video_s >= self.VID_END:
            self.video_timer.stop()
            self.video_playing = False
            self.label_status.setText("Reproducción finalizada.")
            return

        self._show_frame(frame)
        self._update_graphs_from_video(t_video_s)

    # ==================================================================
    # CONTROLES DE REPRODUCCIÓN
    # ==================================================================

    def toggle_play(self):
        if self.video_cap is None:
            return
        if self.video_playing:
            self.video_timer.stop()
            self.video_playing = False
        else:
            self.video_timer.start()
            self.video_playing = True

    def reset_video_position(self):
        if self.video_cap is None:
            return
        self.video_timer.stop()
        self.video_playing = False
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, self.VID_START * 1000)
        ret, frame = self.video_cap.read()
        if ret:
            self._show_frame(frame)
        self.label_status.setText("Vídeo reiniciado a VID_START.")

    def restore_zoom(self):
        if self.original_ax1_xlim and self.original_ax2_xlim:
            self.ax1.set_xlim(self.original_ax1_xlim)
            self.ax2.set_xlim(self.original_ax2_xlim)
        if self.original_ax1_ylim:
            self.ax1.set_ylim(self.original_ax1_ylim)
        if self.original_ax2_ylim:
            self.ax2.set_ylim(self.original_ax2_ylim)
        self.figure.canvas.draw()

    # ==================================================================
    # SINCRONIZACIÓN VÍDEO → GRÁFICOS
    # ==================================================================

    def _update_graphs_from_video(self, t_video_s):
        """
        Mueve las líneas rojas y el slider a la posición
        correspondiente al tiempo actual del vídeo.
        """
        if self.data is None or self.data.empty:
            return

        csv_time = max(0.0, t_video_s - self.VID_START)
        idx = (self.data['Time_common'] - csv_time).abs().idxmin()

        if idx < 0 or idx >= len(self.data):
            return

        current_time = self.data['Time_common'].iloc[idx]

        if self.vertical_line_emg is not None and self.vertical_line_fd is not None:
            self.vertical_line_emg.set_xdata([current_time, current_time])
            self.vertical_line_fd.set_xdata([current_time, current_time])
            self.canvas.draw()

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        self.label_timeinfo.setText(
            f"t_Video: {t_video_s:.2f}s | t_CSV: {current_time:.2f}s"
        )

    def slider_changed(self, value):
        """Mueve el vídeo al frame correspondiente al valor del slider."""
        if self.data is None or self.data.empty:
            return
        if value < 0 or value >= len(self.data):
            return

        csv_time  = self.data['Time_common'].iloc[value]
        t_video_s = max(self.VID_START, min(csv_time + self.VID_START, self.VID_END))

        if self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_MSEC, t_video_s * 1000)
            ret, frame = self.video_cap.read()
            if ret:
                self._show_frame(frame)

        self.label_timeinfo.setText(
            f"t_Video: {t_video_s:.2f}s | t_CSV: {csv_time:.2f}s"
        )

    # ==================================================================
    # PLOTEO PRINCIPAL
    # ==================================================================

    def initial_plot(self):
        """
        Redibuja los dos subplots usando los canales actualmente seleccionados.
        Preserva el zoom actual si ya había datos cargados previamente.
        """
        # Guardar zoom actual ANTES de redibujar
        zoom_actual = None
        if self.original_ax1_xlim is not None:
            current_xlim = self.ax1.get_xlim()
            current_ylim1 = self.ax1.get_ylim()
            current_ylim2 = self.ax2.get_ylim()
            # Solo guardar si el zoom es diferente al original (hay zoom activo)
            es_zoom_activo = (
                abs(current_xlim[0] - self.original_ax1_xlim[0]) > 0.01 or
                abs(current_xlim[1] - self.original_ax1_xlim[1]) > 0.01
            )
            if es_zoom_activo:
                zoom_actual = (current_xlim, current_ylim1, current_ylim2)

        self.ax1.clear()
        self.ax2.clear()

        if self.data is None:
            return

        time = self.data['Time_common']
        lw   = self.slider_grosor.value() * 0.5  # 0.5, 1.0, 1.5, 2.0, 2.5

        # ── Subplot 1: EMG ────────────────────────────────────────────
        if self.emg_cols_selected:
            # Ordenar por amplitud de mayor a menor
            cols_amp = sorted(
                self.emg_cols_selected,
                key=lambda c: self.data[c].max() - self.data[c].min(),
                reverse=True
            )
            for col in cols_amp:
                color = self._emg_color_map.get(col, None)
                self.ax1.plot(time, self.data[col], label=col,
                              linewidth=lw, color=color)
        else:
            self.ax1.text(
                0.5, 0.5, "Sin canales EMG seleccionados",
                ha='center', va='center', transform=self.ax1.transAxes,
                color='gray', fontsize=10
            )

        self.ax1.set_title("Señales EMG")
        self.ax1.set_xlabel("Tiempo (s)")
        self.ax1.set_ylabel("Amplitud (mV)")
        if self.emg_cols_selected:
            self.ax1.legend(loc='upper right', fontsize=8)
        self.ax1.grid(True)

        # ── Subplot 2: ForceDecks ─────────────────────────────────────
        color_map = {'Z Total': '#2ca02c', 'Z Left': '#1f77b4', 'Z Right': '#ff7f0e'}
        if self.fd_cols_selected:
            for fd_col in self.fd_cols_selected:
                if fd_col in self.data.columns:
                    color = color_map.get(fd_col, '#333333')
                    self.ax2.plot(time, self.data[fd_col], label=fd_col,
                                  linewidth=lw, color=color)
        else:
            self.ax2.text(
                0.5, 0.5, "Sin canales ForceDecks seleccionados",
                ha='center', va='center', transform=self.ax2.transAxes,
                color='gray', fontsize=10
            )

        titulo_fd = " + ".join(self.fd_cols_selected) if self.fd_cols_selected else "ForceDecks"
        self.ax2.set_title(f"ForceDecks — {titulo_fd}")
        self.ax2.set_xlabel("Tiempo (s)")
        self.ax2.set_ylabel("Fuerza (N)")
        if self.fd_cols_selected:
            self.ax2.legend(loc='upper right', fontsize=8)
        self.ax2.grid(True)

        # ── Fases seleccionadas ───────────────────────────────────────
        for xmin, xmax in self.selected_phases:
            self.ax1.axvspan(xmin, xmax, color='orange', alpha=0.3)
            self.ax2.axvspan(xmin, xmax, color='orange', alpha=0.3)

        # ── Línea roja de posición actual ────────────────────────────
        current_time = self.data['Time_common'].iloc[0] if len(self.data) > 0 else 0.0
        self.vertical_line_emg = self.ax1.axvline(current_time, color='red', linestyle='--', linewidth=1)
        self.vertical_line_fd  = self.ax2.axvline(current_time, color='red', linestyle='--', linewidth=1)

        # ── SpanSelector (si modo selección activo) ───────────────────
        if self.selection_mode:
            self.emg_selector = SpanSelector(self.ax1, self.on_select, 'horizontal', useblit=True)
            self.fd_selector  = SpanSelector(self.ax2, self.on_select, 'horizontal', useblit=True)

        self.figure.tight_layout()

        # Aplicar modo oscuro/claro si está activo
        if self._dark_mode:
            self.figure.patch.set_facecolor('#1e1e1e')
            for ax in [self.ax1, self.ax2]:
                ax.set_facecolor('#2d2d2d')
                ax.tick_params(colors='#cccccc')
                ax.xaxis.label.set_color('#cccccc')
                ax.yaxis.label.set_color('#cccccc')
                ax.title.set_color('#ffffff')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#555555')
        else:
            self.figure.patch.set_facecolor('white')
            for ax in [self.ax1, self.ax2]:
                ax.set_facecolor('white')

        # Guardar límites originales (solo la primera vez o al cargar nuevo CSV)
        if self.original_ax1_xlim is None:
            self.original_ax1_xlim = self.ax1.get_xlim()
            self.original_ax1_ylim = self.ax1.get_ylim()
            self.original_ax2_xlim = self.ax2.get_xlim()
            self.original_ax2_ylim = self.ax2.get_ylim()
        else:
            # Actualizar límites Y originales (pueden cambiar al añadir/quitar canales)
            # pero mantener los X originales intactos
            self.original_ax1_ylim = self.ax1.get_ylim()
            self.original_ax2_ylim = self.ax2.get_ylim()

        # Restaurar zoom activo si lo había
        if zoom_actual is not None:
            xlim, ylim1, ylim2 = zoom_actual
            self.ax1.set_xlim(xlim)
            self.ax2.set_xlim(xlim)
            self.ax1.set_ylim(ylim1)
            self.ax2.set_ylim(ylim2)

        self.canvas.draw()

    # ==================================================================
    # GESTIÓN DE FASES
    # ==================================================================

    def toggle_selection_mode(self, enabled):
        self.selection_mode = enabled
        self.plot_signals()

    def update_phase_action(self, action):
        self.phase_action = action

    def plot_signals(self):
        """Alias de initial_plot que activa SpanSelectors si procede."""
        if self.data is None:
            return
        self.initial_plot()
        if self.selection_mode:
            if not hasattr(self, 'emg_selector') or self.emg_selector is None:
                self.emg_selector = SpanSelector(self.ax1, self.on_select, 'horizontal', useblit=True)
            if not hasattr(self, 'fd_selector') or self.fd_selector is None:
                self.fd_selector  = SpanSelector(self.ax2, self.on_select, 'horizontal', useblit=True)

    def on_select(self, xmin, xmax):
        if not self.selection_mode or self.data is None:
            return
        if self.phase_action == "Añadir fase":
            self.selected_phases.append((xmin, xmax))
            self.label_status.setText(f"Fase añadida: {xmin:.2f}s — {xmax:.2f}s")
        elif self.phase_action == "Borrar fase":
            self.selected_phases = [
                p for p in self.selected_phases
                if not (xmin <= p[0] <= xmax and xmin <= p[1] <= xmax)
            ]
            self.label_status.setText(f"Fase eliminada: {xmin:.2f}s — {xmax:.2f}s")
        self.plot_signals()

    # ==================================================================
    # EXPORTACIÓN DE FASES
    # ==================================================================

    def export_selected_phases(self):
        """
        Exporta los datos de las fases seleccionadas a un CSV con
        estadísticas (min, max, media) por canal y los datos brutos.
        """
        if not self.selected_phases or self.data is None or self.data.empty:
            self.label_status.setText("No hay fases seleccionadas para exportar.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar Fases", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        with open(file_path, 'w') as f:
            for i, (xmin, xmax) in enumerate(self.selected_phases, start=1):
                phase_data = self.data[
                    (self.data['Time_common'] >= xmin) &
                    (self.data['Time_common'] <= xmax)
                ]
                if phase_data.empty:
                    continue

                analyze_cols = [c for c in phase_data.columns if c != 'Time_common']
                stats = phase_data[analyze_cols].describe().loc[['min', 'max', 'mean']]

                f.write(f"Fase {i}: {xmin:.2f}s — {xmax:.2f}s\n")
                f.write("Estadísticas:\n")
                f.write(stats.to_csv())
                f.write("\nDatos:\n")
                phase_data.to_csv(f, index=False)
                f.write("\n\n")

        self.label_status.setText(f"✅ Fases exportadas a: {file_path}")


# ==================================================================
# FLUJO PRINCIPAL
# ==================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = SignalViewer()
    viewer.show()
    sys.exit(app.exec_())
