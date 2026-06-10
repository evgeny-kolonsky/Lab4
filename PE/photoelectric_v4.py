#!/usr/bin/env python3
"""
Photoelectric Effect — I-V Curve Measurement System  v4
========================================================
Keithley 2400 SourceMeter  →  sources voltage on photocell + measures current

Changes vs v3:
  • English interface (back from Russian)
  • Dark theme
  • Y-axis can be toggled between Linear and Symlog (symmetric log with a
    linear region of ±1 nA), with full mouse-wheel zoom in both modes
  • Curve colour derived from wavelength; curves described by wavelength + comment
  • Connections / Raw command / Keithley parameters on a separate Settings tab
  • Sweep segments can be saved as defaults
  • "Delete selected curve" instead of "Clear all data"

Dependencies: pip install PyQt6 pyqtgraph pyserial numpy
"""

import sys
import time
import csv
import json
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QFileDialog, QSplitter, QProgressBar, QLineEdit,
    QGridLayout, QTextEdit, QTabWidget, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor
import pyqtgraph as pg


# ---------------------------------------------------------------------------
#  Wavelength → colour  (from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python)
# ---------------------------------------------------------------------------

def wavelength_to_color(wavelength, gamma=0.8) -> str:
    """Convert a wavelength in nm (380–750) to an approximate RGB hex colour."""
    wavelength = float(wavelength)
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if 380 <= wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif 510 <= wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif 645 <= wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    r = int(R * 255)
    g = int(G * 255)
    b = int(B * 255)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def brighten_for_dark(hex_color: str, min_lum: int = 90) -> str:
    """Lift very dark colours so they stay visible on a dark background."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    lum = max(r, g, b)
    if lum < min_lum and lum > 0:
        scale = min_lum / lum
        r, g, b = (min(255, int(c * scale)) for c in (r, g, b))
    elif lum == 0:
        r = g = b = min_lum
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


# ---------------------------------------------------------------------------
#  Symlog transform helpers  (linear region |y| <= linthresh, log beyond)
# ---------------------------------------------------------------------------

def symlog_fwd(y, linthresh=1e-9):
    y = np.asarray(y, dtype=float)
    a = np.abs(y)
    small = a <= linthresh
    out = np.empty_like(y)
    out[small] = y[small] / linthresh
    big = ~small
    out[big] = np.sign(y[big]) * (1.0 + np.log10(a[big] / linthresh))
    return out


def symlog_inv(t, linthresh=1e-9):
    t = np.asarray(t, dtype=float)
    a = np.abs(t)
    small = a <= 1.0
    out = np.empty_like(t)
    out[small] = t[small] * linthresh
    big = ~small
    out[big] = np.sign(t[big]) * linthresh * np.power(10.0, a[big] - 1.0)
    return out


def _fmt_tick_current(cur: float) -> str:
    a = abs(cur)
    if a < 1e-13:
        return "0"
    if a >= 1e-3:
        s = f"{cur * 1e3:g}m"
    elif a >= 1e-6:
        s = f"{cur * 1e6:g}\u00b5"
    elif a >= 1e-9:
        s = f"{cur * 1e9:g}n"
    else:
        s = f"{cur * 1e12:g}p"
    return s + "A"


class SymLogAxis(pg.AxisItem):
    """Left axis that can display symlog tick labels while the ViewBox stays
    in plain linear (transformed) coordinates — so mouse-wheel zoom keeps
    working in both modes."""

    def __init__(self, linthresh=1e-9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linthresh = linthresh
        self.log_mode = False

    def tickValues(self, minVal, maxVal, size):
        if not self.log_mode:
            return super().tickValues(minVal, maxVal, size)
        # decade currents in both polarities + zero
        decades = [10.0 ** e for e in range(-13, 0)]
        currents = [0.0]
        for d in decades:
            currents.append(d)
            currents.append(-d)
        majors = []
        for c in currents:
            t = float(symlog_fwd(c, self.linthresh))
            if minVal <= t <= maxVal:
                majors.append(t)
        majors = sorted(set(majors))
        return [(1.0, majors)]

    def tickStrings(self, values, scale, spacing):
        if not self.log_mode:
            return super().tickStrings(values, scale, spacing)
        out = []
        for v in values:
            cur = float(symlog_inv(v, self.linthresh))
            out.append(_fmt_tick_current(cur))
        return out


# ---------------------------------------------------------------------------
#  Utility
# ---------------------------------------------------------------------------

def format_current(amps: float) -> str:
    a = abs(amps)
    if a == 0:
        return "0 A"
    if a >= 1e-3:
        return f"{amps * 1e3:.4f} mA"
    if a >= 1e-6:
        return f"{amps * 1e6:.4f} \u00b5A"
    if a >= 1e-9:
        return f"{amps * 1e9:.3f} nA"
    return f"{amps * 1e12:.2f} pA"


def list_serial_ports() -> list:
    return [p.device for p in serial.tools.list_ports.comports()]


# ---------------------------------------------------------------------------
#  Instrument driver (with logging callback)
# ---------------------------------------------------------------------------

class Keithley2400:
    """Keithley 2400 SourceMeter via RS-232 / COM-USB adapter."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 3,
                 flow: str = "none", term_tx: str = "\n", log=None):
        self._log = log or (lambda *a: None)
        self._term_tx = term_tx
        xon = flow == "xonxoff"
        rts = flow == "rtscts"
        self._log(f"[K2400] Opening {port} @ {baudrate} baud, flow={flow} …")
        self.ser = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=8, parity="N", stopbits=1,
            timeout=timeout, xonxoff=xon, rtscts=rts
        )
        time.sleep(0.3)
        self._flush()
        self._log(f"[K2400] Port opened OK")

    def _flush(self):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def write(self, cmd: str):
        full = cmd + self._term_tx
        self.ser.write(full.encode())
        self._log(f"[K2400] TX: {cmd}")
        time.sleep(0.05)

    def _read_response(self, timeout: float = 3) -> str:
        """Read response handling any terminator: CR, LF, or CR+LF."""
        old_to = self.ser.timeout
        self.ser.timeout = timeout
        buf = b""
        deadline = time.time() + timeout
        while time.time() < deadline:
            ch = self.ser.read(1)
            if not ch:
                break  # timeout on single char
            if ch in (b"\n", b"\r"):
                if buf:  # got data before terminator → done
                    break
                continue  # leading CR/LF, skip
            buf += ch
        self.ser.timeout = old_to

        if not buf:
            time.sleep(0.2)
            n = self.ser.in_waiting
            if n:
                raw = self.ser.read(n)
                self._log(f"[K2400] RX (late {n}b): {raw!r}")
                buf = raw

        resp = buf.decode(errors="replace").strip()
        return resp

    def query(self, cmd: str, timeout: float = 3) -> str:
        self._flush()
        self.write(cmd)
        resp = self._read_response(timeout)
        self._log(f"[K2400] RX: '{resp}'")
        if not resp:
            self._log("[K2400] ⚠ Empty response (timeout?)")
        return resp

    def idn(self) -> str:
        return self.query("*IDN?", timeout=3)

    def reset(self):
        self.write("*RST")
        time.sleep(1.0)
        self.write("*CLS")
        time.sleep(0.2)

    def configure_source_measure(self, compliance: float = 1e-3, nplc: float = 1):
        if compliance < 1e-9:
            compliance = 1e-3
            self._log(f"[K2400] ⚠ Compliance too low, using 1 mA")
        self._log(f"[K2400] Configuring: source V, measure I, "
                  f"compl={compliance:.2e}, NPLC={nplc}")
        cmds = [
            ":SOUR:FUNC VOLT",
            ":SOUR:VOLT:RANG 20",
            ":SOUR:VOLT 0",
            ':SENS:FUNC "CURR"',
            f":SENS:CURR:PROT {compliance:.6e}",
            ":SENS:CURR:RANG:AUTO ON",
            f":SENS:CURR:NPLC {nplc}",
            ":FORM:ELEM VOLT,CURR",
        ]
        for c in cmds:
            self.write(c)
            time.sleep(0.1)
        time.sleep(0.3)

    def set_voltage(self, voltage: float):
        self.write(f":SOUR:VOLT {voltage:.6f}")

    def output_on(self):
        self.write(":OUTP ON")

    def output_off(self):
        self.write(":OUTP OFF")

    def read(self) -> tuple:
        """Trigger + read.  Returns (voltage_actual, current) in V, A."""
        resp = self.query(":READ?", timeout=10)
        if not resp:
            self._log("[K2400] Retrying :READ? after clearing errors …")
            self.write("*CLS")
            time.sleep(0.3)
            resp = self.query(":READ?", timeout=10)
        if not resp:
            raise TimeoutError("No response from Keithley on :READ?")
        parts = resp.split(",")
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
        if len(parts) == 1:
            return 0.0, float(parts[0])
        raise ValueError(f"Unexpected :READ? response: '{resp}'")

    def check_errors(self) -> str:
        time.sleep(0.2)
        return self.query(":SYST:ERR?", timeout=3)

    def close(self):
        try:
            self.output_off()
            self.write(":SYST:LOC")
        except Exception:
            pass
        self.ser.close()
        self._log("[K2400] Closed")


# ---------------------------------------------------------------------------
#  Data
# ---------------------------------------------------------------------------

@dataclass
class SweepSegment:
    start_v: float
    end_v: float
    step_v: float
    settle_ms: int = 500
    averages: int = 3


@dataclass
class DataPoint:
    voltage_set: float
    voltage_actual: float
    current_avg: float
    current_std: float
    readings: list
    sweep_id: int = 0
    wavelength: int = 0
    comment: str = ""
    led_voltage: float = 0.0


# ---------------------------------------------------------------------------
#  Sweep worker
# ---------------------------------------------------------------------------

class SweepWorker(QThread):
    point_acquired = pyqtSignal(object)
    progress_update = pyqtSignal(int, int)
    log_msg = pyqtSignal(str)
    sweep_done = pyqtSignal()

    def __init__(self, keithley: Keithley2400,
                 segments: List[SweepSegment],
                 sweep_id: int = 0,
                 wavelength: int = 0,
                 comment: str = "",
                 led_voltage: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.keithley = keithley
        self.segments = segments
        self.sweep_id = sweep_id
        self.wavelength = wavelength
        self.comment = comment
        self.led_voltage = led_voltage
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def _build_voltage_list(self):
        points = []
        for seg in self.segments:
            if seg.step_v <= 0:
                continue
            direction = 1 if seg.end_v >= seg.start_v else -1
            n = int(round(abs(seg.end_v - seg.start_v) / seg.step_v))
            for i in range(n + 1):
                v = round(seg.start_v + i * seg.step_v * direction, 6)
                points.append((v, seg.settle_ms, seg.averages))
        seen = {}
        for p in points:
            seen[p[0]] = p
        return [seen[k] for k in sorted(seen.keys())]

    def run(self):
        try:
            self._run_sweep()
        except Exception as e:
            self.log_msg.emit(f"❌ SWEEP EXCEPTION: {e}")
        finally:
            try:
                self.keithley.set_voltage(0)
            except Exception:
                pass
            self.sweep_done.emit()

    def _run_sweep(self):
        points = self._build_voltage_list()
        total = len(points)
        if total == 0:
            self.log_msg.emit("⚠ No voltage points generated from segments")
            return

        self.log_msg.emit(f"▶ Sweep: {total} points, λ = {self.wavelength} nm")

        for idx, (v_set, settle_ms, n_avg) in enumerate(points):
            if self._stop_flag:
                self.log_msg.emit("■ Sweep stopped by user")
                break

            self.keithley.set_voltage(v_set)
            self.progress_update.emit(idx + 1, total)

            self.log_msg.emit(
                f"  [{idx+1}/{total}] V = {v_set:+.4f} V, "
                f"settling {settle_ms} ms …"
            )
            time.sleep(settle_ms / 1000.0)

            readings = []
            v_actual = v_set
            for j in range(n_avg):
                if self._stop_flag:
                    break
                try:
                    v_a, curr = self.keithley.read()
                    v_actual = v_a
                    readings.append(curr)
                except Exception as e:
                    self.log_msg.emit(f"  ⚠ Read #{j+1} error: {e}")
                time.sleep(0.02)

            if readings:
                dp = DataPoint(
                    voltage_set=v_set,
                    voltage_actual=v_actual,
                    current_avg=float(np.mean(readings)),
                    current_std=float(np.std(readings)) if len(readings) > 1 else 0.0,
                    readings=readings,
                    sweep_id=self.sweep_id,
                    wavelength=self.wavelength,
                    comment=self.comment,
                    led_voltage=self.led_voltage,
                )
                self.point_acquired.emit(dp)
                self.log_msg.emit(
                    f"  → I = {format_current(dp.current_avg)}  "
                    f"(σ = {format_current(dp.current_std)})"
                )
            else:
                self.log_msg.emit(f"  ⚠ No valid readings at {v_set:+.4f} V")

        self.log_msg.emit(f"✓ Sweep finished — {len(points)} points requested")


# ---------------------------------------------------------------------------
#  Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    LINTHRESH = 1e-9  # 1 nA linear threshold for symlog

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photoelectric Effect — I-V Measurement  v4")
        self.resize(1500, 900)

        self.keithley: Optional[Keithley2400] = None
        self.worker: Optional[SweepWorker] = None
        self.data: List[DataPoint] = []
        self.settings = QSettings("Technion_Physics", "Photoelectric")
        self._k_term_tx = "\n"
        self.log_mode = False  # False = linear, True = symlog

        # each entry: dict(id, wl, comment, curve, ebars)
        self.sweep_curves = []
        self._next_sweep_id = 1

        self._build_ui()
        self._restore_settings()
        self._refresh_ports()

    # -- logging --

    def log(self, msg: str):
        print(msg, flush=True)
        self.log_text.append(msg)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        self.statusBar().showMessage(msg.replace("\n", " "))
        QApplication.processEvents()

    # ---------------------------------------------------------------
    #  UI
    # ---------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)

        # ═══ LEFT PANEL ═══
        left = QWidget()
        left.setMaximumWidth(520)
        left.setMinimumWidth(440)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        lv.setSpacing(10)

        # --- Curve parameters (wavelength + comment) ---
        curve_grp = QGroupBox("Curve parameters")
        cgl = QGridLayout(curve_grp)
        cgl.setVerticalSpacing(10)

        cgl.addWidget(QLabel("Wavelength:"), 0, 0)
        self.sb_wavelength = QSpinBox()
        self.sb_wavelength.setRange(250, 999)
        self.sb_wavelength.setValue(550)
        self.sb_wavelength.setSuffix(" nm")
        self.sb_wavelength.valueChanged.connect(self._update_color_preview)
        cgl.addWidget(self.sb_wavelength, 0, 1)

        self.lbl_color = QLabel()
        self.lbl_color.setFixedSize(18, 18)
        self.lbl_color.setToolTip("Curve colour (from wavelength)")
        cgl.addWidget(self.lbl_color, 0, 2, Qt.AlignmentFlag.AlignLeft)

        cgl.addWidget(QLabel("LED_V (intensity):"), 1, 0)
        self.sb_led_v = QDoubleSpinBox()
        self.sb_led_v.setRange(0, 30)
        self.sb_led_v.setValue(12.0)
        self.sb_led_v.setDecimals(1)
        self.sb_led_v.setSuffix(" V")
        self.sb_led_v.setToolTip("Light-intensity label stored with the data "
                                 "(LED_V column, v2-compatible)")
        cgl.addWidget(self.sb_led_v, 1, 1)

        cgl.addWidget(QLabel("Comment:"), 2, 0)
        self.le_comment = QLineEdit()
        self.le_comment.setPlaceholderText("e.g. green filter, 5 mW")
        cgl.addWidget(self.le_comment, 2, 1, 1, 2)

        lv.addWidget(curve_grp)

        # --- Sweep segments ---
        seg_grp = QGroupBox("Sweep segments (Keithley → photocell)")
        sv = QVBoxLayout(seg_grp)

        self.tbl_seg = QTableWidget(0, 5)
        self.tbl_seg.setHorizontalHeaderLabels(
            ["Start V", "End V", "Step V", "Settle ms", "Avg N"]
        )
        hdr = self.tbl_seg.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_seg.verticalHeader().setDefaultSectionSize(34)
        self.tbl_seg.setMinimumHeight(180)
        self.tbl_seg.cellChanged.connect(self._update_pts_label)
        sv.addWidget(self.tbl_seg)

        sb = QHBoxLayout()
        btn_a = QPushButton("+ Add")
        btn_a.clicked.connect(lambda: self._add_seg_row())
        btn_d = QPushButton("− Del")
        btn_d.clicked.connect(self._del_seg_row)
        btn_def = QPushButton("Reset")
        btn_def.setToolTip("Load saved default segments")
        btn_def.clicked.connect(self._default_segments)
        sb.addWidget(btn_a)
        sb.addWidget(btn_d)
        sb.addWidget(btn_def)
        sv.addLayout(sb)

        btn_savedef = QPushButton("Save as defaults")
        btn_savedef.setToolTip("Remember current segments as the defaults")
        btn_savedef.clicked.connect(self._save_segments_as_default)
        sv.addWidget(btn_savedef)

        self.lbl_pts = QLabel("≈ 0 points")
        sv.addWidget(self.lbl_pts)

        lv.addWidget(seg_grp)

        # --- Controls ---
        ctrl_grp = QGroupBox("Sweep control")
        cv = QVBoxLayout(ctrl_grp)
        cv.setSpacing(10)

        r1 = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start sweep")
        self.btn_start.setStyleSheet("font-weight:bold; padding:12px;")
        self.btn_start.clicked.connect(self._start_sweep)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_sweep)
        r1.addWidget(self.btn_start)
        r1.addWidget(self.btn_stop)
        cv.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Manual V:"))
        self.sb_man_v = QDoubleSpinBox()
        self.sb_man_v.setRange(-21, 21)
        self.sb_man_v.setDecimals(3)
        self.sb_man_v.setSuffix(" V")
        r2.addWidget(self.sb_man_v)
        self.btn_man = QPushButton("Set + Read")
        self.btn_man.clicked.connect(self._manual_measure)
        r2.addWidget(self.btn_man)
        cv.addLayout(r2)
        self.lbl_man = QLabel("")
        self.lbl_man.setWordWrap(True)
        cv.addWidget(self.lbl_man)

        lv.addWidget(ctrl_grp)

        # --- Curves list + management ---
        curves_grp = QGroupBox("Curves")
        cvg = QVBoxLayout(curves_grp)
        self.curve_list = QListWidget()
        self.curve_list.setMinimumHeight(110)
        cvg.addWidget(self.curve_list)

        cbtns = QHBoxLayout()
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_del_curve = QPushButton("🗑 Delete selected curve")
        self.btn_del_curve.clicked.connect(self._delete_selected_curve)
        cbtns.addWidget(self.btn_export)
        cbtns.addWidget(self.btn_del_curve)
        cvg.addLayout(cbtns)

        lv.addWidget(curves_grp)
        lv.addStretch()

        # ═══ RIGHT PANEL ═══
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(2, 2, 2, 2)

        # plot toolbar (Y-axis scale toggle)
        ptb = QHBoxLayout()
        ptb.addWidget(QLabel("Y axis:"))
        self.cb_yscale = QComboBox()
        self.cb_yscale.addItems(["Linear", "Symlog (lin ±1 nA)"])
        self.cb_yscale.currentIndexChanged.connect(self._on_yscale_changed)
        ptb.addWidget(self.cb_yscale)
        self.btn_autorange = QPushButton("Auto-range")
        self.btn_autorange.setToolTip("Fit all data (zoom stays on mouse wheel)")
        self.btn_autorange.clicked.connect(lambda: self.plot_w.autoRange())
        ptb.addWidget(self.btn_autorange)
        ptb.addStretch()
        rv.addLayout(ptb)

        # plot
        pg.setConfigOptions(antialias=True)
        self.y_axis = SymLogAxis(linthresh=self.LINTHRESH, orientation="left")
        self.plot_w = pg.PlotWidget(axisItems={"left": self.y_axis})
        self.plot_w.setBackground("#1b1b1f")
        label_style = {"font-size": "14pt", "color": "#dddddd"}
        self.plot_w.setLabel("bottom", "Voltage", units="V", **label_style)
        self.plot_w.setLabel("left", "Current", units="A", **label_style)
        tick_font = QFont("Segoe UI", 11)
        self.plot_w.getAxis("bottom").setStyle(tickFont=tick_font)
        self.plot_w.getAxis("left").setStyle(tickFont=tick_font)
        self.plot_w.getAxis("bottom").setTextPen("#dddddd")
        self.plot_w.getAxis("left").setTextPen("#dddddd")
        self.plot_w.setTitle("I-V Characteristic", size="15pt", color="#dddddd")
        self.plot_w.showGrid(x=True, y=True, alpha=0.25)
        self.plot_w.addLegend(offset=(60, 10), labelTextSize="12pt",
                              labelTextColor="#dddddd")
        # mouse-wheel zoom is on by default for the ViewBox; keep it enabled
        self.plot_w.getViewBox().setMouseEnabled(x=True, y=True)
        rv.addWidget(self.plot_w, stretch=3)

        # tabs: data + log + settings
        tabs = QTabWidget()

        self.tbl_data = QTableWidget(0, 5)
        self.tbl_data.setHorizontalHeaderLabels(
            ["λ (nm)", "V_set (V)", "I_avg", "I_avg (A)", "σ_I (A)"]
        )
        self.tbl_data.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_data.verticalHeader().setDefaultSectionSize(30)
        self.tbl_data.setAlternatingRowColors(True)
        tabs.addTab(self.tbl_data, "Data")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 11))
        tabs.addTab(self.log_text, "Log")

        tabs.addTab(self._build_settings_tab(), "Settings ⚙")

        rv.addWidget(tabs, stretch=2)

        self.statusBar().showMessage("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.statusBar().addPermanentWidget(self.progress)

        main_h.addWidget(left, stretch=0)
        main_h.addWidget(right, stretch=1)

        self._default_segments()
        self._update_color_preview()

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(12)

        # --- Connections ---
        conn_grp = QGroupBox("Connections")
        cg = QGridLayout(conn_grp)
        cg.setVerticalSpacing(10)

        cg.addWidget(QLabel("Keithley 2400:"), 0, 0)
        self.cb_k_port = QComboBox()
        self.cb_k_port.setEditable(True)
        self.cb_k_port.setMinimumWidth(140)
        cg.addWidget(self.cb_k_port, 0, 1)

        cg.addWidget(QLabel("Baud:"), 0, 2)
        self.cb_k_baud = QComboBox()
        self.cb_k_baud.addItems(["9600", "19200", "38400", "57600", "115200"])
        self.cb_k_baud.setCurrentText("9600")
        cg.addWidget(self.cb_k_baud, 0, 3)

        self.btn_k_conn = QPushButton("Connect")
        self.btn_k_conn.clicked.connect(self._toggle_keithley)
        cg.addWidget(self.btn_k_conn, 0, 4)

        cg.addWidget(QLabel("Flow ctrl:"), 1, 0)
        self.cb_k_flow = QComboBox()
        self.cb_k_flow.addItems(["none", "xonxoff", "rtscts"])
        self.cb_k_flow.setCurrentText("none")
        cg.addWidget(self.cb_k_flow, 1, 1)

        self.btn_k_scan = QPushButton("🔍 Auto-scan")
        self.btn_k_scan.setToolTip(
            "Try all baud/flow combinations to find working settings"
        )
        self.btn_k_scan.clicked.connect(self._auto_scan_keithley)
        cg.addWidget(self.btn_k_scan, 1, 4)

        self.btn_refresh = QPushButton("↻ Ports")
        self.btn_refresh.clicked.connect(self._refresh_ports)
        cg.addWidget(self.btn_refresh, 2, 4)

        self.lbl_k_idn = QLabel("Keithley: not connected")
        self.lbl_k_idn.setWordWrap(True)
        cg.addWidget(self.lbl_k_idn, 3, 0, 1, 5)

        lay.addWidget(conn_grp)

        # --- Raw command terminal ---
        raw_grp = QGroupBox("Raw SCPI command")
        rgl = QHBoxLayout(raw_grp)
        self.le_raw_cmd = QComboBox()
        self.le_raw_cmd.setEditable(True)
        self.le_raw_cmd.addItems(["*IDN?", "*RST", ":SYST:ERR?", ":OUTP?",
                                  ":SOUR:VOLT?", ":MEAS:CURR?", ":READ?"])
        self.le_raw_cmd.setMinimumWidth(220)
        rgl.addWidget(QLabel("Command:"))
        rgl.addWidget(self.le_raw_cmd, stretch=1)
        self.btn_raw_send = QPushButton("Send")
        self.btn_raw_send.clicked.connect(self._send_raw_cmd)
        rgl.addWidget(self.btn_raw_send)
        lay.addWidget(raw_grp)

        # --- Measurement parameters ---
        meas_grp = QGroupBox("Keithley measurement parameters")
        mg = QGridLayout(meas_grp)
        mg.setVerticalSpacing(10)

        mg.addWidget(QLabel("NPLC:"), 0, 0)
        self.sb_nplc = QDoubleSpinBox()
        self.sb_nplc.setRange(0.01, 10)
        self.sb_nplc.setValue(1.0)
        self.sb_nplc.setDecimals(2)
        self.sb_nplc.setToolTip(
            "Integration time in power-line cycles\n"
            "50 Hz → 1 NPLC = 20 ms, 10 NPLC = 200 ms"
        )
        mg.addWidget(self.sb_nplc, 0, 1)

        mg.addWidget(QLabel("Compliance:"), 1, 0)
        self.sb_compl = QDoubleSpinBox()
        self.sb_compl.setDecimals(6)
        self.sb_compl.setRange(0.000001, 1.0)
        self.sb_compl.setSingleStep(0.001)
        self.sb_compl.setValue(0.001)
        self.sb_compl.setSuffix(" A")
        mg.addWidget(self.sb_compl, 1, 1)

        lay.addWidget(meas_grp)
        lay.addStretch()
        return w

    # ---------------------------------------------------------------
    #  Y-axis scale (linear / symlog)
    # ---------------------------------------------------------------

    def _on_yscale_changed(self, idx):
        self.log_mode = (idx == 1)
        self.y_axis.log_mode = self.log_mode
        if self.log_mode:
            self.plot_w.getAxis("left").enableAutoSIPrefix(False)
            self.plot_w.setLabel("left", "Current (symlog)",
                                 color="#dddddd", **{"font-size": "14pt"})
        else:
            self.plot_w.getAxis("left").enableAutoSIPrefix(True)
            self.plot_w.setLabel("left", "Current", units="A",
                                 color="#dddddd", **{"font-size": "14pt"})
        # re-plot every curve in the new coordinate system
        for entry in self.sweep_curves:
            self._replot_curve(entry)
        self.plot_w.autoRange()

    def _y_transform(self, y):
        """Map a current value/array into plot coordinates for the current mode."""
        if self.log_mode:
            return symlog_fwd(np.asarray(y, dtype=float), self.LINTHRESH)
        return np.asarray(y, dtype=float)

    def _replot_curve(self, entry):
        pts = [d for d in self.data if d.sweep_id == entry["id"]]
        if not pts:
            entry["curve"].setData([], [])
            entry["ebars"].setData(x=np.array([]), y=np.array([]))
            return
        vv = np.array([p.voltage_set for p in pts])
        ii = np.array([p.current_avg for p in pts])
        ss = np.array([p.current_std for p in pts])

        y = self._y_transform(ii)
        entry["curve"].setData(vv, y)

        # error bars: asymmetric in symlog, symmetric in linear
        if self.log_mode:
            top = self._y_transform(ii + ss) - y
            bottom = y - self._y_transform(ii - ss)
            top = np.clip(top, 0, None)
            bottom = np.clip(bottom, 0, None)
            entry["ebars"].setData(x=vv, y=y, top=top, bottom=bottom)
        else:
            entry["ebars"].setData(x=vv, y=y, height=2 * ss)

    # ---------------------------------------------------------------
    #  Colour preview
    # ---------------------------------------------------------------

    def _update_color_preview(self):
        color = brighten_for_dark(wavelength_to_color(self.sb_wavelength.value()))
        self.lbl_color.setStyleSheet(
            f"background-color:{color}; border:1px solid #888; border-radius:3px;"
        )

    # ---------------------------------------------------------------
    #  Settings persistence
    # ---------------------------------------------------------------

    def _restore_settings(self):
        pk = self.settings.value("k_port", "")
        if pk:
            self.cb_k_port.setCurrentText(pk)

    def _save_settings(self):
        self.settings.setValue("k_port", self.cb_k_port.currentText())

    def closeEvent(self, event):
        self._save_settings()
        self._disconnect_all()
        event.accept()

    # ---------------------------------------------------------------
    #  Ports
    # ---------------------------------------------------------------

    def _refresh_ports(self):
        ports = list_serial_ports()
        prev = self.cb_k_port.currentText()
        self.cb_k_port.clear()
        self.cb_k_port.addItems(ports)
        if prev in ports:
            self.cb_k_port.setCurrentText(prev)
        self.log(f"Serial ports: {ports or '(none found)'}")

    # ---------------------------------------------------------------
    #  Keithley connection
    # ---------------------------------------------------------------

    def _toggle_keithley(self):
        if self.keithley:
            self.keithley.close()
            self.keithley = None
            self.btn_k_conn.setText("Connect")
            self.lbl_k_idn.setText("Keithley: not connected")
            self.log("[K2400] Disconnected")
            return

        port = self.cb_k_port.currentText().strip()
        baud = int(self.cb_k_baud.currentText())
        flow = self.cb_k_flow.currentText()
        if not port:
            QMessageBox.warning(self, "Error", "Select Keithley COM port")
            return

        try:
            self.keithley = Keithley2400(
                port, baud, flow=flow, term_tx=self._k_term_tx,
                log=self.log
            )
            idn = self.keithley.idn()
            if idn:
                self.lbl_k_idn.setText(f"Keithley: {idn}")
                self.btn_k_conn.setText("Disconnect")
                self.log(f"[K2400] Connected ✓  IDN: {idn}")
            else:
                self.log("[K2400] ⚠ Port opened but IDN returned empty.")
                self.log("[K2400]   Try: Auto-scan button, or check RS-232 on instrument")
                self.lbl_k_idn.setText("Keithley: connected but no IDN response")
                self.btn_k_conn.setText("Disconnect")
        except Exception as e:
            self.keithley = None
            self.log(f"[K2400] ✗ Connection failed: {e}")
            QMessageBox.critical(self, "Keithley", f"Connection failed:\n{e}")

    def _disconnect_all(self):
        if self.keithley:
            try:
                self.keithley.close()
            except Exception:
                pass
        self.keithley = None

    # ---------------------------------------------------------------
    #  Auto-scan & raw terminal
    # ---------------------------------------------------------------

    def _auto_scan_keithley(self):
        port = self.cb_k_port.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Error", "Select Keithley COM port first")
            return

        if self.keithley:
            self.keithley.close()
            self.keithley = None
            self.btn_k_conn.setText("Connect")

        bauds = [9600, 19200, 38400, 57600, 115200]
        flows = ["none", "xonxoff", "rtscts"]

        self.log("=" * 50)
        self.log(f"[Scan] Auto-scanning {port} …")
        self.log(f"[Scan] Trying {len(bauds)} baud × {len(flows)} flow = "
                 f"{len(bauds)*len(flows)} combinations")

        for baud in bauds:
            for flow in flows:
                self.log(f"[Scan]  Trying baud={baud}, flow={flow} …")
                QApplication.processEvents()
                try:
                    xon = flow == "xonxoff"
                    rts = flow == "rtscts"
                    ser = serial.Serial(
                        port=port, baudrate=baud,
                        bytesize=8, parity="N", stopbits=1,
                        timeout=1.5, xonxoff=xon, rtscts=rts
                    )
                    time.sleep(0.2)
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()

                    for term_name, term in [("LF", "\n"), ("CR+LF", "\r\n"), ("CR", "\r")]:
                        ser.reset_input_buffer()
                        ser.write(f"*IDN?{term}".encode())

                        buf = b""
                        deadline = time.time() + 2.0
                        while time.time() < deadline:
                            ch = ser.read(1)
                            if not ch:
                                break
                            if ch in (b"\n", b"\r"):
                                if buf:
                                    break
                                continue
                            buf += ch

                        if not buf:
                            time.sleep(0.3)
                            n = ser.in_waiting
                            if n:
                                buf = ser.read(n)
                                self.log(f"[Scan]    raw ({n}b): {buf!r}")

                        resp = buf.decode(errors="replace").strip()

                        if resp and ("KEITHLEY" in resp.upper() or "2400" in resp
                                     or len(resp) > 5):
                            self.log(f"[Scan] ✓ FOUND! baud={baud}, flow={flow}, "
                                     f"term={term_name}")
                            self.log(f"[Scan]   IDN: {resp}")
                            ser.close()

                            self.cb_k_baud.setCurrentText(str(baud))
                            self.cb_k_flow.setCurrentText(flow)
                            self._k_term_tx = term
                            self.keithley = Keithley2400(
                                port, baud, flow=flow, term_tx=term,
                                log=self.log
                            )
                            self.lbl_k_idn.setText(f"Keithley: {resp}")
                            self.btn_k_conn.setText("Disconnect")
                            QMessageBox.information(
                                self, "Found!",
                                f"Keithley responds at:\n"
                                f"Baud: {baud}\nFlow: {flow}\n"
                                f"Terminator: {term_name}\n\n"
                                f"IDN: {resp}"
                            )
                            return

                    ser.close()
                except serial.SerialException as e:
                    self.log(f"[Scan]    port error: {e}")
                except Exception as e:
                    self.log(f"[Scan]    error: {e}")

        self.log("[Scan] ✗ No response at any setting.")
        QMessageBox.warning(
            self, "Not found",
            "No response from Keithley at any baud/flow setting.\n\n"
            "Check:\n"
            "• Is this the correct COM port?\n"
            "• Is the cable RS-232 (not GPIB)?\n"
            "• On the instrument: Menu → Communication → RS-232"
        )

    def _send_raw_cmd(self):
        if not self.keithley:
            self.log("[Raw] Keithley not connected")
            return
        cmd = self.le_raw_cmd.currentText().strip()
        if not cmd:
            return
        self.log(f"[Raw] Sending: {cmd}")
        try:
            if cmd.endswith("?"):
                resp = self.keithley.query(cmd, timeout=3)
                self.log(f"[Raw] Response: '{resp}'")
            else:
                self.keithley.write(cmd)
                self.log(f"[Raw] Sent (no response expected)")
        except Exception as e:
            self.log(f"[Raw] Error: {e}")

    # ---------------------------------------------------------------
    #  Segments table
    # ---------------------------------------------------------------

    def _add_seg_row(self, start=0, end=1, step=0.1, settle=500, avg=3):
        self.tbl_seg.blockSignals(True)
        r = self.tbl_seg.rowCount()
        self.tbl_seg.insertRow(r)
        for col, val in enumerate([start, end, step, settle, avg]):
            it = QTableWidgetItem(str(val))
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.tbl_seg.setItem(r, col, it)
        self.tbl_seg.blockSignals(False)
        self._update_pts_label()

    def _del_seg_row(self):
        r = self.tbl_seg.currentRow()
        if r >= 0:
            self.tbl_seg.removeRow(r)
            self._update_pts_label()

    _FALLBACK_SEGMENTS = [
        (-5, -1, 0.5, 500, 3),
        (-1, 1, 0.1, 800, 5),
        (1, 15, 0.5, 500, 3),
    ]

    def _default_segments(self):
        raw = self.settings.value("seg_defaults", "")
        rows = None
        if raw:
            try:
                rows = json.loads(raw)
            except Exception:
                rows = None
        if not rows:
            rows = [list(r) for r in self._FALLBACK_SEGMENTS]

        self.tbl_seg.setRowCount(0)
        for r in rows:
            self._add_seg_row(*r)

    def _save_segments_as_default(self):
        segs = self._read_segments()
        if not segs:
            QMessageBox.warning(self, "Save", "No valid segments to save")
            return
        rows = [[s.start_v, s.end_v, s.step_v, s.settle_ms, s.averages]
                for s in segs]
        self.settings.setValue("seg_defaults", json.dumps(rows))
        self.log(f"[Settings] Saved {len(rows)} segments as defaults")
        QMessageBox.information(
            self, "Saved",
            "Current segments saved as defaults.\n"
            "The 'Reset' button will now load them."
        )

    def _read_segments(self) -> List[SweepSegment]:
        segs = []
        for r in range(self.tbl_seg.rowCount()):
            try:
                segs.append(SweepSegment(
                    start_v=float(self.tbl_seg.item(r, 0).text()),
                    end_v=float(self.tbl_seg.item(r, 1).text()),
                    step_v=abs(float(self.tbl_seg.item(r, 2).text())),
                    settle_ms=int(float(self.tbl_seg.item(r, 3).text())),
                    averages=max(1, int(float(self.tbl_seg.item(r, 4).text()))),
                ))
            except (ValueError, AttributeError):
                pass
        return segs

    def _update_pts_label(self):
        segs = self._read_segments()
        total = 0
        for s in segs:
            if s.step_v > 0:
                total += int(round(abs(s.end_v - s.start_v) / s.step_v)) + 1
        self.lbl_pts.setText(f"≈ {total} points")

    # ---------------------------------------------------------------
    #  Manual measure
    # ---------------------------------------------------------------

    def _manual_measure(self):
        if not self.keithley:
            QMessageBox.warning(self, "Error", "Keithley not connected")
            return
        v = self.sb_man_v.value()
        try:
            self.keithley.reset()
            self.keithley.configure_source_measure(
                compliance=self.sb_compl.value(),
                nplc=self.sb_nplc.value()
            )
            self.keithley.set_voltage(v)
            self.keithley.output_on()
            time.sleep(0.5)
            v_act, curr = self.keithley.read()
            self.keithley.output_off()
            self.lbl_man.setText(
                f"V_set={v:+.3f}  V_act={v_act:+.6f}  I={format_current(curr)}"
            )
            self.log(f"[Manual] V={v:+.3f} → I={format_current(curr)}")
        except Exception as e:
            self.lbl_man.setText(f"Error: {e}")
            self.log(f"[Manual] Error: {e}")

    # ---------------------------------------------------------------
    #  Sweep
    # ---------------------------------------------------------------

    def _start_sweep(self):
        if not self.keithley:
            QMessageBox.warning(self, "Error", "Keithley not connected")
            return
        segs = self._read_segments()
        if not segs:
            QMessageBox.warning(self, "Error", "No sweep segments defined")
            return

        wl = self.sb_wavelength.value()
        comment = self.le_comment.text().strip()
        led_v = self.sb_led_v.value()
        self.log("=" * 50)
        self.log(f"Starting sweep, λ = {wl} nm, LED_V = {led_v:.1f}  «{comment}»")

        try:
            self.log("[K2400] Resetting …")
            self.keithley.reset()
            self.log("[K2400] Clearing errors …")
            self.keithley.write("*CLS")
            time.sleep(0.3)

            self.log("[K2400] Configuring source-measure …")
            self.keithley.configure_source_measure(
                compliance=self.sb_compl.value(),
                nplc=self.sb_nplc.value()
            )

            err = self.keithley.check_errors()
            if err and "No error" not in err and not err.startswith("0,") and not err.startswith("+0"):
                self.log(f"[K2400] ⚠ Error after config: {err}")
                if "-" in err:
                    QMessageBox.warning(
                        self, "Keithley error",
                        f"Error after configuration:\n{err}\n\n"
                        "Sweep will try to continue."
                    )

            self.log("[K2400] Output ON")
            self.keithley.output_on()
            time.sleep(0.5)

            self.log("[K2400] Test read at 0V …")
            v_act, curr = self.keithley.read()
            self.log(f"[K2400] Test OK: V={v_act:.4f}, I={format_current(curr)}")

        except Exception as e:
            self.log(f"[K2400] ✗ Setup failed: {e}")
            try:
                self.keithley.output_off()
            except Exception:
                pass
            QMessageBox.critical(self, "Error", f"Keithley setup failed:\n{e}")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        sweep_id = self._next_sweep_id
        self._next_sweep_id += 1
        color = brighten_for_dark(wavelength_to_color(wl))
        name = f"{wl} nm" + (f" — {comment}" if comment else "")
        curve = self.plot_w.plot(
            [], [],
            pen=pg.mkPen(color, width=2),
            symbol="o", symbolSize=6, symbolBrush=color,
            name=name
        )
        err_bars = pg.ErrorBarItem(pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
        self.plot_w.addItem(err_bars)

        entry = dict(id=sweep_id, wl=wl, comment=comment,
                     curve=curve, ebars=err_bars)
        self.sweep_curves.append(entry)

        item = QListWidgetItem(name)
        item.setForeground(QColor(color))
        item.setData(Qt.ItemDataRole.UserRole, sweep_id)
        self.curve_list.addItem(item)
        self.curve_list.setCurrentItem(item)

        self.worker = SweepWorker(self.keithley, segs, sweep_id, wl, comment, led_v)
        self.worker.point_acquired.connect(self._on_point)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.log_msg.connect(self.log)
        self.worker.sweep_done.connect(self._on_sweep_done)
        self.worker.start()

    def _stop_sweep(self):
        if self.worker:
            self.worker.stop()
            self.log("Stopping sweep …")

    def _on_point(self, dp: DataPoint):
        self.data.append(dp)

        r = self.tbl_data.rowCount()
        self.tbl_data.insertRow(r)
        vals = [
            f"{dp.wavelength}",
            f"{dp.voltage_set:+.4f}",
            format_current(dp.current_avg),
            f"{dp.current_avg:.6e}",
            f"{dp.current_std:.3e}",
        ]
        for col, v in enumerate(vals):
            self.tbl_data.setItem(r, col, QTableWidgetItem(v))
        self.tbl_data.scrollToBottom()

        entry = next((e for e in self.sweep_curves if e["id"] == dp.sweep_id), None)
        if entry:
            self._replot_curve(entry)

    def _on_progress(self, cur, total):
        self.progress.setMaximum(total)
        self.progress.setValue(cur)

    def _on_sweep_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        try:
            if self.keithley:
                self.keithley.output_off()
        except Exception:
            pass
        if self.sweep_curves:
            last_id = self.sweep_curves[-1]["id"]
            n = sum(1 for d in self.data if d.sweep_id == last_id)
            self.log(f"Sweep complete — {n} points collected")

    # ---------------------------------------------------------------
    #  Export / curve management
    # ---------------------------------------------------------------

    def _export_csv(self):
        if not self.data:
            QMessageBox.information(self, "Export", "No data to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV",
            f"photoelectric_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            # original v2 columns first (for compatibility),
            # then the two extra columns last
            w.writerow([
                "LED_V", "V_set", "V_actual", "I_avg_A",
                "I_std_A", "N", "Readings_A",
                "Wavelength_nm", "Comment"
            ])
            for dp in self.data:
                w.writerow([
                    f"{dp.led_voltage:.1f}",
                    f"{dp.voltage_set:.6f}",
                    f"{dp.voltage_actual:.6f}",
                    f"{dp.current_avg:.12e}",
                    f"{dp.current_std:.12e}",
                    len(dp.readings),
                    ";".join(f"{r:.12e}" for r in dp.readings),
                    dp.wavelength,
                    dp.comment,
                ])
        self.log(f"Saved: {path}")

    def _delete_selected_curve(self):
        row = self.curve_list.currentRow()
        if row < 0:
            QMessageBox.information(self, "Delete curve",
                                    "Select a curve in the list")
            return
        item = self.curve_list.item(row)
        sweep_id = item.data(Qt.ItemDataRole.UserRole)
        entry = next((e for e in self.sweep_curves if e["id"] == sweep_id), None)

        r = QMessageBox.question(
            self, "Delete curve",
            f"Delete curve «{item.text()}» and all its points?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if r != QMessageBox.StandardButton.Yes:
            return

        if entry:
            try:
                self.plot_w.removeItem(entry["curve"])
                self.plot_w.removeItem(entry["ebars"])
            except Exception:
                pass
            self.sweep_curves.remove(entry)

        self.data = [d for d in self.data if d.sweep_id != sweep_id]
        self.curve_list.takeItem(row)
        self._rebuild_data_table()
        self.log(f"Curve {sweep_id} deleted")

    def _rebuild_data_table(self):
        self.tbl_data.setRowCount(0)
        for dp in self.data:
            r = self.tbl_data.rowCount()
            self.tbl_data.insertRow(r)
            vals = [
                f"{dp.wavelength}",
                f"{dp.voltage_set:+.4f}",
                format_current(dp.current_avg),
                f"{dp.current_avg:.6e}",
                f"{dp.current_std:.3e}",
            ]
            for col, v in enumerate(vals):
                self.tbl_data.setItem(r, col, QTableWidgetItem(v))


# ---------------------------------------------------------------------------
#  Dark theme
# ---------------------------------------------------------------------------

def apply_dark_theme(app: QApplication):
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Base, QColor(30, 30, 32))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 54))
    pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Button, QColor(60, 60, 64))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(225, 225, 225))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(45, 45, 48))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(220, 220, 220))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(38, 110, 180))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(150, 150, 150))
    pal.setColor(QPalette.ColorRole.Link, QColor(90, 160, 240))
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.Text, QColor(120, 120, 120))
    pal.setColor(QPalette.ColorGroup.Disabled,
                 QPalette.ColorRole.ButtonText, QColor(120, 120, 120))
    app.setPalette(pal)


# ---------------------------------------------------------------------------
#  Entry
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_dark_theme(app)

    app.setFont(QFont("Segoe UI", 11))
    app.setStyleSheet("""
        QWidget { font-size: 14px; }
        QGroupBox {
            font-weight: bold;
            font-size: 15px;
            margin-top: 12px;
            padding-top: 12px;
            border: 1px solid #555;
            border-radius: 6px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            padding: 8px 12px;
            background-color: #3c3c40;
            border: 1px solid #555;
            border-radius: 4px;
        }
        QPushButton:hover { background-color: #474750; }
        QPushButton:pressed { background-color: #2a6eb4; }
        QPushButton:disabled { color: #777; }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            padding: 5px; min-height: 24px;
            background-color: #2b2b2e; border: 1px solid #555; border-radius: 4px;
        }
        QTabBar::tab {
            padding: 9px 16px; font-size: 14px;
            background: #3a3a3e; border: 1px solid #555;
        }
        QTabBar::tab:selected { background: #2a6eb4; }
        QTableWidget { font-size: 13px; gridline-color: #555; }
        QHeaderView::section {
            padding: 6px; font-weight: bold;
            background-color: #3a3a3e; border: 1px solid #555;
        }
        QListWidget { font-size: 14px; background-color: #2b2b2e; }
    """)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
