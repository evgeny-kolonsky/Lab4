#!/usr/bin/env python3
"""
Photoelectric Effect — I-V Curve Measurement System  v2
========================================================
Keithley 2400 SourceMeter  →  sources voltage on photocell + measures current
TENMA 72-2715 (optional)   →  controls LED light-source intensity

Dependencies: pip install PyQt6 pyqtgraph pyserial numpy
"""

import sys
import time
import csv
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import serial
import serial.tools.list_ports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QMessageBox, QFileDialog, QSplitter, QProgressBar,
    QGridLayout, QTextEdit, QTabWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QFont, QColor, QPalette, QTextCursor
import pyqtgraph as pg


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
        return f"{amps * 1e6:.4f} µA"
    if a >= 1e-9:
        return f"{amps * 1e9:.3f} nA"
    return f"{amps * 1e12:.2f} pA"


def list_serial_ports() -> list:
    return [p.device for p in serial.tools.list_ports.comports()]


# ---------------------------------------------------------------------------
#  Instrument drivers (with logging callback)
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
            # last resort: check if anything accumulated in buffer
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
        # enforce minimum compliance
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


class Tenma72_2715:
    """TENMA 72-2715 (Korad KA-protocol) via USB-serial."""

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 1,
                 log=None):
        self._log = log or (lambda *a: None)
        self._log(f"[TENMA] Opening {port} @ {baudrate} baud …")
        self.ser = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=8, parity="N", stopbits=1,
            timeout=timeout
        )
        time.sleep(0.3)
        self._log("[TENMA] Port opened OK")

    def idn(self) -> str:
        self.ser.reset_input_buffer()
        self.ser.write(b"*IDN?")
        self._log("[TENMA] TX: *IDN?")
        time.sleep(0.2)
        resp = self.ser.read(50).decode(errors="replace").strip("\x00").strip()
        self._log(f"[TENMA] RX: '{resp}'")
        return resp

    def set_voltage(self, voltage: float):
        cmd = f"VSET1:{voltage:05.2f}"
        self.ser.write(cmd.encode())
        self._log(f"[TENMA] TX: {cmd}")
        time.sleep(0.1)

    def get_voltage(self) -> float:
        self.ser.reset_input_buffer()
        self.ser.write(b"VOUT1?")
        time.sleep(0.1)
        raw = self.ser.read(5).decode(errors="replace")
        self._log(f"[TENMA] VOUT1? → '{raw}'")
        return float(raw)

    def set_current_limit(self, current: float):
        cmd = f"ISET1:{current:05.3f}"
        self.ser.write(cmd.encode())
        self._log(f"[TENMA] TX: {cmd}")
        time.sleep(0.1)

    def output_on(self):
        self.ser.write(b"OUT1")
        self._log("[TENMA] TX: OUT1")
        time.sleep(0.2)

    def output_off(self):
        self.ser.write(b"OUT0")
        self._log("[TENMA] TX: OUT0")
        time.sleep(0.2)

    def close(self):
        try:
            self.output_off()
        except Exception:
            pass
        self.ser.close()
        self._log("[TENMA] Closed")


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
                 led_voltage: float = 0.0,
                 parent=None):
        super().__init__(parent)
        self.keithley = keithley
        self.segments = segments
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
        # deduplicate at segment boundaries, keep later config
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
            # safe shutdown
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

        self.log_msg.emit(f"▶ Sweep: {total} points, LED = {self.led_voltage:.1f} V")

        for idx, (v_set, settle_ms, n_avg) in enumerate(points):
            if self._stop_flag:
                self.log_msg.emit("■ Sweep stopped by user")
                break

            # set voltage
            self.keithley.set_voltage(v_set)
            self.progress_update.emit(idx + 1, total)

            # settle
            self.log_msg.emit(
                f"  [{idx+1}/{total}] V = {v_set:+.4f} V, "
                f"settling {settle_ms} ms …"
            )
            time.sleep(settle_ms / 1000.0)

            # read with averaging
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

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photoelectric Effect — I-V Measurement  v2")
        self.resize(1280, 780)

        self.keithley: Optional[Keithley2400] = None
        self.tenma: Optional[Tenma72_2715] = None
        self.worker: Optional[SweepWorker] = None
        self.data: List[DataPoint] = []
        self.settings = QSettings("Technion_Physics", "Photoelectric")
        self._k_term_tx = "\n"  # default TX terminator for Keithley

        # color cycle for multiple sweeps at different LED voltages
        self.colors = [
            "#2196F3", "#F44336", "#4CAF50", "#FF9800",
            "#9C27B0", "#00BCD4", "#FFEB3B", "#E91E63",
            "#8BC34A", "#607D8B",
        ]
        self.sweep_curves = []  # list of (led_v, curve, error_bars)

        self._build_ui()
        self._restore_settings()
        self._refresh_ports()

    # -- logging --

    def log(self, msg: str):
        print(msg, flush=True)  # console output for copy-paste
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
        left.setMaximumWidth(430)
        left.setMinimumWidth(350)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        lv.setSpacing(6)

        # --- Connections ---
        conn_grp = QGroupBox("Connections")
        cg = QGridLayout(conn_grp)

        cg.addWidget(QLabel("Keithley 2400:"), 0, 0)
        self.cb_k_port = QComboBox()
        self.cb_k_port.setEditable(True)
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

        cg.addWidget(QLabel("TENMA (LED):"), 2, 0)
        self.cb_t_port = QComboBox()
        self.cb_t_port.setEditable(True)
        cg.addWidget(self.cb_t_port, 2, 1)
        self.btn_t_conn = QPushButton("Connect")
        self.btn_t_conn.clicked.connect(self._toggle_tenma)
        cg.addWidget(self.btn_t_conn, 2, 4)

        self.btn_refresh = QPushButton("↻ Ports")
        self.btn_refresh.clicked.connect(self._refresh_ports)
        cg.addWidget(self.btn_refresh, 3, 4)

        self.lbl_k_idn = QLabel("Keithley: not connected")
        self.lbl_k_idn.setWordWrap(True)
        cg.addWidget(self.lbl_k_idn, 4, 0, 1, 5)
        self.lbl_t_idn = QLabel("TENMA: not connected")
        self.lbl_t_idn.setWordWrap(True)
        cg.addWidget(self.lbl_t_idn, 5, 0, 1, 5)

        # raw terminal for debugging
        raw_row = QHBoxLayout()
        self.le_raw_cmd = QComboBox()
        self.le_raw_cmd.setEditable(True)
        self.le_raw_cmd.addItems(["*IDN?", "*RST", ":SYST:ERR?", ":OUTP?",
                                  ":SOUR:VOLT?", ":MEAS:CURR?", ":READ?"])
        self.le_raw_cmd.setMinimumWidth(150)
        raw_row.addWidget(QLabel("Raw cmd:"))
        raw_row.addWidget(self.le_raw_cmd)
        self.btn_raw_send = QPushButton("Send")
        self.btn_raw_send.clicked.connect(self._send_raw_cmd)
        raw_row.addWidget(self.btn_raw_send)
        cg.addLayout(raw_row, 6, 0, 1, 5)

        lv.addWidget(conn_grp)

        # --- Light source (TENMA → LED) ---
        led_grp = QGroupBox("Light source (TENMA → LED)")
        led_lay = QGridLayout(led_grp)

        led_lay.addWidget(QLabel("LED voltage:"), 0, 0)
        self.sb_led_v = QDoubleSpinBox()
        self.sb_led_v.setRange(0, 30)
        self.sb_led_v.setValue(12.0)
        self.sb_led_v.setDecimals(1)
        self.sb_led_v.setSuffix(" V")
        led_lay.addWidget(self.sb_led_v, 0, 1)

        self.btn_led_set = QPushButton("Set LED voltage")
        self.btn_led_set.clicked.connect(self._set_led_voltage)
        led_lay.addWidget(self.btn_led_set, 0, 2)

        self.btn_led_off = QPushButton("LED Off")
        self.btn_led_off.clicked.connect(self._led_off)
        led_lay.addWidget(self.btn_led_off, 0, 3)

        led_lay.addWidget(QLabel("Current limit:"), 1, 0)
        self.sb_led_ilim = QDoubleSpinBox()
        self.sb_led_ilim.setRange(0.001, 5.0)
        self.sb_led_ilim.setValue(0.5)
        self.sb_led_ilim.setDecimals(3)
        self.sb_led_ilim.setSuffix(" A")
        led_lay.addWidget(self.sb_led_ilim, 1, 1)

        lv.addWidget(led_grp)

        # --- Measurement parameters ---
        meas_grp = QGroupBox("Keithley measurement parameters")
        mg = QGridLayout(meas_grp)

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
        self.sb_compl.setRange(0.000001, 1.0)   # 1 µA to 1 A
        self.sb_compl.setSingleStep(0.001)
        self.sb_compl.setValue(0.001)             # 1 mA default
        self.sb_compl.setSuffix(" A")
        mg.addWidget(self.sb_compl, 1, 1)

        lv.addWidget(meas_grp)

        # --- Sweep segments ---
        seg_grp = QGroupBox("Sweep segments (Keithley → photocell)")
        sv = QVBoxLayout(seg_grp)

        self.tbl_seg = QTableWidget(0, 5)
        self.tbl_seg.setHorizontalHeaderLabels(
            ["Start V", "End V", "Step V", "Settle ms", "Avg N"]
        )
        hdr = self.tbl_seg.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.tbl_seg.setMaximumHeight(140)
        self.tbl_seg.cellChanged.connect(self._update_pts_label)
        sv.addWidget(self.tbl_seg)

        sb = QHBoxLayout()
        btn_a = QPushButton("+ Add")
        btn_a.clicked.connect(lambda: self._add_seg_row())
        btn_d = QPushButton("− Del")
        btn_d.clicked.connect(self._del_seg_row)
        btn_def = QPushButton("Defaults")
        btn_def.clicked.connect(self._default_segments)
        sb.addWidget(btn_a)
        sb.addWidget(btn_d)
        sb.addWidget(btn_def)
        sv.addLayout(sb)

        self.lbl_pts = QLabel("≈ 0 points")
        sv.addWidget(self.lbl_pts)

        lv.addWidget(seg_grp)

        # --- Controls ---
        ctrl_grp = QGroupBox("Sweep control")
        cv = QVBoxLayout(ctrl_grp)

        r1 = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start sweep")
        self.btn_start.setStyleSheet("font-weight:bold; padding:8px;")
        self.btn_start.clicked.connect(self._start_sweep)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_sweep)
        r1.addWidget(self.btn_start)
        r1.addWidget(self.btn_stop)
        cv.addLayout(r1)

        # manual single-point
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Manual V:"))
        self.sb_man_v = QDoubleSpinBox()
        self.sb_man_v.setRange(-21, 21)
        self.sb_man_v.setDecimals(3)
        self.sb_man_v.setSuffix(" V")
        r2.addWidget(self.sb_man_v)
        self.btn_man = QPushButton("Set+Read")
        self.btn_man.clicked.connect(self._manual_measure)
        r2.addWidget(self.btn_man)
        cv.addLayout(r2)
        self.lbl_man = QLabel("")
        cv.addWidget(self.lbl_man)

        lv.addWidget(ctrl_grp)

        # --- Export ---
        er = QHBoxLayout()
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.clicked.connect(self._export_csv)
        self.btn_clear = QPushButton("🗑 Clear all data")
        self.btn_clear.clicked.connect(self._clear_data)
        er.addWidget(self.btn_export)
        er.addWidget(self.btn_clear)
        lv.addLayout(er)

        lv.addStretch()

        # ═══ RIGHT PANEL ═══
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(2, 2, 2, 2)

        # plot
        pg.setConfigOptions(antialias=True, background='w', foreground='k')
        self.plot_w = pg.PlotWidget()
        self.plot_w.setLabel("bottom", "Voltage", units="V")
        self.plot_w.setLabel("left", "Current", units="A")
        self.plot_w.setTitle("I-V Characteristic")
        self.plot_w.showGrid(x=True, y=True, alpha=0.3)
        self.plot_w.addLegend(offset=(60, 10))
        rv.addWidget(self.plot_w, stretch=3)

        # tabs: data table + log
        tabs = QTabWidget()

        # data table
        self.tbl_data = QTableWidget(0, 5)
        self.tbl_data.setHorizontalHeaderLabels(
            ["LED V", "V_set (V)", "I_avg", "I_avg (A)", "σ_I (A)"]
        )
        self.tbl_data.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tbl_data.setAlternatingRowColors(True)
        tabs.addTab(self.tbl_data, "Data")

        # log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        tabs.addTab(self.log_text, "Log")

        rv.addWidget(tabs, stretch=1)

        # status bar
        self.statusBar().showMessage("Ready")
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.statusBar().addPermanentWidget(self.progress)

        # assemble
        main_h.addWidget(left, stretch=0)
        main_h.addWidget(right, stretch=1)

        # defaults
        self._default_segments()

    # ---------------------------------------------------------------
    #  Settings persistence
    # ---------------------------------------------------------------

    def _restore_settings(self):
        pk = self.settings.value("k_port", "")
        pt = self.settings.value("t_port", "")
        if pk:
            self.cb_k_port.setCurrentText(pk)
        if pt:
            self.cb_t_port.setCurrentText(pt)

    def _save_settings(self):
        self.settings.setValue("k_port", self.cb_k_port.currentText())
        self.settings.setValue("t_port", self.cb_t_port.currentText())

    def closeEvent(self, event):
        self._save_settings()
        self._disconnect_all()
        event.accept()

    # ---------------------------------------------------------------
    #  Ports
    # ---------------------------------------------------------------

    def _refresh_ports(self):
        ports = list_serial_ports()
        for cb in (self.cb_k_port, self.cb_t_port):
            prev = cb.currentText()
            cb.clear()
            cb.addItems(ports)
            if prev in ports:
                cb.setCurrentText(prev)
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

    # ---------------------------------------------------------------
    #  TENMA connection
    # ---------------------------------------------------------------

    def _toggle_tenma(self):
        if self.tenma:
            self.tenma.close()
            self.tenma = None
            self.btn_t_conn.setText("Connect")
            self.lbl_t_idn.setText("TENMA: not connected")
            self.log("[TENMA] Disconnected")
            return

        port = self.cb_t_port.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Error", "Select TENMA COM port")
            return

        try:
            self.tenma = Tenma72_2715(port, log=self.log)
            idn = self.tenma.idn()
            if idn:
                self.lbl_t_idn.setText(f"TENMA: {idn}")
                self.log(f"[TENMA] Connected ✓  IDN: {idn}")
            else:
                self.lbl_t_idn.setText("TENMA: connected (no IDN)")
                self.log("[TENMA] Port opened, IDN empty — may still work")
            self.btn_t_conn.setText("Disconnect")
        except Exception as e:
            self.tenma = None
            self.log(f"[TENMA] ✗ Connection failed: {e}")
            QMessageBox.critical(self, "TENMA", f"Connection failed:\n{e}")

    def _disconnect_all(self):
        for inst in (self.keithley, self.tenma):
            if inst:
                try:
                    inst.close()
                except Exception:
                    pass
        self.keithley = None
        self.tenma = None

    # ---------------------------------------------------------------
    #  Auto-scan & raw terminal
    # ---------------------------------------------------------------

    def _auto_scan_keithley(self):
        """Try all baud × flow combinations to find working settings."""
        port = self.cb_k_port.currentText().strip()
        if not port:
            QMessageBox.warning(self, "Error", "Select Keithley COM port first")
            return

        # disconnect if connected
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
                if self._stop_flag_scan if hasattr(self, '_stop_flag_scan') else False:
                    break
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

                    # Try sending *IDN? with different terminators
                    for term_name, term in [("LF", "\n"), ("CR+LF", "\r\n"), ("CR", "\r")]:
                        ser.reset_input_buffer()
                        ser.write(f"*IDN?{term}".encode())

                        # char-by-char read — handles any response terminator
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
                            # fallback: raw read
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

                            # apply found settings
                            self.cb_k_baud.setCurrentText(str(baud))
                            self.cb_k_flow.setCurrentText(flow)
                            self._k_term_tx = term  # remember working terminator
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
        self.log("[Scan]   Possible causes:")
        self.log("[Scan]   1. Wrong COM port (is this really the Keithley?)")
        self.log("[Scan]   2. Cable issue (check RS-232 ↔ USB adapter)")
        self.log("[Scan]   3. Keithley set to GPIB mode, not RS-232")
        self.log("[Scan]      → On instrument: Menu → Communication → RS-232")
        self.log("[Scan]   4. COM-USB adapter is actually GPIB-USB → need pyvisa")
        QMessageBox.warning(
            self, "Not found",
            "No response from Keithley at any baud/flow setting.\n\n"
            "Check:\n"
            "• Is this the correct COM port?\n"
            "• Is the cable RS-232 (not GPIB)?\n"
            "• On the instrument: Menu → Communication → RS-232"
        )

    def _send_raw_cmd(self):
        """Send a raw SCPI command and show response."""
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
    #  LED control
    # ---------------------------------------------------------------

    def _set_led_voltage(self):
        if not self.tenma:
            QMessageBox.warning(self, "Error", "TENMA not connected")
            return
        v = self.sb_led_v.value()
        try:
            self.tenma.set_current_limit(self.sb_led_ilim.value())
            self.tenma.set_voltage(v)
            self.tenma.output_on()
            self.log(f"[LED] Set to {v:.1f} V, I_lim = {self.sb_led_ilim.value():.3f} A")
        except Exception as e:
            self.log(f"[LED] Error: {e}")

    def _led_off(self):
        if self.tenma:
            try:
                self.tenma.output_off()
                self.log("[LED] Output OFF")
            except Exception as e:
                self.log(f"[LED] Error: {e}")

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

    def _default_segments(self):
        self.tbl_seg.setRowCount(0)
        self._add_seg_row(-5, -1, 0.5, 500, 3)
        self._add_seg_row(-1, 1, 0.1, 800, 5)
        self._add_seg_row(1, 15, 0.5, 500, 3)

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

        led_v = self.sb_led_v.value()
        self.log("=" * 50)
        self.log(f"Starting sweep, LED = {led_v:.1f} V")

        # configure keithley
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

            # check for errors after config
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

            # test read at 0V
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

        # UI state
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)

        # create curve for this sweep
        ci = len(self.sweep_curves) % len(self.colors)
        color = self.colors[ci]
        curve = self.plot_w.plot(
            [], [],
            pen=pg.mkPen(color, width=2),
            symbol="o", symbolSize=5, symbolBrush=color,
            name=f"LED {led_v:.0f}V"
        )
        err_bars = pg.ErrorBarItem(pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine))
        self.plot_w.addItem(err_bars)
        self.sweep_curves.append((led_v, curve, err_bars))

        # worker
        self.worker = SweepWorker(self.keithley, segs, led_v)
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

        # table
        r = self.tbl_data.rowCount()
        self.tbl_data.insertRow(r)
        vals = [
            f"{dp.led_voltage:.1f}",
            f"{dp.voltage_set:+.4f}",
            format_current(dp.current_avg),
            f"{dp.current_avg:.6e}",
            f"{dp.current_std:.3e}",
        ]
        for col, v in enumerate(vals):
            self.tbl_data.setItem(r, col, QTableWidgetItem(v))
        self.tbl_data.scrollToBottom()

        # update the latest curve
        if self.sweep_curves:
            led_v, curve, err_bars = self.sweep_curves[-1]
            pts = [d for d in self.data if d.led_voltage == led_v]
            vv = [p.voltage_set for p in pts]
            ii = [p.current_avg for p in pts]
            ss = [p.current_std for p in pts]
            curve.setData(vv, ii)
            err_bars.setData(
                x=np.array(vv), y=np.array(ii),
                height=2 * np.array(ss)
            )

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
        n = sum(1 for d in self.data
                if self.sweep_curves and d.led_voltage == self.sweep_curves[-1][0])
        self.log(f"Sweep complete — {n} points collected")

    # ---------------------------------------------------------------
    #  Export / Clear
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
            w.writerow([
                "LED_V", "V_set", "V_actual", "I_avg_A",
                "I_std_A", "N", "Readings_A"
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
                ])
        self.log(f"Saved: {path}")

    def _clear_data(self):
        if self.data:
            r = QMessageBox.question(
                self, "Clear", "Clear all data and curves?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        self.data.clear()
        self.tbl_data.setRowCount(0)
        for _, curve, ebars in self.sweep_curves:
            self.plot_w.removeItem(curve)
            self.plot_w.removeItem(ebars)
        self.sweep_curves.clear()
        self.log("Data cleared")


# ---------------------------------------------------------------------------
#  Entry
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # light theme — just use Fusion defaults, no custom palette needed
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
