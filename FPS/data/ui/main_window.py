"""
MainWindow — toàn bộ layout PyQt5.
Chỉnh sửa file này để thay đổi giao diện.
Logic phân tích nằm ở core/, theme nằm ở ui/theme.py.
"""
from __future__ import annotations
import json
import os

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QTextEdit,
    QFrame, QScrollArea, QGridLayout, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMessageBox, QButtonGroup, QRadioButton,
    QSpinBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from .theme   import DARK
from .widgets import DropZone, StatCard, LatencyBar, StepProgressWidget
from .worker  import AnalyzeWorker
from ..core.device import detect_all, best_torch_device, list_torch_devices
from ..utils  import fmt_flops, fmt_params


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Profiler — Edge AI")
        self.resize(1320, 920)
        self.setMinimumSize(980, 700)

        self._devices          = detect_all()
        self._torch_dev        = best_torch_device()
        self._worker: AnalyzeWorker | None = None
        self._result: dict | None          = None
        self._pending_weights: str | None  = None

        # device radio buttons — populated in _make_device_selector()
        self._dev_btn_group: QButtonGroup | None = None
        self._dev_radio_vals: list[str] = []

        self._build_ui()
        self._populate_device_info()

    # ══════════════════════════════════════════════════════════════════════
    # UI BUILD
    # ══════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)

        vlay.addWidget(self._make_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {DARK['border']}; }}")
        splitter.addWidget(self._make_left_panel())
        splitter.addWidget(self._make_right_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([380, 940])
        vlay.addWidget(splitter, stretch=1)

        self.status_lbl = QLabel("Sẵn sàng. Kéo file model vào để bắt đầu.")
        self.status_lbl.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 12px; padding: 9px 20px;"
            f" background: {DARK['surface']};"
            f" border-top: 1px solid {DARK['border']};")
        vlay.addWidget(self.status_lbl)

    # ── Header ────────────────────────────────────────────────────────────

    def _make_header(self) -> QFrame:
        frame = QFrame()
        frame.setFixedHeight(68)
        frame.setStyleSheet(
            f"background: {DARK['surface']};"
            f" border-bottom: 1px solid {DARK['border']};")

        h = QHBoxLayout(frame)
        h.setContentsMargins(26, 0, 26, 0)
        h.setSpacing(12)

        title = QLabel("⚡  Model Profiler")
        title.setStyleSheet(
            f"font-size: 21px; font-weight: 800; color: {DARK['accent2']};")

        def _badge(color: str) -> QLabel:
            lbl = QLabel()
            lbl.setStyleSheet(
                f"color: {color}; font-size: 12px; font-weight: 600;"
                f" background: {DARK['card']};"
                f" border: 1px solid {DARK['border']};"
                " border-radius: 8px; padding: 6px 14px;")
            return lbl

        self.device_badge = _badge(DARK["green"])
        self.torch_badge  = _badge(DARK["accent3"])
        self.ort_badge    = _badge(DARK["yellow"])

        h.addWidget(title)
        h.addSpacing(20)
        h.addWidget(self.device_badge)
        h.addWidget(self.torch_badge)
        h.addWidget(self.ort_badge)
        h.addStretch()

        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setMinimumHeight(38)
        self.export_btn.setMinimumWidth(130)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_json)
        h.addWidget(self.export_btn)

        return frame

    # ── Left panel ────────────────────────────────────────────────────────

    def _make_left_panel(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {DARK['surface']};")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(14)

        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        lay.addWidget(self.drop_zone)

        # File slots
        lay.addWidget(self._make_file_slots())

        # Device selector + benchmark settings
        lay.addWidget(self._make_device_selector())

        # Run / Stop
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.setMinimumHeight(48)
        self.run_btn.setToolTip("Bắt đầu phân tích (Enter)")
        self.run_btn.setShortcut("Return")
        self.run_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: 700; }")
        self.run_btn.clicked.connect(self._on_run_clicked)

        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setToolTip("Dừng phân tích")
        self.stop_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: 700; }")
        self.stop_btn.clicked.connect(self._on_stop_clicked)

        btn_row.addWidget(self.run_btn, stretch=3)
        btn_row.addWidget(self.stop_btn, stretch=1)
        lay.addLayout(btn_row)

        # Step progress
        self.step_progress = StepProgressWidget()
        lay.addWidget(self.step_progress)

        # Device info text
        dev_title = QLabel("THIẾT BỊ")
        dev_title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px; margin-top: 6px;")
        lay.addWidget(dev_title)

        self.dev_text = QTextEdit()
        self.dev_text.setReadOnly(True)
        self.dev_text.setStyleSheet(
            f"font-family: 'JetBrains Mono','Consolas',monospace; font-size: 12px;"
            f" background: {DARK['card']}; border: 1px solid {DARK['border']};"
            " border-radius: 8px; padding: 8px;")
        lay.addWidget(self.dev_text, stretch=1)

        return w

    def _make_file_slots(self) -> QFrame:
        frame = QFrame()
        frame.setMinimumHeight(108)
        frame.setStyleSheet(f"""
            QFrame {{
                background: {DARK['card']};
                border: 1px solid {DARK['border']};
                border-radius: 12px;
            }}
        """)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        # Weights slot
        w_row = QHBoxLayout()
        w_row.setSpacing(10)
        self.slot_weights_icon = QLabel("○")
        self.slot_weights_icon.setStyleSheet(
            f"color: {DARK['muted2']}; font-size: 18px; min-width: 24px;")
        self.slot_weights_lbl = QLabel("Weights: chưa có file")
        self.slot_weights_lbl.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 13px;")
        self.slot_weights_lbl.setWordWrap(True)
        w_row.addWidget(self.slot_weights_icon)
        w_row.addWidget(self.slot_weights_lbl, stretch=1)
        lay.addLayout(w_row)

        return frame

    def _make_device_selector(self) -> QFrame:
        """
        Frame chọn device (radio) + nhập warmup/runs.
        Tự động build radio buttons từ list_torch_devices().
        """
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background: {DARK['card']};
                border: 1px solid {DARK['border']};
                border-radius: 12px;
            }}
        """)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(10)

        # ── Device radio ──────────────────────────────────────────────────
        dev_lbl = QLabel("DEVICE")
        dev_lbl.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px;")
        lay.addWidget(dev_lbl)

        self._dev_btn_group  = QButtonGroup(self)
        self._dev_radio_vals = []

        dev_list = list_torch_devices()
        for i, dev in enumerate(dev_list):
            rb = QRadioButton(dev["label"])
            rb.setStyleSheet(f"""
                QRadioButton {{
                    color: {DARK['text']}; font-size: 13px; spacing: 8px;
                }}
                QRadioButton::indicator {{
                    width: 16px; height: 16px;
                    border: 2px solid {DARK['border']};
                    border-radius: 8px;
                    background: {DARK['card2']};
                }}
                QRadioButton::indicator:checked {{
                    background: {DARK['accent']};
                    border-color: {DARK['accent']};
                }}
            """)
            rb.setChecked(dev["is_default"])
            self._dev_btn_group.addButton(rb, i)
            self._dev_radio_vals.append(dev["value"])
            lay.addWidget(rb)

        # fallback nếu không có device nào
        if not dev_list:
            rb = QRadioButton("CPU")
            rb.setChecked(True)
            self._dev_btn_group.addButton(rb, 0)
            self._dev_radio_vals.append("cpu")
            lay.addWidget(rb)

        # ── Warmup / Runs ─────────────────────────────────────────────────
        bench_lbl = QLabel("BENCHMARK")
        bench_lbl.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px; margin-top: 4px;")
        lay.addWidget(bench_lbl)

        spin_row = QHBoxLayout()
        spin_row.setSpacing(12)

        # Warmup
        wu_lbl = QLabel("Warmup")
        wu_lbl.setStyleSheet(f"color: {DARK['muted']}; font-size: 12px;")
        self.spin_warmup = QSpinBox()
        self.spin_warmup.setRange(0, 10000)
        self.spin_warmup.setValue(0)          # 0 = dùng default theo device
        self.spin_warmup.setSpecialValueText("auto")
        self.spin_warmup.setFixedWidth(90)
        self.spin_warmup.setFixedHeight(34)
        self.spin_warmup.setStyleSheet(f"""
            QSpinBox {{
                background: {DARK['card2']}; color: {DARK['text']};
                border: 1px solid {DARK['border']}; border-radius: 7px;
                padding: 4px 8px; font-size: 13px;
            }}
            QSpinBox::up-button, QSpinBox::down-button {{
                width: 20px;
                background: {DARK['border']};
                border-radius: 4px;
            }}
        """)

        # Runs
        r_lbl = QLabel("Runs")
        r_lbl.setStyleSheet(f"color: {DARK['muted']}; font-size: 12px;")
        self.spin_runs = QSpinBox()
        self.spin_runs.setRange(0, 100000)
        self.spin_runs.setValue(0)            # 0 = dùng default theo device
        self.spin_runs.setSpecialValueText("auto")
        self.spin_runs.setFixedWidth(90)
        self.spin_runs.setFixedHeight(34)
        self.spin_runs.setStyleSheet(self.spin_warmup.styleSheet())

        spin_row.addWidget(wu_lbl)
        spin_row.addWidget(self.spin_warmup)
        spin_row.addSpacing(8)
        spin_row.addWidget(r_lbl)
        spin_row.addWidget(self.spin_runs)
        spin_row.addStretch()
        lay.addLayout(spin_row)

        hint = QLabel("0 / auto = dùng default theo device")
        hint.setStyleSheet(
            f"color: {DARK['muted2']}; font-size: 11px; font-style: italic;")
        lay.addWidget(hint)

        return frame

    # ── Right panel (tabs) ────────────────────────────────────────────────

    def _make_right_panel(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                font-size: 13px; font-weight: 600;
                padding: 10px 18px; min-width: 90px;
            }
        """)
        self.tabs.addTab(self._make_overview_tab(), "📊  Tổng quan")
        self.tabs.addTab(self._make_layers_tab(),   "🗂  Layers")
        self.tabs.addTab(self._make_latency_tab(),  "⏱  Latency")

        self.tab_raw = QTextEdit()
        self.tab_raw.setReadOnly(True)
        self.tab_raw.setStyleSheet("font-size: 12px;")
        self.tab_raw.setPlaceholderText(
            "Raw JSON kết quả sẽ xuất hiện ở đây...")
        self.tabs.addTab(self.tab_raw, "{ }  Raw")
        return self.tabs

    def _make_overview_tab(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(22, 22, 22, 22)
        lay.setSpacing(18)

        grid = QGridLayout()
        grid.setSpacing(16)
        self.card_params   = StatCard("Parameters")
        self.card_flops    = StatCard("FLOPs")
        self.card_size     = StatCard("Model size")
        self.card_filesize = StatCard("File size")
        self.card_fps      = StatCard("Throughput")
        self.card_latency  = StatCard("Latency (mean)")
        grid.addWidget(self.card_params,   0, 0)
        grid.addWidget(self.card_flops,    0, 1)
        grid.addWidget(self.card_size,     0, 2)
        grid.addWidget(self.card_filesize, 1, 0)
        grid.addWidget(self.card_fps,      1, 1)
        grid.addWidget(self.card_latency,  1, 2)
        lay.addLayout(grid)

        meta_title = QLabel("METADATA")
        meta_title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px; margin-top: 8px;")
        lay.addWidget(meta_title)
        self.meta_table = self._make_kv_table()
        lay.addWidget(self.meta_table, stretch=1)
        scroll.setWidget(content)
        return scroll

    def _make_layers_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)

        hdr = QHBoxLayout()
        title = QLabel("Layer / Initializer list")
        title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 12px; font-weight: 700;")
        self.layers_count = QLabel("")
        self.layers_count.setStyleSheet(
            f"color: {DARK['accent2']}; font-size: 12px;")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.layers_count)
        lay.addLayout(hdr)

        self.layers_table = QTableWidget(0, 4)
        self.layers_table.setHorizontalHeaderLabels(
            ["Tên layer", "Shape", "Params", "Dtype"])
        self.layers_table.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QHeaderView::section {
                font-size: 12px; font-weight: 700; padding: 8px;
            }
        """)
        hh = self.layers_table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hh.setFixedHeight(36)
        self.layers_table.verticalHeader().setDefaultSectionSize(32)
        self.layers_table.setAlternatingRowColors(True)
        self.layers_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.layers_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.layers_table)
        return w

    def _make_latency_tab(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        lay = QVBoxLayout(content)
        lay.setContentsMargins(22, 22, 22, 22)
        lay.setSpacing(14)

        title = QLabel("LATENCY BREAKDOWN")
        title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px;")
        lay.addWidget(title)

        self.bar_mean = LatencyBar("Mean")
        self.bar_min  = LatencyBar("Min")
        self.bar_max  = LatencyBar("Max")
        self.bar_p50  = LatencyBar("P50")
        self.bar_p95  = LatencyBar("P95 (tail)")
        for b in (self.bar_mean, self.bar_min, self.bar_max,
                  self.bar_p50, self.bar_p95):
            lay.addWidget(b)

        info_title = QLabel("RUN INFO")
        info_title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px; margin-top: 14px;")
        lay.addWidget(info_title)
        self.lat_info_table = self._make_kv_table()
        lay.addWidget(self.lat_info_table, stretch=1)
        scroll.setWidget(content)
        return scroll

    def _make_kv_table(self) -> QTableWidget:
        t = QTableWidget(0, 2)
        t.setHorizontalHeaderLabels(["Key", "Value"])
        t.setStyleSheet("""
            QTableWidget { font-size: 12px; }
            QHeaderView::section {
                font-size: 12px; font-weight: 700; padding: 8px;
            }
        """)
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        t.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        t.horizontalHeader().setFixedHeight(34)
        t.verticalHeader().setVisible(False)
        t.verticalHeader().setDefaultSectionSize(30)
        t.setAlternatingRowColors(True)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        return t

    # ══════════════════════════════════════════════════════════════════════
    # Device info
    # ══════════════════════════════════════════════════════════════════════

    def _populate_device_info(self):
        d = self._devices
        lines = []
        board = d.get("board")
        if board:
            lines.append(f"🔲 {board}")
        lines.append(f"💻 {d['hostname']}  ({d['os']})")
        lines.append(f"🔧 {d['arch']}")
        lines.append("")
        lines.append(f"🖥 {d['cpu_name']}")
        lines.append(
            f"   {d['cpu_cores_physical']}P / {d['cpu_cores_logical']}L cores")
        lines.append(f"   {d['cpu_freq']}")
        lines.append(f"   RAM: {d['ram_total']} (avail {d['ram_avail']})")
        lines.append("")

        tv = d.get("torch_version")
        if tv:
            lines.append(f"🔥 PyTorch {tv}")
            if d.get("cuda_available"):
                lines.append(f"   CUDA {d.get('cuda_version', '?')}")
                for gpu in d.get("cuda_devices", []):
                    lines.append(f"   🎮 {gpu['name']}")
                    lines.append(
                        f"      VRAM: {gpu['vram']}  {gpu['sm']}  {gpu['mp']} MPs")
                if d.get("cudnn_version"):
                    lines.append(f"   cuDNN: {d['cudnn_version']}")
            elif d.get("mps_available"):
                lines.append("   🍎 MPS (Apple Silicon)")
            else:
                lines.append("   CPU only")
        else:
            lines.append("⚠ PyTorch: chưa cài")
        lines.append("")
        ort_v = d.get("onnxrt_version")
        if ort_v:
            lines.append(f"🟡 ONNX Runtime {ort_v}")
            for p in d.get("onnxrt_providers", []):
                lines.append(f"   • {p}")
        else:
            lines.append("⚠ ONNX Runtime: chưa cài")

        self.dev_text.setPlainText("\n".join(lines))

        # badges
        if board:
            self.device_badge.setText(f"🔲 {board[:30]}")
        elif d.get("cuda_available") and d.get("cuda_devices"):
            self.device_badge.setText(f"🎮 {d['cuda_devices'][0]['name']}")
        elif d.get("mps_available"):
            self.device_badge.setText("🍎 Apple MPS")
        else:
            self.device_badge.setText(f"🖥 {d['cpu_name'][:28]}")

        if tv:
            dev_str = ("CUDA" if d.get("cuda_available")
                       else "MPS" if d.get("mps_available") else "CPU")
            self.torch_badge.setText(f"PyTorch {tv} · {dev_str}")
        else:
            self.torch_badge.setText("PyTorch ✗")

        self.ort_badge.setText(
            f"OnnxRT {ort_v}" if ort_v else "OnnxRT ✗")

    # ══════════════════════════════════════════════════════════════════════
    # Helpers lấy giá trị từ UI controls
    # ══════════════════════════════════════════════════════════════════════

    def _get_selected_device(self) -> str:
        """Trả về device string từ radio button đang chọn."""
        if self._dev_btn_group is None:
            return self._torch_dev
        checked_id = self._dev_btn_group.checkedId()
        if 0 <= checked_id < len(self._dev_radio_vals):
            return self._dev_radio_vals[checked_id]
        return self._torch_dev

    def _get_warmup(self) -> int | None:
        """0 (auto) → None, else giá trị từ spinbox."""
        v = self.spin_warmup.value()
        return None if v == 0 else v

    def _get_runs(self) -> int | None:
        """0 (auto) → None, else giá trị từ spinbox."""
        v = self.spin_runs.value()
        return None if v == 0 else v

    # ══════════════════════════════════════════════════════════════════════
    # File selection
    # ══════════════════════════════════════════════════════════════════════

    def _on_files_dropped(self, weights_path, _script_path=None):
        if self._worker and self._worker.isRunning():
            self.status_lbl.setText(
                "⚠ Đang analyze — đợi xong rồi chọn file mới.")
            return
        if weights_path:
            self._pending_weights = weights_path
        self._update_slots()

    def _update_slots(self):
        if self._pending_weights:
            fname = os.path.basename(self._pending_weights)
            fsize = os.path.getsize(self._pending_weights) / 1e6
            ext   = os.path.splitext(self._pending_weights)[1].upper()[1:]
            self.slot_weights_icon.setText("●")
            self.slot_weights_icon.setStyleSheet(
                f"color: {DARK['green']}; font-size: 18px; min-width: 24px;")
            self.slot_weights_lbl.setText(
                f"{fname}  ({fsize:.2f} MB · {ext})")
            self.slot_weights_lbl.setStyleSheet(
                f"color: {DARK['text']}; font-size: 13px;")
        else:
            self.slot_weights_icon.setText("○")
            self.slot_weights_icon.setStyleSheet(
                f"color: {DARK['muted2']}; font-size: 18px; min-width: 24px;")
            self.slot_weights_lbl.setText("Weights: chưa có file")
            self.slot_weights_lbl.setStyleSheet(
                f"color: {DARK['muted']}; font-size: 13px;")

        self.run_btn.setEnabled(bool(self._pending_weights))
        self.step_progress.set_idle()
        if self._pending_weights:
            self.status_lbl.setText(
                f"Sẵn sàng: {os.path.basename(self._pending_weights)}"
                " — nhấn ▶ Run")

    # ══════════════════════════════════════════════════════════════════════
    # Run / Stop
    # ══════════════════════════════════════════════════════════════════════

    def _on_run_clicked(self):
        if not self._pending_weights:
            return
        self._start_analysis(self._pending_weights)

    def _on_stop_clicked(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(2000)
        self.run_btn.setEnabled(bool(self._pending_weights))
        self.stop_btn.setEnabled(False)
        self.step_progress.set_error("Đã dừng bởi người dùng")
        self.status_lbl.setText("⏹ Đã dừng.")

    def _start_analysis(self, weights_path: str):
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.step_progress.set_running()
        self.tab_raw.setPlainText("")
        self._clear_overview()

        device_str = self._get_selected_device()
        warmup     = self._get_warmup()
        runs       = self._get_runs()

        self._worker = AnalyzeWorker(
            weights_path=weights_path,
            device_str=device_str,
            warmup=warmup,
            runs=runs,
        )
        self._worker.progress.connect(self.status_lbl.setText)
        self._worker.step_changed.connect(self.step_progress.advance_step)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: dict):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.step_progress.set_done()
        self._result = result
        self._populate_results(result)
        self.export_btn.setEnabled(True)
        fname = os.path.basename(result.get("path", ""))
        lat   = result.get("latency_mean_ms", 0) or 0
        fps   = result.get("throughput_fps",  0) or 0
        dev   = result.get("device_used", "").upper()
        self.status_lbl.setText(
            f"✅  {fname}  [{dev}]"
            f"  |  Params: {fmt_params(result.get('total_params', 0))}"
            f"  |  Latency: {lat:.2f} ms"
            f"  |  FPS: {fps:.1f}")

    def _on_error(self, msg: str):
        self.run_btn.setEnabled(bool(self._pending_weights))
        self.stop_btn.setEnabled(False)
        first_line = msg.split("\n")[0]
        self.step_progress.set_error(first_line)
        self.status_lbl.setText(f"❌ {first_line}")
        QMessageBox.critical(self, "Lỗi phân tích", msg[:1200])

    # ══════════════════════════════════════════════════════════════════════
    # Populate results
    # ══════════════════════════════════════════════════════════════════════

    def _clear_overview(self):
        for card in (self.card_params, self.card_flops, self.card_size,
                     self.card_filesize, self.card_fps, self.card_latency):
            card.set_value("—", "")
        self.meta_table.setRowCount(0)
        self.layers_table.setRowCount(0)
        self.layers_count.setText("")
        for b in (self.bar_mean, self.bar_min, self.bar_max,
                  self.bar_p50, self.bar_p95):
            b.set_value(0)
        self.lat_info_table.setRowCount(0)

    def _populate_results(self, r: dict):
        # ── Stat cards ──────────────────────────────────────────────────
        tp = r.get("total_params", 0)
        self.card_params.set_value(
            fmt_params(tp),
            "thop" if "params_thop" in r else "state_dict",
            DARK["accent2"])

        flops = r.get("flops")
        if flops:
            self.card_flops.set_value(
                r.get("flops_str", fmt_flops(flops)), "MACs", DARK["green"])
        elif r.get("total_nodes"):
            self.card_flops.set_value(
                str(r["total_nodes"]), "ONNX nodes", DARK["yellow"])
        else:
            self.card_flops.set_value(
                "N/A",
                "pip install thop"
                if r.get("format", "").startswith("PyTorch") else "")

        self.card_size.set_value(f"{r.get('model_size_mb', 0):.2f}", "MB")
        self.card_filesize.set_value(f"{r.get('file_size_mb', 0):.2f}", "MB")

        fps = r.get("throughput_fps", 0) or 0
        self.card_fps.set_value(
            f"{fps:.1f}" if fps else "N/A", "frames/sec",
            DARK["green"] if fps > 30 else DARK["yellow"])

        lat = r.get("latency_mean_ms", 0) or 0
        self.card_latency.set_value(
            f"{lat:.2f}" if lat else "N/A", "milliseconds",
            DARK["green"]  if lat and lat < 20  else
            DARK["yellow"] if lat and lat < 100 else DARK["red"])

        # ── Metadata ────────────────────────────────────────────────────
        rows: list[tuple[str, str]] = [
            ("Format",          r.get("format", "?")),
            ("File",            os.path.basename(r.get("path", ""))),
            ("Device",          r.get("device_used", "").upper()),
            ("Analyzed at",     r.get("analyzed_at", "")),
            ("Checkpoint type", r.get("ck_type", r.get("producer", ""))),
        ]
        if r.get("missing_keys_count"):
            n      = r["missing_keys_count"]
            sample = r.get("missing_keys_sample", [])
            rows.append((
                "Missing keys",
                f"{n}  (e.g. {sample[:2]})" if sample else str(n)))
        if r.get("ir_version"):
            rows.append(("ONNX IR version", str(r["ir_version"])))
        if r.get("opset"):
            rows.append(("Opset", ", ".join(r["opset"])))
        if r.get("producer"):
            rows.append((
                "Producer",
                f"{r['producer']} {r.get('producer_version', '')}"))
        for idx, inp in enumerate(r.get("inputs", [])):
            rows.append((
                f"Input[{idx}]",
                f"{inp.get('name', '')}  shape={inp.get('shape', '')}"))
        for idx, out in enumerate(r.get("outputs", [])):
            rows.append((
                f"Output[{idx}]",
                f"{out.get('name', '')}  shape={out.get('shape', '')}"))
        if r.get("input_size"):
            rows.append(("Input size",
                         f"{r['input_size']}×{r['input_size']}"))
        for k, v in (r.get("meta") or {}).items():
            if v is not None:
                rows.append((str(k), str(v)[:120]))
        for dtype, cnt in (r.get("dtypes") or {}).items():
            rows.append((f"dtype  {dtype}", f"{cnt} tensors"))
        if r.get("gpu_mem_peak_mb"):
            rows.append(("GPU mem peak",
                         f"{r['gpu_mem_peak_mb']:.1f} MB"))
        if r.get("ort_provider_used"):
            rows.append(("ORT provider", r["ort_provider_used"]))
        if r.get("op_summary"):
            top = list(r["op_summary"].items())[:10]
            rows.append((
                "Top ops",
                ", ".join(f"{k}:{v}" for k, v in top)))
        if r.get("profile_error"):
            rows.append(("⚠ Profile error", r["profile_error"][:200]))
        if r.get("flops_err"):
            rows.append(("⚠ FLOPs error",   r["flops_err"][:200]))
        self._fill_kv_table(self.meta_table, rows)

        # ── Layers ──────────────────────────────────────────────────────
        layers = r.get("layers", [])
        self.layers_table.setRowCount(0)
        self.layers_count.setText(f"{len(layers)} entries")
        for i, layer in enumerate(layers):
            self.layers_table.insertRow(i)
            self.layers_table.setItem(
                i, 0, QTableWidgetItem(str(layer.get("name", ""))))
            self.layers_table.setItem(
                i, 1, QTableWidgetItem(str(layer.get("shape", ""))))
            item_p = QTableWidgetItem(fmt_params(layer.get("params", 0)))
            item_p.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.layers_table.setItem(i, 2, item_p)
            self.layers_table.setItem(
                i, 3, QTableWidgetItem(str(layer.get("dtype", "—"))))

        # ── Latency bars ────────────────────────────────────────────────
        mean = r.get("latency_mean_ms", 0) or 0
        mn   = r.get("latency_min_ms",  0) or 0
        mx   = r.get("latency_max_ms",  0) or 0
        p50  = r.get("latency_p50_ms",  0) or 0
        p95  = r.get("latency_p95_ms",  0) or 0
        ref  = mx or 1
        self.bar_mean.set_value(mean, mean / ref)
        self.bar_min.set_value(mn,   mn   / ref)
        self.bar_max.set_value(mx,   1.0)
        self.bar_p50.set_value(p50,  p50  / ref)
        self.bar_p95.set_value(p95,  p95  / ref)

        lat_info: list[tuple[str, str]] = []
        if r.get("benchmark_runs"):
            lat_info.append(("Runs",   str(r["benchmark_runs"])))
            lat_info.append(("Warmup", str(r["benchmark_warmup"])))
        if r.get("ort_provider_used"):
            lat_info.append(("ORT Provider", r["ort_provider_used"]))
        lat_info.append(("Device",     r.get("device_used", "").upper()))
        lat_info.append(("Throughput", f"{fps:.1f} FPS"))
        if r.get("gpu_mem_peak_mb"):
            lat_info.append((
                "GPU mem peak",
                f"{r['gpu_mem_peak_mb']:.1f} MB"))
        if r.get("benchmark_error"):
            lat_info.append(("⚠ Error",     r["benchmark_error"][:200]))
        if r.get("ort_error"):
            lat_info.append(("⚠ ORT error", r["ort_error"][:200]))
        self._fill_kv_table(self.lat_info_table, lat_info)

        # ── Raw JSON ────────────────────────────────────────────────────
        self.tab_raw.setPlainText(
            json.dumps(r, indent=2, default=str, ensure_ascii=False))
        self.tabs.setCurrentIndex(0)

    def _fill_kv_table(
        self,
        table: QTableWidget,
        rows: list[tuple[str, str]],
    ):
        table.setRowCount(0)
        for key, val in rows:
            i = table.rowCount()
            table.insertRow(i)
            ki = QTableWidgetItem(str(key))
            ki.setForeground(QColor(DARK["muted"]))
            table.setItem(i, 0, ki)
            table.setItem(i, 1, QTableWidgetItem(str(val)))

    # ══════════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════════

    def _export_json(self):
        if not self._result:
            return
        default = (
            os.path.splitext(
                os.path.basename(self._result.get("path", "result"))
            )[0] + "_profile.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export JSON", default, "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    self._result, f,
                    indent=2, default=str, ensure_ascii=False)
            self.status_lbl.setText(f"✓ Exported → {path}")