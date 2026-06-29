"""
Các widget tái sử dụng:
  - DropZone         : kéo thả file (weights + script)
  - StatCard         : card hiển thị 1 chỉ số lớn
  - LatencyBar       : dải latency có progress bar mini
  - StepProgressWidget: thanh tiến trình theo bước với spinner
"""
from __future__ import annotations
import os

from PyQt5.QtWidgets import (
    QFrame, QLabel, QVBoxLayout, QHBoxLayout,
    QProgressBar, QFileDialog,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

from .theme import DARK

# Các bước phân tích — (label, pct_target)
STEPS: list[tuple[str, int]] = [
    ("🔍 Đọc file",            5),
    ("⚙️  Phân tích cấu trúc", 30),
    ("📐 Tính FLOPs",          55),
    ("⏱️  Benchmark latency",  70),
    ("✅ Hoàn tất",            100),
]


# ══════════════════════════════════════════════════════════════════════════════
# DropZone
# ══════════════════════════════════════════════════════════════════════════════
class DropZone(QFrame):
    """
    Kéo thả 1 hoặc 2 file:
      - weights (.pt / .pth / .onnx)  → slot weights
      - script  (.py)                 → slot script
    Click → dialog chọn weights.
    Signal: files_dropped(weights_path_or_None, script_path_or_None)
    """
    files_dropped = pyqtSignal(object, object)

    WEIGHT_EXTS = {".pt", ".pth", ".onnx"}
    SCRIPT_EXTS = {".py"}
    ALL_EXTS    = WEIGHT_EXTS | SCRIPT_EXTS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(190)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(20, 24, 20, 24)
        lay.setSpacing(12)

        self.icon_lbl = QLabel("⬇")
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.icon_lbl.setStyleSheet(
            f"font-size: 46px; color: {DARK['accent']}; background: transparent;")

        self.title_lbl = QLabel("Kéo file vào đây")
        self.title_lbl.setAlignment(Qt.AlignCenter)
        self.title_lbl.setStyleSheet(
            f"font-size: 20px; font-weight: 800; color: {DARK['text']};"
            " background: transparent;")

        self.sub_lbl = QLabel(".pt / .pth / .onnx  +  model.py (tuỳ chọn)")
        self.sub_lbl.setAlignment(Qt.AlignCenter)
        self.sub_lbl.setStyleSheet(
            f"font-size: 14px; color: {DARK['muted']}; background: transparent;")

        lay.addWidget(self.icon_lbl)
        lay.addWidget(self.title_lbl)
        lay.addWidget(self.sub_lbl)
        self._set_style(hover=False)

    def _set_style(self, hover: bool):
        bc = DARK["accent"] if hover else DARK["border"]
        bg = DARK["card2"]  if hover else DARK["card"]
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: 3px dashed {bc};
                border-radius: 18px;
            }}
        """)

    # ── drag / drop events ────────────────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in e.mimeData().urls()]
            if any(os.path.splitext(p)[1].lower() in self.ALL_EXTS for p in paths):
                e.acceptProposedAction()
                self._set_style(hover=True)
                return
        e.ignore()

    def dragLeaveEvent(self, _e):
        self._set_style(hover=False)

    def dropEvent(self, e: QDropEvent):
        self._set_style(hover=False)
        weights = script = None
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            ext  = os.path.splitext(path)[1].lower()
            if ext in self.WEIGHT_EXTS and weights is None:
                weights = path
            elif ext in self.SCRIPT_EXTS and script is None:
                script = path
        if weights or script:
            self.files_dropped.emit(weights, script)

    def mousePressEvent(self, _e):
        path, _ = QFileDialog.getOpenFileName(
            self, "Chọn file model weights", "",
            "Model weights (*.pt *.pth *.onnx);;All (*)")
        if path:
            self.files_dropped.emit(path, None)


# ══════════════════════════════════════════════════════════════════════════════
# StatCard
# ══════════════════════════════════════════════════════════════════════════════
class StatCard(QFrame):
    """Card hiển thị 1 chỉ số: tiêu đề nhỏ + số lớn + phụ đề."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(118)
        self.setStyleSheet(f"""
            QFrame {{
                background: {DARK['card']};
                border: 1px solid {DARK['border']};
                border-radius: 14px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(6)

        self._title = QLabel(label.upper())
        self._title.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 11px; font-weight: 700;"
            " letter-spacing: 1px;")

        self._val = QLabel("—")
        self._val.setStyleSheet(
            f"color: {DARK['text']}; font-size: 30px; font-weight: 800;")
        self._val.setWordWrap(True)

        self._sub = QLabel("")
        self._sub.setStyleSheet(f"color: {DARK['muted']}; font-size: 12px;")

        lay.addWidget(self._title)
        lay.addWidget(self._val)
        lay.addWidget(self._sub)

    def set_value(self, val: str, sub: str = "", color: str | None = None):
        self._val.setText(str(val))
        self._sub.setText(sub)
        clr = color or DARK["text"]
        self._val.setStyleSheet(
            f"color: {clr}; font-size: 30px; font-weight: 800;")


# ══════════════════════════════════════════════════════════════════════════════
# LatencyBar
# ══════════════════════════════════════════════════════════════════════════════
class LatencyBar(QFrame):
    """Dải hiển thị 1 giá trị latency kèm progress bar mini."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(56)
        self.setStyleSheet(f"""
            QFrame {{
                background: {DARK['card2']};
                border: 1px solid {DARK['border']};
                border-radius: 12px;
            }}
        """)
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 14, 20, 14)
        row.setSpacing(14)

        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 13px; font-weight: 600;"
            " min-width: 110px;")

        self._val = QLabel("—")
        self._val.setStyleSheet(
            f"color: {DARK['accent2']}; font-size: 22px; font-weight: 800;")

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setFixedHeight(10)
        self._bar.setTextVisible(False)

        row.addWidget(self._lbl)
        row.addWidget(self._val)
        row.addStretch()
        row.addWidget(self._bar, stretch=3)

    def set_value(self, ms: float, relative: float = 0.5):
        self._val.setText(f"{ms:.2f} ms")
        self._bar.setValue(int(relative * 100))


# ══════════════════════════════════════════════════════════════════════════════
# StepProgressWidget
# ══════════════════════════════════════════════════════════════════════════════
class StepProgressWidget(QFrame):
    """
    Thanh tiến trình theo bước với dot indicator + spinner braille.
    Trạng thái: idle | running | done | error
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background: {DARK['card']};
                border: 1px solid {DARK['border']};
                border-radius: 14px;
            }}
        """)
        self._dots_tick = 0
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(10)

        # top row
        top = QHBoxLayout()
        self._status = QLabel("Kéo file vào ô trên hoặc nhấn  ▶ Run")
        self._status.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 14px; font-weight: 600;")
        self._spinner = QLabel("")
        self._spinner.setStyleSheet(
            f"color: {DARK['accent2']}; font-size: 16px; min-width: 34px;")
        self._pct = QLabel("")
        self._pct.setStyleSheet(f"color: {DARK['muted']}; font-size: 13px;")
        top.addWidget(self._status)
        top.addWidget(self._spinner)
        top.addStretch()
        top.addWidget(self._pct)
        outer.addLayout(top)

        # progress bar
        self._bar = QProgressBar()
        self._bar.setObjectName("step_bar")
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(12)
        outer.addWidget(self._bar)

        # dot row
        dot_row = QHBoxLayout()
        dot_row.setSpacing(0)
        self._dots: list[QLabel | None]  = []
        self._dot_texts: list[QLabel | None] = []

        for i, (label, _) in enumerate(STEPS):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignHCenter)

            dot = QLabel("○")
            dot.setAlignment(Qt.AlignCenter)
            dot.setStyleSheet(f"color: {DARK['muted2']}; font-size: 18px;")
            dot.setFixedWidth(28)

            # strip emoji prefix, truncate
            short = label.split(" ", 1)[-1][:14]
            txt = QLabel(short)
            txt.setAlignment(Qt.AlignCenter)
            txt.setStyleSheet(
                f"color: {DARK['muted2']}; font-size: 11px; max-width: 90px;")
            txt.setWordWrap(True)

            col.addWidget(dot)
            col.addWidget(txt)
            self._dots.append(dot)
            self._dot_texts.append(txt)

            dot_row.addLayout(col)
            if i < len(STEPS) - 1:
                sep = QLabel("─────")
                sep.setAlignment(Qt.AlignCenter)
                sep.setStyleSheet(f"color: {DARK['muted2']}; font-size: 12px;")
                dot_row.addWidget(sep, stretch=1)
                self._dots.append(None)
                self._dot_texts.append(None)

        outer.addLayout(dot_row)

        # spinner timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    # ── public API ────────────────────────────────────────────────────────
    def set_idle(self):
        self._timer.stop()
        self._spinner.setText("")
        self._status.setText("Kéo file vào ô trên hoặc nhấn  ▶ Run")
        self._status.setStyleSheet(
            f"color: {DARK['muted']}; font-size: 14px; font-weight: 600;")
        self._bar.setValue(0)
        self._pct.setText("")
        self._reset_dots()

    def set_running(self):
        self._reset_dots()
        self._bar.setValue(0)
        self._timer.start(350)

    def advance_step(self, step_idx: int, label: str):
        pct = STEPS[step_idx][1]
        self._bar.setValue(pct)
        self._pct.setText(f"{pct}%")
        self._status.setText(label)
        self._status.setStyleSheet(
            f"color: {DARK['text']}; font-size: 14px; font-weight: 600;")

        real = 0
        for i, dot in enumerate(self._dots):
            if dot is None:
                continue
            txt = self._dot_texts[i]
            if real < step_idx:
                dot.setText("●")
                dot.setStyleSheet(f"color: {DARK['green']}; font-size: 18px;")
                if txt:
                    txt.setStyleSheet(f"color: {DARK['green']}; font-size: 11px;")
            elif real == step_idx:
                dot.setText("◉")
                dot.setStyleSheet(f"color: {DARK['accent2']}; font-size: 18px;")
                if txt:
                    txt.setStyleSheet(
                        f"color: {DARK['accent2']}; font-size: 11px; font-weight: 700;")
            else:
                dot.setText("○")
                dot.setStyleSheet(f"color: {DARK['muted2']}; font-size: 18px;")
                if txt:
                    txt.setStyleSheet(f"color: {DARK['muted2']}; font-size: 11px;")
            real += 1

    def set_done(self):
        self._timer.stop()
        self._spinner.setText("")
        self._bar.setValue(100)
        self._pct.setText("100%")
        self._status.setText("✅ Phân tích hoàn tất!")
        self._status.setStyleSheet(
            f"color: {DARK['green']}; font-size: 14px; font-weight: 700;")
        for dot in self._dots:
            if dot is not None:
                dot.setText("●")
                dot.setStyleSheet(f"color: {DARK['green']}; font-size: 18px;")

    def set_error(self, msg: str = ""):
        self._timer.stop()
        self._spinner.setText("")
        self._bar.setValue(0)
        self._pct.setText("")
        self._status.setText(f"❌ {msg[:60]}")
        self._status.setStyleSheet(
            f"color: {DARK['red']}; font-size: 14px; font-weight: 700;")

    # ── internal ─────────────────────────────────────────────────────────
    def _reset_dots(self):
        for i, dot in enumerate(self._dots):
            if dot is None:
                continue
            dot.setText("○")
            dot.setStyleSheet(f"color: {DARK['muted2']}; font-size: 18px;")
            txt = self._dot_texts[i]
            if txt:
                txt.setStyleSheet(f"color: {DARK['muted2']}; font-size: 11px;")

    def _tick(self):
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._dots_tick = (self._dots_tick + 1) % len(frames)
        self._spinner.setText(frames[self._dots_tick])