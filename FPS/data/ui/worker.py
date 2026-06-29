"""
Worker thread: chạy core.analyzer.analyze() trong background,
emit signals để UI cập nhật thanh tiến trình.
"""
from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from ..core.analyzer import analyze
from .widgets import STEPS


class AnalyzeWorker(QThread):
    """
    Signals:
      progress(str)         — message text hiển thị ở status bar
      step_changed(int,str) — (step_index, label) để StepProgressWidget cập nhật
      done(dict)            — kết quả phân tích
      error(str)            — traceback đầy đủ
    """
    progress     = pyqtSignal(str)
    step_changed = pyqtSignal(int, str)
    done         = pyqtSignal(dict)
    error        = pyqtSignal(str)

    def __init__(
        self,
        weights_path: str,
        device_str:   str,
        warmup:       int | None = None,
        runs:         int | None = None,
    ):
        super().__init__()
        self.weights_path = weights_path
        self.device_str   = device_str
        self.warmup       = warmup
        self.runs         = runs

    def _step(self, idx: int):
        label = STEPS[idx][0]
        self.step_changed.emit(idx, label)
        self.progress.emit(label)

    def run(self):
        import traceback
        try:
            self._step(0)   # Đọc file
            self._step(1)   # Phân tích cấu trúc
            self._step(2)   # Tính FLOPs

            result = analyze(
                weights_path=self.weights_path,
                device_str=self.device_str,
                warmup=self.warmup,
                runs=self.runs,
            )

            self._step(3)   # Benchmark latency (đã xong trong analyze)
            self._step(4)   # Hoàn tất
            self.done.emit(result)

        except Exception as ex:
            self.error.emit(f"{ex}\n\n{traceback.format_exc()}")