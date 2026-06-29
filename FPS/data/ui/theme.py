"""
Theme tập trung: màu sắc, stylesheet, palette.
Chỉnh sửa DARK dict hoặc STYLESHEET để thay đổi toàn bộ giao diện.
"""
from PyQt5.QtWidgets import QStyleFactory
from PyQt5.QtGui import QColor, QPalette

# ── Bảng màu ─────────────────────────────────────────────────────────────────
# Thay đổi bất kỳ giá trị nào ở đây để đổi toàn bộ giao diện.
DARK: dict[str, str] = {
    "bg":      "#0d0d17",   # nền ngoài cùng
    "surface": "#13131f",   # nền panel
    "card":    "#1a1a2e",   # card thường
    "card2":   "#1f1f35",   # card hover / input
    "border":  "#2a2a45",   # viền
    "accent":  "#6c63ff",   # tím chính (button primary, tab active)
    "accent2": "#a78bfa",   # tím nhạt (giá trị nổi bật)
    "accent3": "#38bdf8",   # xanh dương (script badge)
    "green":   "#34d399",   # thành công / done
    "yellow":  "#fbbf24",   # cảnh báo
    "red":     "#f87171",   # lỗi / stop
    "text":    "#e8e6f5",   # chữ chính
    "muted":   "#7b78a0",   # chữ phụ
    "muted2":  "#4a4870",   # chữ rất mờ / disabled
}


def apply_dark_palette(app) -> None:
    """Áp dụng QPalette dark cho toàn bộ app."""
    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(DARK["bg"]))
    pal.setColor(QPalette.WindowText,      QColor(DARK["text"]))
    pal.setColor(QPalette.Base,            QColor(DARK["surface"]))
    pal.setColor(QPalette.AlternateBase,   QColor(DARK["card"]))
    pal.setColor(QPalette.Text,            QColor(DARK["text"]))
    pal.setColor(QPalette.Button,          QColor(DARK["card"]))
    pal.setColor(QPalette.ButtonText,      QColor(DARK["text"]))
    pal.setColor(QPalette.Highlight,       QColor(DARK["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.Link,            QColor(DARK["accent2"]))
    app.setPalette(pal)


# ── Stylesheet ────────────────────────────────────────────────────────────────
# Mỗi block có comment rõ ràng để dễ chỉnh từng phần.
STYLESHEET = f"""
/* ── Base ── */
QMainWindow, QWidget {{
    background: {DARK['bg']};
    color: {DARK['text']};
    font-family: 'Segoe UI', 'Inter', 'SF Pro Text', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}}
QLabel {{ background: transparent; }}

/* ── Buttons ── */
QPushButton {{
    background: {DARK['card2']};
    color: {DARK['text']};
    border: 1px solid {DARK['border']};
    border-radius: 7px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {DARK['accent']};
    border-color: {DARK['accent']};
    color: white;
}}
QPushButton:pressed   {{ background: #5a52e0; }}
QPushButton:disabled  {{
    background: {DARK['card']};
    color: {DARK['muted2']};
    border-color: {DARK['muted2']};
}}

/* ── Button: primary (Run) ── */
QPushButton#primary {{
    background: {DARK['accent']};
    border-color: {DARK['accent']};
    color: white;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 28px;
    border-radius: 8px;
}}
QPushButton#primary:hover    {{ background: {DARK['accent2']}; border-color: {DARK['accent2']}; }}
QPushButton#primary:disabled {{
    background: {DARK['muted2']};
    border-color: {DARK['muted2']};
    color: {DARK['muted']};
}}

/* ── Button: stop ── */
QPushButton#stop_btn {{
    background: {DARK['card2']};
    color: {DARK['red']};
    border: 1px solid {DARK['red']};
    border-radius: 8px;
    font-size: 13px;
    font-weight: 700;
    padding: 10px 20px;
}}
QPushButton#stop_btn:hover {{
    background: {DARK['red']};
    color: white;
}}

/* ── Tabs ── */
QTabWidget::pane {{
    border: 1px solid {DARK['border']};
    border-radius: 8px;
    background: {DARK['card']};
}}
QTabBar::tab {{
    background: {DARK['surface']};
    color: {DARK['muted']};
    border: 1px solid {DARK['border']};
    border-bottom: none;
    padding: 8px 18px;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    margin-right: 3px;
}}
QTabBar::tab:selected {{
    background: {DARK['card']};
    color: {DARK['text']};
    font-weight: 700;
    border-bottom: 2px solid {DARK['accent']};
}}
QTabBar::tab:hover:!selected {{
    color: {DARK['text']};
    background: {DARK['card2']};
}}

/* ── Table ── */
QTableWidget {{
    background: {DARK['surface']};
    border: none;
    gridline-color: {DARK['border']};
    border-radius: 6px;
    selection-background-color: {DARK['accent']};
}}
QTableWidget::item          {{ padding: 5px 8px; }}
QTableWidget::item:alternate {{ background: {DARK['card']}; }}
QHeaderView::section {{
    background: {DARK['card2']};
    color: {DARK['muted']};
    border: none;
    border-bottom: 1px solid {DARK['border']};
    padding: 7px 10px;
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* ── ScrollArea & ScrollBar ── */
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{
    background: {DARK['surface']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {DARK['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{ height: 0px; }}

/* ── TextEdit (log / raw JSON) ── */
QTextEdit {{
    background: {DARK['surface']};
    border: 1px solid {DARK['border']};
    border-radius: 7px;
    color: {DARK['text']};
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 8px;
}}

/* ── ProgressBar (main) ── */
QProgressBar {{
    background: {DARK['surface']};
    border: 1px solid {DARK['border']};
    border-radius: 5px;
    height: 8px;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {DARK['accent']}, stop:1 {DARK['accent2']});
    border-radius: 5px;
}}

/* ── ProgressBar: step indicator (xanh lá) ── */
QProgressBar#step_bar {{
    height: 6px;
    border-radius: 3px;
}}
QProgressBar#step_bar::chunk {{
    background: {DARK['green']};
    border-radius: 3px;
}}
"""
