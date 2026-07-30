"""Single source of truth for the app's visual styling.

The stylesheet is applied once to the ``QApplication`` (see ``app.main``)
rather than per-widget, so every screen *and* every popup dialog
(``QDialog``, ``QMessageBox``, combo box dropdowns) inherits the same look.
Widgets that need a one-off treatment opt in via ``setObjectName`` and the
matching ``#objectName`` selector below.
"""

# ---------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------
# Surfaces, from furthest back to closest to the user.
BACKGROUND = "#22262b"
SURFACE = "#2b3037"
SURFACE_RAISED = "#333941"
BORDER = "#3f4650"
BORDER_STRONG = "#4d5560"

# Text.
TEXT = "#eef1f5"
TEXT_MUTED = "#a7b0bb"

# Accent (primary actions, focus rings, selection).
ACCENT = "#3d90e3"
ACCENT_HOVER = "#4f9de8"
ACCENT_PRESSED = "#2f7bc7"

# Semantic.
DANGER = "#d9534f"
DANGER_HOVER = "#e0645f"

# The plot legend sits on a light card so it reads against the Plotly
# figure it annotates rather than against the dark chrome.
LEGEND_CARD_BG = "#f0f0f0"
LEGEND_CARD_TEXT = "#2e2e2e"

FONT_FAMILY = "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"


STYLESHEET = f"""
/* ---------- Base ---------- */
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: {FONT_FAMILY};
    font-size: 14px;
}}

QDialog, QMessageBox {{
    background-color: {SURFACE};
}}

QToolTip {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    padding: 6px 8px;
}}

/* ---------- Labels ---------- */
QLabel {{
    background: transparent;
    color: {TEXT};
}}

QLabel#titleLabel {{
    font-size: 26px;
    font-weight: 600;
    padding: 18px 10px 4px 10px;
}}

QLabel#subtitleLabel {{
    font-size: 15px;
    color: {TEXT_MUTED};
    padding: 4px 10px 18px 10px;
}}

QLabel#loadingLabel {{
    font-size: 16px;
    color: {TEXT_MUTED};
    padding: 20px;
}}

/* ---------- Buttons ---------- */
QPushButton {{
    background-color: {ACCENT};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 9px 20px;
    font-size: 14px;
    font-weight: 500;
    min-width: 90px;
}}

QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}

QPushButton:pressed {{
    background-color: {ACCENT_PRESSED};
}}

QPushButton:disabled {{
    background-color: {SURFACE_RAISED};
    color: {TEXT_MUTED};
}}

/* Secondary buttons (Back / Cancel) recede next to primary actions. */
QPushButton#secondaryButton {{
    background-color: transparent;
    color: {TEXT_MUTED};
    border: 1px solid {BORDER_STRONG};
}}

QPushButton#secondaryButton:hover {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
}}

QPushButton#dangerButton {{
    background-color: {DANGER};
}}

QPushButton#dangerButton:hover {{
    background-color: {DANGER_HOVER};
}}

/* Large call-to-action on the opening screen. */
QPushButton#ctaButton {{
    font-size: 16px;
    padding: 13px 34px;
    min-width: 240px;
}}

/* ---------- Inputs ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 10px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QComboBox:focus, QSpinBox:focus {{
    border: 1px solid {ACCENT};
}}

QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled {{
    color: {TEXT_MUTED};
    background-color: {BACKGROUND};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    margin-right: 8px;
}}

QComboBox QAbstractItemView {{
    background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER_STRONG};
    border-radius: 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: none;
    padding: 4px;
}}

/* Read-only patient summary panel on the plots screen. */
QTextEdit#summaryPanel {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
}}

/* ---------- Checkboxes ---------- */
QCheckBox {{
    background: transparent;
    spacing: 8px;
    padding: 3px 0;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 4px;
    background-color: {SURFACE};
}}

QCheckBox::indicator:hover {{
    border: 1px solid {ACCENT};
}}

QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
}}

/* ---------- Group boxes ---------- */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 16px 14px 12px 14px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_MUTED};
    font-size: 12px;
    text-transform: uppercase;
}}

/* ---------- Tables ---------- */
QTableWidget, QTableView {{
    background-color: {SURFACE};
    alternate-background-color: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}

QTableWidget::item, QTableView::item {{
    padding: 7px 8px;
    border: none;
}}

QHeaderView::section {{
    background-color: {SURFACE_RAISED};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER_STRONG};
    padding: 9px 8px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}}

QTableCornerButton::section {{
    background-color: {SURFACE_RAISED};
    border: none;
}}

/* ---------- Toolbar ---------- */
QToolBar {{
    background-color: {SURFACE};
    border: none;
    border-top: 1px solid {BORDER};
    padding: 6px;
    spacing: 4px;
}}

QToolButton {{
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 6px;
}}

QToolButton:hover {{
    background-color: {SURFACE_RAISED};
}}

/* ---------- Progress bar ---------- */
QProgressBar {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 10px;
    text-align: center;
    color: transparent;
}}

QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 5px;
}}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 11px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-height: 28px;
}}

QScrollBar::handle:vertical:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 11px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 5px;
    min-width: 28px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {TEXT_MUTED};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}

QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ---------- Graphics view (plot legend card) ---------- */
QGraphicsView#legendView {{
    background-color: {LEGEND_CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
"""
