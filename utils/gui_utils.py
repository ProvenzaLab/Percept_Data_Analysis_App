from datetime import datetime
import pandas as pd
import plotly.io as pio
import tempfile
from PySide6.QtWidgets import QFileDialog

DATE_FORMAT = "%Y-%m-%d"


def add_extension(filename, extension):
    return (
        filename
        if filename.lower().endswith(extension.lower())
        else filename + extension
    )


def save_raw_data(data, filename):
    ext = filename.split(".")[-1].lower()
    if ext == "json":
        data.to_json(filename, orient="records", lines=True)
    elif ext == "xlsx":
        data.to_excel(filename, index=False)
    elif ext == "tsv":
        data.to_csv(filename, sep="\t", index=False)
    elif ext == "txt":
        with open(filename, "w") as f:
            f.write(data.to_string(index=False))
    else:
        filename = add_extension(filename, ".csv")
        data.to_csv(filename, index=False)


def save_lin_ar_feature(df, filename, param_dict):
    model = param_dict["model"]
    # ``param_dict["hemisphere"]`` is a string ("left"/"right") written by
    # the settings menu. Treat anything that is not "left" as the right
    # hemisphere so unknown/legacy values fall back to "right".
    hemisphere = "left" if str(param_dict.get("hemisphere", "")).lower() == "left" else "right"
    data = df.groupby("days_since_dbs").head(1)[
        [f"lfp_{hemisphere}_day_r2_{model}", "days_since_dbs"]
    ]
    data["date"] = pd.to_datetime(
        df.groupby("days_since_dbs").head(1)["timestamp"]
    ).dt.date

    ext = filename.split(".")[-1].lower()

    if ext == "json":
        data.to_json(filename, orient="records", lines=True)
    elif ext == "xlsx":
        data.to_excel(filename, index=False)
    elif ext == "tsv":
        data.to_csv(filename, sep="\t", index=False)
    elif ext == "txt":
        with open(filename, "w") as f:
            f.write(data.to_string(index=False))
    else:
        filename = add_extension(filename, ".csv")
        data.to_csv(filename, index=False)


def save_plot(fig, filename):
    ext = filename.split(".")[-1].lower()

    if ext in ["png", "jpg", "jpeg", "webp", "svg", "pdf"]:
        # The on-screen figure is autosize=True (responsive to the window);
        # a static raster/vector export has no such notion, so give it an
        # explicit size rather than falling back to kaleido's small default.
        pio.write_image(fig, filename, format=ext, width=1400, height=1000)
    else:
        filename = add_extension(filename, ".html")
        fig.write_html(
            filename,
            include_plotlyjs="cdn",
            config={"responsive": True},
            default_width="100%",
            default_height="100%",
        )


def open_file_dialog(parent):
    file_dialog = QFileDialog()
    return file_dialog.getOpenFileNames(
        parent, "Select patient JSON files", "", "JSON files (*.json)"
    )[0]


def open_save_dialog(parent, title, default_filter):
    file_path, _ = QFileDialog.getSaveFileName(parent, title, "", default_filter)
    return file_path


def create_temp_plot(fig):
    """Write ``fig`` to a temp HTML file that resizes with its container.

    ``fig.to_html(full_html=True)``'s default template doesn't set a height
    on ``<html>``/``<body>``, so ``default_height="100%"`` on the plot div
    has nothing to size itself against. Build the wrapper document manually
    instead, and pass ``config={"responsive": True}`` so Plotly re-lays-out
    the figure whenever the embedding QWebEngineView is resized.
    """
    plot_div = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True},
        default_width="100%",
        default_height="100%",
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<style>html, body { margin: 0; height: 100%; }</style>"
        f"</head><body>{plot_div}</body></html>"
    )
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8"
    ) as temp_file:
        temp_file.write(html)
        return temp_file.name


def validate_date(date_str):
    try:
        datetime.strptime(date_str, DATE_FORMAT)
        return True
    except ValueError:
        return False


def select_folder():
    folder_path = QFileDialog.getExistingDirectory(
        None, "Select parent directory containing patient JSON files"
    )
    return folder_path
