import html
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


QUALITY_METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "precision_at_1",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "mean_confidence",
]


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | int | str | None) -> str:
    number = _safe_float(value)
    if number is None:
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    return f"{number:.3f}"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 12,
    fill: str = "#172033",
    anchor: str = "start",
    weight: str = "400",
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}" {extra}>{html.escape(str(text))}</text>'
    )


def _color_scale(value: float, max_value: float) -> str:
    ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
    r = int(239 - ratio * 183)
    g = int(246 - ratio * 118)
    b = int(255 - ratio * 60)
    return f"#{r:02x}{g:02x}{b:02x}"


def _bar_color(value: float) -> str:
    if value >= 0.85:
        return "#188a5a"
    if value >= 0.7:
        return "#2f7dd1"
    if value >= 0.5:
        return "#d88a21"
    return "#c94949"


def _metrics_bar_svg(metrics: Dict[str, float], title: str) -> str:
    rows = [
        (key, float(metrics[key]))
        for key in QUALITY_METRICS
        if key in metrics and _safe_float(metrics[key]) is not None
    ]
    rows = [(key, value) for key, value in rows if 0.0 <= value <= 1.0]
    width = 920
    row_h = 34
    top = 70
    left = 240
    chart_w = 560
    height = top + row_h * max(1, len(rows)) + 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(28, 34, title, 22, "#172033", weight="700"),
        _svg_text(left, 58, "0", 11, "#667085", anchor="middle"),
        _svg_text(left + chart_w / 2, 58, "0.5", 11, "#667085", anchor="middle"),
        _svg_text(left + chart_w, 58, "1.0", 11, "#667085", anchor="middle"),
        f'<line x1="{left}" y1="64" x2="{left + chart_w}" y2="64" stroke="#d7dde8"/>',
    ]
    for idx, (key, value) in enumerate(rows):
        y = top + idx * row_h
        bar_w = chart_w * value
        parts.extend([
            _svg_text(28, y + 20, key, 13, "#344054"),
            f'<rect x="{left}" y="{y + 4}" width="{chart_w}" height="20" rx="4" fill="#eef2f7"/>',
            f'<rect x="{left}" y="{y + 4}" width="{bar_w:.1f}" height="20" rx="4" fill="{_bar_color(value)}"/>',
            _svg_text(left + chart_w + 16, y + 20, _fmt(value), 13, "#172033", weight="700"),
        ])
    if not rows:
        parts.append(_svg_text(28, top + 20, "No 0-1 metrics found.", 14, "#667085"))
    parts.append("</svg>")
    return "\n".join(parts)


def _per_class_svg(report: pd.DataFrame, top_n: int, sort_by: str) -> str:
    if sort_by in report.columns:
        data = report.sort_values(sort_by, ascending=True).tail(top_n)
    else:
        data = report.tail(top_n)
    data = data.reset_index(drop=True)
    width = 1100
    left = 260
    chart_w = 650
    row_h = 34
    top = 76
    height = top + row_h * max(1, len(data)) + 50
    colors = {"precision": "#2f7dd1", "recall": "#d88a21", "f1": "#188a5a"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(28, 34, f"Per-Class Metrics (top {len(data)} by {sort_by})", 22, "#172033", weight="700"),
    ]
    legend_x = left
    for label, color in colors.items():
        parts.append(f'<rect x="{legend_x}" y="22" width="14" height="14" rx="3" fill="{color}"/>')
        parts.append(_svg_text(legend_x + 20, 34, label, 12, "#344054"))
        legend_x += 104
    for idx, row in data.iterrows():
        y = top + idx * row_h
        cls = str(row.get("class", idx))
        support = row.get("support", "")
        parts.append(_svg_text(28, y + 20, cls[:34], 12, "#344054"))
        parts.append(_svg_text(214, y + 20, f"n={support}", 11, "#667085", anchor="end"))
        metric_w = chart_w / 3 - 10
        for metric_idx, metric in enumerate(["precision", "recall", "f1"]):
            value = max(0.0, min(1.0, float(row.get(metric, 0.0))))
            x = left + metric_idx * (metric_w + 10)
            parts.append(f'<rect x="{x:.1f}" y="{y + 6}" width="{metric_w:.1f}" height="18" rx="4" fill="#eef2f7"/>')
            parts.append(
                f'<rect x="{x:.1f}" y="{y + 6}" width="{metric_w * value:.1f}" '
                f'height="18" rx="4" fill="{colors[metric]}"/>'
            )
            parts.append(_svg_text(x + metric_w + 6, y + 20, _fmt(value), 11, "#172033"))
    parts.append("</svg>")
    return "\n".join(parts)


def _confusion_svg(confusion: pd.DataFrame, normalize: bool = False) -> str:
    values = confusion.astype(float)
    values.index = [str(x) for x in values.index.tolist()]
    values.columns = [str(x) for x in values.columns.tolist()]
    classes = values.index.tolist()
    if normalize:
        denom = values.sum(axis=1).replace(0, 1)
        values = values.div(denom, axis=0)
    n = len(classes)
    cell = 44 if n <= 20 else max(18, min(36, int(780 / max(1, n))))
    left = 220
    top = 160
    width = left + cell * n + 70
    height = top + cell * n + 90
    max_value = float(values.to_numpy().max()) if n else 0.0
    title = "Confusion Matrix (row-normalized)" if normalize else "Confusion Matrix"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(28, 34, title, 22, "#172033", weight="700"),
        _svg_text(left + cell * n / 2, 76, "Predicted class", 13, "#667085", anchor="middle"),
        _svg_text(28, top + cell * n / 2, "True class", 13, "#667085", anchor="middle", extra='transform="rotate(-90 28 {0:.1f})"'.format(top + cell * n / 2)),
    ]
    for idx, cls in enumerate(classes):
        x = left + idx * cell + cell / 2
        y = top - 12
        parts.append(_svg_text(x, y, cls[:18], 10, "#344054", anchor="end", extra=f'transform="rotate(-45 {x:.1f} {y:.1f})"'))
        parts.append(_svg_text(left - 10, top + idx * cell + cell * 0.62, cls[:26], 10, "#344054", anchor="end"))
    for r, true_cls in enumerate(classes):
        for c, pred_cls in enumerate(classes):
            value = float(values.loc[true_cls, pred_cls])
            x = left + c * cell
            y = top + r * cell
            fill = _color_scale(value, max_value)
            text_value = f"{value:.2f}" if normalize else str(int(value))
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            if cell >= 28:
                parts.append(_svg_text(x + cell / 2, y + cell * 0.58, text_value, 10, "#172033", anchor="middle", weight="700"))
    parts.append("</svg>")
    return "\n".join(parts)


def _metrics_table(metrics: Dict[str, float]) -> str:
    rows = []
    for key in sorted(metrics):
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{html.escape(_fmt(metrics[key]))}</td></tr>")
    return "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _report_table(report: pd.DataFrame) -> str:
    return report.to_html(index=False, classes="data-table", border=0, escape=True, float_format=lambda x: f"{x:.3f}")


def _copy_svg_to_html(svg_paths: Iterable[Tuple[str, Path]]) -> str:
    sections = []
    for title, path in svg_paths:
        sections.append(f"<section><h2>{html.escape(title)}</h2>{path.read_text(encoding='utf-8')}</section>")
    return "\n".join(sections)


def create_eval_charts(
    eval_dir: str | Path,
    output_dir: str | Path | None = None,
    top_n: int = 40,
    sort_by: str = "f1",
) -> List[Path]:
    """
    Create SVG charts and a compact HTML dashboard from evaluation outputs.
    """
    eval_dir = Path(eval_dir)
    output_dir = Path(output_dir) if output_dir else eval_dir / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / "test_metrics.json"
    metrics_csv_path = eval_dir / "test_metrics.csv"
    report_path = eval_dir / "classification_report.csv"
    confusion_path = eval_dir / "confusion_matrix.csv"
    missing = [str(path.name) for path in [report_path, confusion_path] if not path.exists()]
    if not metrics_path.exists() and not metrics_csv_path.exists():
        missing.append("test_metrics.json or test_metrics.csv")
    if missing:
        raise FileNotFoundError(f"Missing evaluation output file(s): {', '.join(missing)}")

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        metrics = pd.read_csv(metrics_csv_path).iloc[0].dropna().to_dict()
    report = pd.read_csv(report_path)
    confusion = pd.read_csv(confusion_path, index_col=0)

    chart_paths = {
        "metrics": output_dir / "metrics_overview.svg",
        "per_class": output_dir / "per_class_metrics.svg",
        "confusion": output_dir / "confusion_matrix.svg",
        "confusion_normalized": output_dir / "confusion_matrix_normalized.svg",
    }
    _write(chart_paths["metrics"], _metrics_bar_svg(metrics, "Overall Metrics"))
    _write(chart_paths["per_class"], _per_class_svg(report, top_n=top_n, sort_by=sort_by))
    _write(chart_paths["confusion"], _confusion_svg(confusion, normalize=False))
    _write(chart_paths["confusion_normalized"], _confusion_svg(confusion, normalize=True))

    html_path = output_dir / "dashboard.html"
    svg_sections = _copy_svg_to_html([
        ("Overall Metrics", chart_paths["metrics"]),
        ("Per-Class Metrics", chart_paths["per_class"]),
        ("Confusion Matrix", chart_paths["confusion"]),
        ("Normalized Confusion Matrix", chart_paths["confusion_normalized"]),
    ])
    dashboard = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>HOAM Evaluation Dashboard</title>
  <style>
    body {{ margin: 32px; font-family: Arial, sans-serif; color: #172033; background: #f6f8fb; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 32px; }}
    section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 1px 6px rgba(16,24,40,.08); overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e6eaf0; text-align: left; }}
    th {{ background: #eef2f7; }}
    .meta {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>HOAM Evaluation Dashboard</h1>
  <p class="meta">Source: {html.escape(str(eval_dir))}</p>
  {svg_sections}
  <section><h2>Metric Values</h2>{_metrics_table(metrics)}</section>
  <section><h2>Classification Report</h2>{_report_table(report)}</section>
</body>
</html>
"""
    _write(html_path, dashboard)
    return [*chart_paths.values(), html_path]
