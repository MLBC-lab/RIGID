"""Offline, self-contained reports generated only from recorded run artifacts."""

from __future__ import annotations

import html
from pathlib import Path

from .runs import load_screening
from .storage import write_text


def render_report(screening_path: str | Path, output: str | Path) -> None:
    record = load_screening(screening_path)

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    body = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width'><title>RIGID run report</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:0 1rem;",
        "color:#182230}h1{color:#c00022}table{border-collapse:collapse;width:100%;margin:1rem 0}",
        "th,td{padding:.55rem;border-bottom:1px solid #ddd;text-align:left}code{word-break:break-all}",
        ".note{padding:1rem;background:#fff0cf;border-left:5px solid #c00022}small{color:#546170}</style>",
        "<h1>RIGID · run report</h1>",
        f"<p class='note'>Input data: <strong>{esc(record['source_kind'])}</strong>. "
        f"Model training data: <strong>{esc(record.get('training_source_kind', 'unknown'))}</strong>. "
        "This run is not a reproduction of the manuscript's reported results.</p>",
        f"<p>Split: {esc(record['split'])} · Requested top-k: {esc(record['requested_k'])} · "
        f"Candidate library: {esc(record['candidate_count'])}</p>",
        "<h2>Evaluation</h2><table><tr><th>Measure</th><th>Value</th></tr>",
    ]
    for name, value in record["metrics"].items():
        if name == "per_target":
            continue
        if isinstance(value, dict) and "denominator" in value:
            shown = (
                f"{value['value']:.3f} ({value['numerator']}/{value['denominator']})"
                if value["value"] is not None
                else "Not available (no eligible annotations)"
            )
        else:
            shown = "Not available" if value is None else str(value)
        body.append(f"<tr><td>{esc(name)}</td><td>{esc(shown)}</td></tr>")
    body.append("</table><h2>Candidate rankings</h2>")
    for target, rows in record["rankings"].items():
        body.append(
            f"<details><summary>{esc(target)} · {len(rows)} candidates</summary><table>"
            "<tr><th>Rank / ID</th><th>Structure (SMILES)</th><th>Score</th>"
            "<th>Primary / independent r</th><th>QED</th></tr>"
        )
        for row in rows:
            body.append(
                f"<tr><td>{esc(row['rank'])} · {esc(row['candidate_id'])}</td>"
                f"<td><code>{esc(row['smiles'])}</code></td><td>{row['score']:.3f}</td>"
                f"<td>{row['predicted_profile_correlation']:.3f} / "
                f"{row['independent_profile_correlation']:.3f}</td>"
                f"<td>{row['chemical_properties']['qed']:.3f}</td></tr>"
            )
        body.append("</table></details>")
    body.append("<h2>Interpretation and provenance</h2><ul>")
    body.extend(f"<li>{esc(note)}</li>" for note in record["notes"])
    body.append(
        f"</ul><small>Checkpoint SHA-256: {esc(record['checkpoint_sha256'])}</small></html>"
    )
    path = Path(output)
    if path.exists():
        raise FileExistsError("Report output exists; choose a new filename")
    write_text(path, "\n".join(body))
