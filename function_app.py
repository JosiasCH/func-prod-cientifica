import csv
import hashlib
import io
import json
import logging
import os
import re
import unicodedata
from datetime import datetime, timezone

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


# =========================
# CONFIG / GENERAL HELPERS
# =========================
def get_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required setting: {name}")
    return value


def get_sql_connection_string() -> str:
    return get_env("SQL_CONNECTION_STRING")


def get_blob_service():
    from azure.storage.blob import BlobServiceClient

    conn_str = get_env("DATA_STORAGE_CONNECTION_STRING")
    return BlobServiceClient.from_connection_string(conn_str)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_get(row: dict, aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in row and row[alias] not in (None, ""):
            return str(row[alias]).strip()
    return None


def compute_record_hash(eid: str | None, doi: str | None, title: str | None) -> str:
    raw = f"{eid or ''}|{doi or ''}|{title or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_generic_text(value: str | None) -> str:
    if not value:
        return ""

    value = str(value).lower().strip()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("\\", " ")
    value = value.replace("–", " ").replace("—", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_person_name(value: str | None) -> str:
    if not value:
        return ""
    norm = normalize_generic_text(value)
    norm = norm.replace(",", " ")
    norm = norm.replace("/", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def split_semicolon_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(";") if part and str(part).strip()]


def clean_author_full_name(value: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", str(value)).strip()


def is_ulima_text(value: str | None) -> bool:
    norm = normalize_generic_text(value)
    return ("universidad de lima" in norm) or ("university of lima" in norm)


# =========================
# STORAGE HELPERS
# =========================
def validate_containers() -> dict:
    blob_service = get_blob_service()

    raw_container = get_env("RAW_CONTAINER")
    processed_container = get_env("PROCESSED_CONTAINER")
    logs_container = get_env("LOGS_CONTAINER")

    return {
        "raw_exists": blob_service.get_container_client(raw_container).exists(),
        "processed_exists": blob_service.get_container_client(processed_container).exists(),
        "logs_exists": blob_service.get_container_client(logs_container).exists(),
    }


def get_latest_csv_blob_name(container_name: str) -> str:
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(container_name)

    blobs = [b for b in container_client.list_blobs() if b.name.lower().endswith(".csv")]
    if not blobs:
        raise RuntimeError(f"No CSV files found in container: {container_name}")

    latest = max(blobs, key=lambda b: b.last_modified)
    return latest.name


def get_latest_excel_blob_name(container_name: str) -> str:
    blob_service = get_blob_service()
    container_client = blob_service.get_container_client(container_name)

    blobs = [
        b for b in container_client.list_blobs()
        if b.name.lower().endswith(".xlsx") or b.name.lower().endswith(".xlsm")
    ]
    if not blobs:
        raise RuntimeError(f"No Excel files found in container: {container_name}")

    latest = max(blobs, key=lambda b: b.last_modified)
    return latest.name


def download_blob_text(container_name: str, blob_name: str) -> str:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    data = blob_client.download_blob().readall()
    return data.decode("utf-8-sig")


def download_blob_bytes(container_name: str, blob_name: str) -> bytes:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()


def upload_text_blob(container_name: str, blob_name: str, content: str) -> None:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(content.encode("utf-8"), overwrite=True)


def upload_bytes_blob(container_name: str, blob_name: str, content: bytes) -> None:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(content, overwrite=True)


# =========================
# SQL HELPERS
# =========================
def create_pipeline_run(
    trigger_type: str,
    source_name: str,
    source_file_name: str,
    source_file_path: str
) -> int:
    from mssql_python import connect

    pipeline_name = os.environ.get("PIPELINE_NAME", "scopus_monthly_pipeline")
    sql_conn_str = get_sql_connection_string()

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO meta.pipeline_runs (
                pipeline_name,
                source_name,
                trigger_type,
                source_file_name,
                source_file_path,
                status,
                records_read,
                records_inserted,
                records_updated,
                records_rejected
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pipeline_name,
                source_name,
                trigger_type,
                source_file_name,
                source_file_path,
                "STARTED",
                0,
                0,
                0,
                0
            )
        )
        conn.commit()

        cursor.execute("SELECT TOP 1 run_id FROM meta.pipeline_runs ORDER BY run_id DESC")
        row = cursor.fetchone()
        cursor.close()

    try:
        return int(row.run_id)
    except AttributeError:
        return int(row[0])


def update_pipeline_run(
    run_id: int,
    status: str,
    records_read: int = 0,
    records_inserted: int = 0,
    records_updated: int = 0,
    records_rejected: int = 0,
    error_message: str | None = None
) -> None:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE meta.pipeline_runs
            SET
                ended_at_utc = SYSUTCDATETIME(),
                status = ?,
                records_read = ?,
                records_inserted = ?,
                records_updated = ?,
                records_rejected = ?,
                error_message = ?
            WHERE run_id = ?
            """,
            (
                status,
                records_read,
                records_inserted,
                records_updated,
                records_rejected,
                error_message,
                run_id
            )
        )
        conn.commit()
        cursor.close()


def insert_pipeline_run_smoke(trigger_type: str, status: str, source_name: str = "FUNCTION_SMOKE") -> int:
    return create_pipeline_run(
        trigger_type=trigger_type,
        source_name=source_name,
        source_file_name="N/A",
        source_file_path="N/A"
    )


def execute_upsert_from_staging(run_id: int) -> None:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute("EXEC dbo.usp_upsert_publications_from_stg @run_id = ?", (run_id,))
        conn.commit()
        cursor.close()


def row_attr(row, attr: str, idx: int):
    try:
        return getattr(row, attr)
    except AttributeError:
        return row[idx]


# =========================
# DOCENTES INGESTION
# =========================
DOCENTES_SHEET_CAREERS = {
    "Civil": "Ingeniería Civil",
    "Industrial": "Ingeniería Industrial",
    "Sistemas": "Ingeniería de Sistemas",
}


def normalize_docente_name(value: str | None) -> str:
    if not value:
        return ""
    norm = normalize_generic_text(value)
    norm = norm.replace("/", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def to_title_or_none(value: str | None) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.title()


def build_initials(nombres: str | None) -> str | None:
    if not nombres:
        return None
    parts = [p.strip() for p in nombres.split() if p.strip()]
    if not parts:
        return None
    return "".join(part[0].upper() for part in parts if part)


def parse_docente_principal_raw(docente_raw: str) -> dict:
    raw = str(docente_raw).strip()
    parts = [p.strip() for p in raw.split("/") if p.strip()]

    apellido_1 = None
    apellido_2 = None
    nombres = None

    if len(parts) >= 3:
        apellido_1 = to_title_or_none(parts[0])
        apellido_2 = to_title_or_none(parts[1])
        nombres = to_title_or_none(" ".join(parts[2:]))
    elif len(parts) == 2:
        apellido_1 = to_title_or_none(parts[0])
        nombres = to_title_or_none(parts[1])
    else:
        nombres = to_title_or_none(raw)

    full_for_normalization = " ".join([x for x in [apellido_1, apellido_2, nombres] if x])

    return {
        "nombre_original": raw,
        "nombre_normalizado": normalize_docente_name(full_for_normalization),
        "apellido_1": apellido_1,
        "apellido_2": apellido_2,
        "nombres": nombres,
        "iniciales": build_initials(nombres),
    }


def find_docentes_header_indices(worksheet) -> tuple[int, int | None, int]:
    max_scan_rows = min(10, worksheet.max_row)

    for row_idx in range(1, max_scan_rows + 1):
        row_values = list(
            worksheet.iter_rows(
                min_row=row_idx,
                max_row=row_idx,
                values_only=True
            )
        )[0]

        normalized = [normalize_generic_text(v) for v in row_values]

        docente_idx = None
        codigo_idx = None

        for idx, value in enumerate(normalized):
            if "docente principal" in value:
                docente_idx = idx
            if "codigo" in value and "docente" in value:
                codigo_idx = idx
            elif "codigo docente" in value:
                codigo_idx = idx
            elif "cod docente" in value:
                codigo_idx = idx

        if docente_idx is not None:
            return row_idx, codigo_idx, docente_idx

    raise RuntimeError(
        f"Could not find header row with 'Docente Principal' in sheet '{worksheet.title}'"
    )


def extract_docentes_from_workbook_bytes(
    workbook_bytes: bytes,
    source_file_name: str,
    periodo_academico: str,
) -> list[dict]:
    from openpyxl import load_workbook

    workbook = load_workbook(filename=io.BytesIO(workbook_bytes), data_only=True)
    extracted_rows: list[dict] = []

    for sheet_name, carrera in DOCENTES_SHEET_CAREERS.items():
        if sheet_name not in workbook.sheetnames:
            logging.warning("Sheet '%s' not found in workbook. Skipping.", sheet_name)
            continue

        ws = workbook[sheet_name]
        header_row_idx, codigo_idx, docente_idx = find_docentes_header_indices(ws)

        for excel_row_idx in range(header_row_idx + 1, ws.max_row + 1):
            row_values = list(
                ws.iter_rows(
                    min_row=excel_row_idx,
                    max_row=excel_row_idx,
                    values_only=True
                )
            )[0]

            docente_raw = None
            codigo_raw = None

            if docente_idx is not None and docente_idx < len(row_values):
                docente_raw = row_values[docente_idx]
            if codigo_idx is not None and codigo_idx < len(row_values):
                codigo_raw = row_values[codigo_idx]

            if docente_raw is None or str(docente_raw).strip() == "":
                continue

            parsed = parse_docente_principal_raw(str(docente_raw))

            extracted_rows.append(
                {
                    "periodo_academico": periodo_academico,
                    "source_file_name": source_file_name,
                    "source_sheet_name": sheet_name,
                    "source_row_number": excel_row_idx,
                    "carrera_fuente": carrera,
                    "codigo_docente_raw": str(codigo_raw).strip() if codigo_raw is not None else None,
                    "docente_principal_raw": parsed["nombre_original"],
                    "docente_principal_normalizado": parsed["nombre_normalizado"],
                    "apellido_1": parsed["apellido_1"],
                    "apellido_2": parsed["apellido_2"],
                    "nombres": parsed["nombres"],
                    "iniciales": parsed["iniciales"],
                }
            )

    if not extracted_rows:
        raise RuntimeError("No docentes extracted from workbook.")

    return extracted_rows


def insert_docentes_raw_rows(run_id: int, rows: list[dict]) -> int:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()

        for row in rows:
            cursor.execute(
                """
                INSERT INTO stg.docentes_ulima_raw (
                    run_id,
                    periodo_academico,
                    source_file_name,
                    source_sheet_name,
                    source_row_number,
                    carrera_fuente,
                    codigo_docente_raw,
                    docente_principal_raw,
                    docente_principal_normalizado
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row["periodo_academico"],
                    row["source_file_name"],
                    row["source_sheet_name"],
                    row["source_row_number"],
                    row["carrera_fuente"],
                    row["codigo_docente_raw"],
                    row["docente_principal_raw"],
                    row["docente_principal_normalizado"],
                )
            )

        conn.commit()
        cursor.close()

    return len(rows)


def rebuild_docentes_reference(run_id: int, rows: list[dict], periodo_academico: str) -> int:
    from mssql_python import connect

    unique_map: dict[tuple[str, str, str], dict] = {}

    for row in rows:
        key = (
            row["periodo_academico"],
            row["carrera_fuente"],
            row["docente_principal_normalizado"],
        )

        if key not in unique_map:
            unique_map[key] = {
                "periodo_academico": row["periodo_academico"],
                "carrera": row["carrera_fuente"],
                "codigo_docente": row["codigo_docente_raw"],
                "nombre_original": row["docente_principal_raw"],
                "nombre_normalizado": row["docente_principal_normalizado"],
                "apellido_1": row["apellido_1"],
                "apellido_2": row["apellido_2"],
                "nombres": row["nombres"],
                "iniciales": row["iniciales"],
                "activo": 1,
                "source_run_id": run_id,
            }
        else:
            if not unique_map[key]["codigo_docente"] and row["codigo_docente_raw"]:
                unique_map[key]["codigo_docente"] = row["codigo_docente_raw"]

    deduped_rows = list(unique_map.values())
    sql_conn_str = get_sql_connection_string()

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM ref.docentes_ulima WHERE periodo_academico = ?",
            (periodo_academico,)
        )

        for row in deduped_rows:
            cursor.execute(
                """
                INSERT INTO ref.docentes_ulima (
                    periodo_academico,
                    carrera,
                    codigo_docente,
                    nombre_original,
                    nombre_normalizado,
                    apellido_1,
                    apellido_2,
                    nombres,
                    iniciales,
                    activo,
                    source_run_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["periodo_academico"],
                    row["carrera"],
                    row["codigo_docente"],
                    row["nombre_original"],
                    row["nombre_normalizado"],
                    row["apellido_1"],
                    row["apellido_2"],
                    row["nombres"],
                    row["iniciales"],
                    row["activo"],
                    row["source_run_id"],
                )
            )

        conn.commit()
        cursor.close()

    return len(deduped_rows)


# =========================
# DOCENTES REF MATCHING
# =========================
def build_name_initials(nombres: str | None) -> str:
    if not nombres:
        return ""
    tokens = [t for t in normalize_generic_text(nombres).split() if t]
    return "".join(token[0].upper() for token in tokens if token)


def build_docente_display_name(docente: dict) -> str:
    family = " ".join([x for x in [docente.get("apellido_1"), docente.get("apellido_2")] if x])
    names = docente.get("nombres")
    if family and names:
        return f"{family.title()}, {names.title()}"
    return (docente.get("nombre_original") or "").replace("/", " ").title()


def get_docentes_reference(periodo_academico: str) -> list[dict]:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()
    docentes: list[dict] = []

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                docente_ref_id,
                periodo_academico,
                carrera,
                codigo_docente,
                nombre_original,
                nombre_normalizado,
                apellido_1,
                apellido_2,
                nombres,
                iniciales
            FROM ref.docentes_ulima
            WHERE periodo_academico = ?
              AND activo = 1
            """,
            (periodo_academico,)
        )

        rows = cursor.fetchall()
        cursor.close()

    for row in rows:
        docentes.append(
            {
                "docente_ref_id": row_attr(row, "docente_ref_id", 0),
                "periodo_academico": row_attr(row, "periodo_academico", 1),
                "carrera": row_attr(row, "carrera", 2),
                "codigo_docente": row_attr(row, "codigo_docente", 3),
                "nombre_original": row_attr(row, "nombre_original", 4),
                "nombre_normalizado": row_attr(row, "nombre_normalizado", 5),
                "apellido_1": row_attr(row, "apellido_1", 6),
                "apellido_2": row_attr(row, "apellido_2", 7),
                "nombres": row_attr(row, "nombres", 8),
                "iniciales": row_attr(row, "iniciales", 9),
            }
        )

    return docentes


def split_authors_with_affiliations_blocks(
    author_full_names_value: str | None,
    authors_with_affiliations_value: str | None
) -> list[str]:
    text = str(authors_with_affiliations_value or "").strip()
    if not text:
        return []

    full_names = [clean_author_full_name(x) for x in split_semicolon_values(author_full_names_value)]
    if not full_names:
        return split_semicolon_values(authors_with_affiliations_value)

    positions: list[tuple[str, int]] = []
    search_start = 0

    for name in full_names:
        pos = text.find(name, search_start)
        if pos == -1:
            pos = text.find(name)
        if pos != -1:
            positions.append((name, pos))
            search_start = pos + len(name)

    if len(positions) < 1:
        return split_semicolon_values(authors_with_affiliations_value)

    blocks: list[str] = []
    for i, (_, start_pos) in enumerate(positions):
        end_pos = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        block = text[start_pos:end_pos].strip(" ;")
        if block:
            blocks.append(block)

    return blocks


def parse_scopus_author_name(scopus_author_name: str) -> dict:
    raw = clean_author_full_name(scopus_author_name)

    if "," in raw:
        family_part, given_part = [p.strip() for p in raw.split(",", 1)]
    else:
        tokens = [t.strip() for t in raw.split() if t.strip()]
        if len(tokens) >= 2:
            family_part = " ".join(tokens[:-1])
            given_part = tokens[-1]
        else:
            family_part = raw
            given_part = ""

    family_norm = normalize_generic_text(family_part)
    family_tokens = [t for t in family_norm.split() if t]
    given_norm = normalize_generic_text(given_part.replace(".", " "))
    given_tokens = [t for t in given_norm.split() if t]

    apellido_1 = family_tokens[0] if family_tokens else None
    apellido_2 = " ".join(family_tokens[1:]) if len(family_tokens) > 1 else None
    initials = "".join(token[0].upper() for token in given_tokens if token)

    normalized_full = normalize_person_name(f"{family_part} {given_part}")

    return {
        "raw": raw,
        "apellido_1": apellido_1,
        "apellido_2": apellido_2,
        "given_tokens": given_tokens,
        "initials": initials,
        "normalized_full": normalized_full,
    }


def given_names_match(ref_nombres: str | None, scopus_given_tokens: list[str], scopus_initials: str) -> bool:
    ref_tokens = [t for t in normalize_generic_text(ref_nombres).split() if t]
    if not scopus_given_tokens and not scopus_initials:
        return True

    if scopus_initials and all(len(tok) == 1 for tok in scopus_given_tokens):
        ref_initials = "".join(token[0].upper() for token in ref_tokens if token)
        return ref_initials.startswith(scopus_initials)

    if scopus_given_tokens:
        if len(ref_tokens) < len(scopus_given_tokens):
            return False
        for idx, token in enumerate(scopus_given_tokens):
            if ref_tokens[idx] != token:
                return False
        return True

    return False


def match_scopus_author_to_docente(scopus_author_name: str, docentes_ref: list[dict]) -> dict:
    parsed = parse_scopus_author_name(scopus_author_name)

    exact_matches = [
        d for d in docentes_ref
        if d["nombre_normalizado"] == parsed["normalized_full"]
    ]

    if len(exact_matches) == 1:
        return {
            "matched": True,
            "ambiguous": False,
            "match_method": "DOCENTES_REF_EXACT",
            "docente": exact_matches[0],
        }

    if len(exact_matches) > 1:
        return {
            "matched": False,
            "ambiguous": True,
            "match_method": "DOCENTES_REF_EXACT_AMBIGUOUS",
            "docente": None,
        }

    if not parsed["apellido_1"] or not parsed["apellido_2"]:
        return {
            "matched": False,
            "ambiguous": False,
            "match_method": "DOCENTES_REF_NO_MATCH",
            "docente": None,
        }

    structured_matches = []
    for docente in docentes_ref:
        ref_ap1 = normalize_generic_text(docente.get("apellido_1"))
        ref_ap2 = normalize_generic_text(docente.get("apellido_2"))

        if ref_ap1 != parsed["apellido_1"]:
            continue
        if ref_ap2 != parsed["apellido_2"]:
            continue
        if not given_names_match(docente.get("nombres"), parsed["given_tokens"], parsed["initials"]):
            continue

        structured_matches.append(docente)

    if len(structured_matches) == 1:
        return {
            "matched": True,
            "ambiguous": False,
            "match_method": "DOCENTES_REF_STRUCTURED",
            "docente": structured_matches[0],
        }

    if len(structured_matches) > 1:
        return {
            "matched": False,
            "ambiguous": True,
            "match_method": "DOCENTES_REF_STRUCTURED_AMBIGUOUS",
            "docente": None,
        }

    return {
        "matched": False,
        "ambiguous": False,
        "match_method": "DOCENTES_REF_NO_MATCH",
        "docente": None,
    }


def enrich_ulima_fields_from_ref(
    authors_value: str | None,
    author_full_names_value: str | None,
    authors_with_affiliations_value: str | None,
    affiliations_value: str | None,
    docentes_ref: list[dict],
) -> dict:
    short_authors = split_semicolon_values(authors_value)
    full_authors = [clean_author_full_name(x) for x in split_semicolon_values(author_full_names_value)]
    author_blocks = split_authors_with_affiliations_blocks(
        author_full_names_value=author_full_names_value,
        authors_with_affiliations_value=authors_with_affiliations_value,
    )

    matched_docentes: list[str] = []
    matched_careers: list[str] = []
    first_author_ulima = False
    matched_any = False

    for idx, block in enumerate(author_blocks):
        if not is_ulima_text(block):
            continue

        if idx < len(full_authors):
            scopus_author_name = full_authors[idx]
        elif idx < len(short_authors):
            scopus_author_name = short_authors[idx]
        else:
            scopus_author_name = block.split(",", 2)[0].strip()

        match_result = match_scopus_author_to_docente(scopus_author_name, docentes_ref)

        if match_result["matched"]:
            matched_any = True
            docente = match_result["docente"]
            display_name = build_docente_display_name(docente)

            if display_name and display_name not in matched_docentes:
                matched_docentes.append(display_name)

            carrera = docente.get("carrera")
            if carrera and carrera not in matched_careers:
                matched_careers.append(carrera)

            if idx == 0:
                first_author_ulima = True

    return {
        "ulima_docentes_raw": "; ".join(matched_docentes) if matched_docentes else None,
        "first_author_ulima_raw": "True" if first_author_ulima else "False",
        "carrera_raw": "; ".join(matched_careers) if matched_careers else None,
        "metodo_cruce_scopus_raw": "AUTHORS_WITH_AFFILIATIONS+DOCENTES_REF" if matched_any else None,
        "es_ulima_raw_detected": True if any(is_ulima_text(b) for b in author_blocks) or is_ulima_text(affiliations_value) else False,
    }


# =========================
# CSV MAPPING
# =========================
COLUMN_ALIASES = {
    "eid": ["EID", "eid"],
    "publication_year_raw": ["Year", "Año"],
    "scopus_year_raw": ["Año (Scopus)", "Scopus Year", "Year"],
    "publication_type_raw": ["Tipo de publicación", "Document Type", "Source & document type", "Document Type (Scopus)"],
    "publication_title_raw": ["Título de la publicación", "Title", "Publication Title", "Document title"],
    "authors_raw": ["Authors", "Autor(es)", "Author(s)"],
    "conference_journal_raw": ["Conferencia/Journal", "Conference/Journal", "Conference Journal"],
    "indexation_raw": ["Indexación", "Indexation"],
    "editorial_publication_raw": ["Editorial de publicación", "Publisher"],
    "doi_link_raw": ["DOI /link", "DOI"],
    "publication_date_raw": ["Fecha de publicacion", "Publication date", "Date"],
    "publication_date_scopus_raw": ["Fecha de publicacion (Scopus)"],
    "revista_raw": ["Revista", "Source title", "Source Title"],
    "conference_raw": ["Conferencia", "Conference name"],
    "publication_status_raw": ["Estado publicación", "Publication Stage", "Publication Status"],
    "issn_raw": ["ISSN", "Serial identifiers (e.g. ISSN)"],
    "isbn_raw": ["ISBN"],
    "scopus_link_raw": ["Link SCOPUS", "Link"],
    "alternative_link_raw": ["LINK REVISTA (ALTERNATIVO)", "Alternative Link"],
    "category_tematica_raw": ["Categoria temática", "Categoría temática"],
    "area_idic_raw": ["Area IDIC"],
    "linea_idic_raw": ["Linea IDIC"],
    "area_carrera_raw": ["Area de la carrera"],
    "linea_carrera_raw": ["Linea de la carrera"],
    "carrera_raw": ["Carrera"],
    "es_scopus_raw": ["es_scopus"],
    "metodo_cruce_scopus_raw": ["Método de cruce Scopus", "Metodo de cruce Scopus"],
    "abstract_scopus_raw": ["Abstract Scopus", "Abstract"],
    "author_keywords_raw": ["Author keywords", "Author Keywords"],
    "index_keywords_raw": ["Index keywords", "Index Keywords"],
    "source_title_raw": ["Source title", "Source Title"],
    "document_type_scopus_raw": ["Document Type (Scopus)", "Document Type"],
    "affiliation_raw": ["Affiliations", "Afiliaciones"],
}


def map_row_to_staging(row: dict, docentes_ref: list[dict]) -> dict:
    mapped = {}

    for target_column, aliases in COLUMN_ALIASES.items():
        mapped[target_column] = safe_get(row, aliases)

    authors_value = safe_get(row, ["Authors", "Author(s)", "Autor(es)"])
    author_full_names_value = safe_get(row, ["Author full names"])
    authors_with_affiliations_value = safe_get(row, ["Authors with affiliations"])
    affiliations_value = safe_get(row, ["Affiliations", "Afiliaciones"])

    enrichment = enrich_ulima_fields_from_ref(
        authors_value=authors_value,
        author_full_names_value=author_full_names_value,
        authors_with_affiliations_value=authors_with_affiliations_value,
        affiliations_value=affiliations_value,
        docentes_ref=docentes_ref,
    )

    mapped["authors_raw"] = authors_value or mapped.get("authors_raw")
    mapped["affiliation_raw"] = authors_with_affiliations_value or affiliations_value or mapped.get("affiliation_raw")
    mapped["ulima_docentes_raw"] = enrichment["ulima_docentes_raw"]
    mapped["first_author_ulima_raw"] = enrichment["first_author_ulima_raw"]

    if enrichment["carrera_raw"]:
        mapped["carrera_raw"] = enrichment["carrera_raw"]

    if not mapped.get("indexation_raw"):
        mapped["indexation_raw"] = "Scopus"

    if not mapped.get("es_scopus_raw"):
        mapped["es_scopus_raw"] = "True"

    if enrichment["metodo_cruce_scopus_raw"]:
        mapped["metodo_cruce_scopus_raw"] = enrichment["metodo_cruce_scopus_raw"]
    elif mapped.get("eid") and not mapped.get("metodo_cruce_scopus_raw"):
        mapped["metodo_cruce_scopus_raw"] = "EID"

    record_hash = compute_record_hash(
        mapped.get("eid"),
        mapped.get("doi_link_raw"),
        mapped.get("publication_title_raw"),
    )

    has_identity = any([
        mapped.get("eid"),
        mapped.get("doi_link_raw"),
        mapped.get("publication_title_raw"),
    ])

    mapped["record_hash"] = record_hash
    mapped["is_valid_for_curated"] = 1 if has_identity else 0
    mapped["rejection_reason"] = None if has_identity else "Missing EID/DOI/title"

    return mapped


def parse_csv_text(csv_text: str) -> list[dict]:
    sample = csv_text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(csv_text), dialect=dialect)
    return list(reader)


def insert_rows_to_staging(run_id: int, source_file_name: str, rows: list[dict], docentes_ref: list[dict]) -> tuple[int, int]:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()
    inserted = 0
    rejected = 0

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()

        for idx, row in enumerate(rows, start=1):
            mapped = map_row_to_staging(row, docentes_ref=docentes_ref)

            cursor.execute(
                """
                INSERT INTO stg.scopus_raw_load (
                    run_id,
                    source_row_number,
                    source_file_name,
                    eid,
                    publication_year_raw,
                    scopus_year_raw,
                    publication_type_raw,
                    publication_title_raw,
                    authors_raw,
                    ulima_docentes_raw,
                    ulima_estudiantes_raw,
                    first_author_ulima_raw,
                    conference_journal_raw,
                    indexation_raw,
                    editorial_publication_raw,
                    doi_link_raw,
                    publication_date_raw,
                    publication_date_scopus_raw,
                    revista_raw,
                    conference_raw,
                    publication_status_raw,
                    issn_raw,
                    isbn_raw,
                    scopus_link_raw,
                    alternative_link_raw,
                    category_tematica_raw,
                    area_idic_raw,
                    linea_idic_raw,
                    area_carrera_raw,
                    linea_carrera_raw,
                    carrera_raw,
                    es_scopus_raw,
                    metodo_cruce_scopus_raw,
                    abstract_scopus_raw,
                    author_keywords_raw,
                    index_keywords_raw,
                    source_title_raw,
                    document_type_scopus_raw,
                    affiliation_raw,
                    record_hash,
                    is_valid_for_curated,
                    rejection_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    idx,
                    source_file_name,
                    mapped.get("eid"),
                    mapped.get("publication_year_raw"),
                    mapped.get("scopus_year_raw"),
                    mapped.get("publication_type_raw"),
                    mapped.get("publication_title_raw"),
                    mapped.get("authors_raw"),
                    mapped.get("ulima_docentes_raw"),
                    mapped.get("ulima_estudiantes_raw"),
                    mapped.get("first_author_ulima_raw"),
                    mapped.get("conference_journal_raw"),
                    mapped.get("indexation_raw"),
                    mapped.get("editorial_publication_raw"),
                    mapped.get("doi_link_raw"),
                    mapped.get("publication_date_raw"),
                    mapped.get("publication_date_scopus_raw"),
                    mapped.get("revista_raw"),
                    mapped.get("conference_raw"),
                    mapped.get("publication_status_raw"),
                    mapped.get("issn_raw"),
                    mapped.get("isbn_raw"),
                    mapped.get("scopus_link_raw"),
                    mapped.get("alternative_link_raw"),
                    mapped.get("category_tematica_raw"),
                    mapped.get("area_idic_raw"),
                    mapped.get("linea_idic_raw"),
                    mapped.get("area_carrera_raw"),
                    mapped.get("linea_carrera_raw"),
                    mapped.get("carrera_raw"),
                    mapped.get("es_scopus_raw"),
                    mapped.get("metodo_cruce_scopus_raw"),
                    mapped.get("abstract_scopus_raw"),
                    mapped.get("author_keywords_raw"),
                    mapped.get("index_keywords_raw"),
                    mapped.get("source_title_raw"),
                    mapped.get("document_type_scopus_raw"),
                    mapped.get("affiliation_raw"),
                    mapped.get("record_hash"),
                    mapped.get("is_valid_for_curated"),
                    mapped.get("rejection_reason"),
                )
            )

            if mapped.get("is_valid_for_curated") == 1:
                inserted += 1
            else:
                rejected += 1

        conn.commit()
        cursor.close()

    return inserted, rejected


# =========================
# HTTP FUNCTIONS
# =========================
@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container_status = validate_containers()

        payload = {
            "status": "ok",
            "utc_now": utc_now_iso(),
            "function_app": "funcprodcientificadev01",
            "sql_server": get_env("SQL_SERVER"),
            "sql_database": get_env("SQL_DATABASE"),
            "storage_account": get_env("DATA_STORAGE_ACCOUNT"),
            "containers": container_status,
        }

        return func.HttpResponse(
            json.dumps(payload, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.exception("Health check failed")
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="run-smoke", methods=["GET", "POST"])
def run_smoke(req: func.HttpRequest) -> func.HttpResponse:
    run_id = None
    try:
        container_status = validate_containers()
        run_id = insert_pipeline_run_smoke(trigger_type="MANUAL", status="STARTED")
        update_pipeline_run(run_id, status="SUCCESS")

        payload = {
            "status": "ok",
            "message": "Smoke test executed successfully.",
            "run_id": run_id,
            "containers": container_status,
            "utc_now": utc_now_iso(),
        }

        return func.HttpResponse(
            json.dumps(payload, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.exception("Smoke run failed")
        if run_id is not None:
            update_pipeline_run(run_id, status="FAILED", error_message=str(exc))
        return func.HttpResponse(
            json.dumps({"status": "error", "message": str(exc)}, ensure_ascii=False, indent=2),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="run-ingest-docentes", methods=["GET", "POST"])
def run_ingest_docentes(req: func.HttpRequest) -> func.HttpResponse:
    run_id = None

    try:
        docentes_container = get_env("DOCENTES_CONTAINER")
        periodo_academico = get_env("DOCENTES_ACTIVE_PERIOD")
        processed_container = get_env("PROCESSED_CONTAINER")
        logs_container = get_env("LOGS_CONTAINER")

        requested_blob = req.params.get("blob_name")
        blob_name = requested_blob or get_latest_excel_blob_name(docentes_container)

        run_id = create_pipeline_run(
            trigger_type="MANUAL",
            source_name="DOCENTES_ULIMA",
            source_file_name=blob_name,
            source_file_path=f"{docentes_container}/{blob_name}"
        )

        workbook_bytes = download_blob_bytes(docentes_container, blob_name)

        extracted_rows = extract_docentes_from_workbook_bytes(
            workbook_bytes=workbook_bytes,
            source_file_name=blob_name,
            periodo_academico=periodo_academico,
        )

        raw_rows_loaded = insert_docentes_raw_rows(run_id, extracted_rows)
        docentes_unique_loaded = rebuild_docentes_reference(
            run_id=run_id,
            rows=extracted_rows,
            periodo_academico=periodo_academico,
        )

        processed_name = (
            f"docentes/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.path.basename(blob_name)}"
        )
        upload_bytes_blob(processed_container, processed_name, workbook_bytes)

        log_payload = {
            "run_id": run_id,
            "source_blob": blob_name,
            "processed_blob": processed_name,
            "periodo_academico": periodo_academico,
            "raw_rows_loaded": raw_rows_loaded,
            "docentes_unique_loaded": docentes_unique_loaded,
            "utc_now": utc_now_iso(),
            "status": "SUCCESS",
        }

        log_name = f"docentes_run_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        upload_text_blob(logs_container, log_name, json.dumps(log_payload, ensure_ascii=False, indent=2))

        update_pipeline_run(
            run_id=run_id,
            status="SUCCESS",
            records_read=raw_rows_loaded,
            records_inserted=docentes_unique_loaded,
            records_updated=0,
            records_rejected=0,
            error_message=None
        )

        return func.HttpResponse(
            json.dumps(
                {
                    "status": "ok",
                    "message": "Docentes ingestion completed successfully.",
                    "run_id": run_id,
                    "source_blob": blob_name,
                    "processed_blob": processed_name,
                    "periodo_academico": periodo_academico,
                    "raw_rows_loaded": raw_rows_loaded,
                    "docentes_unique_loaded": docentes_unique_loaded,
                    "utc_now": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.exception("Docentes ingestion failed")

        if run_id is not None:
            update_pipeline_run(
                run_id=run_id,
                status="FAILED",
                error_message=str(exc)
            )

        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "run_id": run_id,
                    "utc_now": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="run-ingest-scopus", methods=["GET", "POST"])
def run_ingest_scopus(req: func.HttpRequest) -> func.HttpResponse:
    run_id = None
    try:
        raw_container = get_env("RAW_CONTAINER")
        processed_container = get_env("PROCESSED_CONTAINER")
        logs_container = get_env("LOGS_CONTAINER")
        periodo_academico = get_env("DOCENTES_ACTIVE_PERIOD")
        docentes_ref = get_docentes_reference(periodo_academico)

        requested_blob = req.params.get("blob_name")
        blob_name = requested_blob or get_latest_csv_blob_name(raw_container)

        run_id = create_pipeline_run(
            trigger_type="MANUAL",
            source_name="SCOPUS",
            source_file_name=blob_name,
            source_file_path=f"{raw_container}/{blob_name}"
        )

        csv_text = download_blob_text(raw_container, blob_name)
        rows = parse_csv_text(csv_text)

        records_read = len(rows)
        records_inserted, records_rejected = insert_rows_to_staging(
            run_id=run_id,
            source_file_name=blob_name,
            rows=rows,
            docentes_ref=docentes_ref
        )

        execute_upsert_from_staging(run_id)

        processed_name = f"processed/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.path.basename(blob_name)}"
        upload_text_blob(processed_container, processed_name, csv_text)

        log_payload = {
            "run_id": run_id,
            "source_blob": blob_name,
            "processed_blob": processed_name,
            "periodo_academico_docentes": periodo_academico,
            "records_read": records_read,
            "records_inserted_to_staging": records_inserted,
            "records_rejected": records_rejected,
            "utc_now": utc_now_iso(),
            "status": "SUCCESS",
        }

        log_name = f"run_{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        upload_text_blob(logs_container, log_name, json.dumps(log_payload, ensure_ascii=False, indent=2))

        update_pipeline_run(
            run_id=run_id,
            status="SUCCESS",
            records_read=records_read,
            records_inserted=records_inserted,
            records_updated=0,
            records_rejected=records_rejected,
            error_message=None
        )

        return func.HttpResponse(
            json.dumps(
                {
                    "status": "ok",
                    "message": "Scopus ingestion completed successfully.",
                    "run_id": run_id,
                    "source_blob": blob_name,
                    "processed_blob": processed_name,
                    "periodo_academico_docentes": periodo_academico,
                    "records_read": records_read,
                    "records_inserted_to_staging": records_inserted,
                    "records_rejected": records_rejected,
                    "utc_now": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.exception("Scopus ingestion failed")

        if run_id is not None:
            update_pipeline_run(
                run_id=run_id,
                status="FAILED",
                error_message=str(exc)
            )

        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                    "run_id": run_id,
                    "utc_now": utc_now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=500,
            mimetype="application/json",
        )
