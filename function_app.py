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
# CONFIG / HELPERS
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


# =========================
# ULIMA / CAREER ENRICHMENT
# =========================
ULIMA_PATTERNS = [
    "universidad de lima",
    "university of lima",
]

CAREER_PATTERNS = {
    "Ingeniería Industrial": [
        "carrera de ingenieria industrial",
        "industrial engineering career",
        "ingenieria industrial",
        "industrial engineering",
    ],
    "Ingeniería de Sistemas": [
        "carrera de ingenieria de sistemas",
        "systems engineering career",
        "ingenieria de sistemas",
        "ingenieria de sistemas computacionales",
        "systems engineering",
        "computer systems engineering",
    ],
    "Ingeniería Civil": [
        "carrera de ingenieria civil",
        "civil engineering career",
        "ingenieria civil",
        "civil engineering",
    ],
}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""

    value = value.lower()
    value = value.replace("\\", " ")
    value = value.replace("–", " ").replace("—", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def split_semicolon_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def clean_author_full_name(value: str) -> str:
    value = re.sub(r"\s*\(\d+\)\s*$", "", value).strip()
    return value


def extract_author_name_from_block(block: str) -> str:
    return block.split(",", 1)[0].strip()


def is_ulima_text(value: str | None) -> bool:
    norm = normalize_text(value)
    return any(pattern in norm for pattern in ULIMA_PATTERNS)


def infer_career_from_text(value: str | None) -> str | None:
    norm = normalize_text(value)

    for career, patterns in CAREER_PATTERNS.items():
        if any(pattern in norm for pattern in patterns):
            return career

    return None


def enrich_ulima_fields(
    authors_value: str | None,
    author_full_names_value: str | None,
    authors_with_affiliations_value: str | None,
    affiliations_value: str | None,
) -> dict:
    short_authors = split_semicolon_values(authors_value)
    full_authors = [clean_author_full_name(x) for x in split_semicolon_values(author_full_names_value)]
    author_blocks = split_semicolon_values(authors_with_affiliations_value)

    ulima_authors: list[str] = []
    careers: list[str] = []
    first_author_ulima = False

    if author_blocks:
        for idx, block in enumerate(author_blocks):
            if is_ulima_text(block):
                if idx == 0:
                    first_author_ulima = True

                if idx < len(full_authors):
                    author_name = full_authors[idx]
                elif idx < len(short_authors):
                    author_name = short_authors[idx]
                else:
                    author_name = extract_author_name_from_block(block)

                if author_name and author_name not in ulima_authors:
                    ulima_authors.append(author_name)

                detected_career = infer_career_from_text(block)
                if detected_career and detected_career not in careers:
                    careers.append(detected_career)

    # fallback débil a affiliations globales si no hubo carrera por bloque autor-afiliación
    if not careers and is_ulima_text(affiliations_value):
        detected_career = infer_career_from_text(affiliations_value)
        if detected_career:
            careers.append(detected_career)

    return {
        "ulima_docentes_raw": "; ".join(ulima_authors) if ulima_authors else None,
        "first_author_ulima_raw": "True" if first_author_ulima else "False",
        "carrera_raw": "; ".join(careers) if careers else None,
    }


# =========================
# STORAGE
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


def download_blob_text(container_name: str, blob_name: str) -> str:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    data = blob_client.download_blob().readall()
    return data.decode("utf-8-sig")


def upload_text_blob(container_name: str, blob_name: str, content: str) -> None:
    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    blob_client.upload_blob(content.encode("utf-8"), overwrite=True)


# =========================
# SQL
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
    "affiliation_raw": ["Afiliaciones", "Affiliations", "Authors with affiliations"],
}


def map_row_to_staging(row: dict) -> dict:
    mapped = {}

    for target_column, aliases in COLUMN_ALIASES.items():
        mapped[target_column] = safe_get(row, aliases)

    authors_value = safe_get(row, ["Authors", "Author(s)", "Autor(es)"])
    author_full_names_value = safe_get(row, ["Author full names"])
    authors_with_affiliations_value = safe_get(row, ["Authors with affiliations"])
    affiliations_value = safe_get(row, ["Affiliations", "Afiliaciones"])

    enrichment = enrich_ulima_fields(
        authors_value=authors_value,
        author_full_names_value=author_full_names_value,
        authors_with_affiliations_value=authors_with_affiliations_value,
        affiliations_value=affiliations_value,
    )

    mapped["authors_raw"] = authors_value or mapped.get("authors_raw")
    mapped["affiliation_raw"] = affiliations_value or mapped.get("affiliation_raw")
    mapped["ulima_docentes_raw"] = enrichment["ulima_docentes_raw"]
    mapped["first_author_ulima_raw"] = enrichment["first_author_ulima_raw"]

    if enrichment["carrera_raw"]:
        mapped["carrera_raw"] = enrichment["carrera_raw"]

    if not mapped.get("indexation_raw"):
        mapped["indexation_raw"] = "Scopus"

    if not mapped.get("es_scopus_raw"):
        mapped["es_scopus_raw"] = "True"

    if enrichment["ulima_docentes_raw"]:
        mapped["metodo_cruce_scopus_raw"] = "AUTHORS_WITH_AFFILIATIONS"
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


def insert_rows_to_staging(run_id: int, source_file_name: str, rows: list[dict]) -> tuple[int, int]:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()
    inserted = 0
    rejected = 0

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()

        for idx, row in enumerate(rows, start=1):
            mapped = map_row_to_staging(row)

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


@app.route(route="run-ingest-scopus", methods=["GET", "POST"])
def run_ingest_scopus(req: func.HttpRequest) -> func.HttpResponse:
    run_id = None
    try:
        raw_container = get_env("RAW_CONTAINER")
        processed_container = get_env("PROCESSED_CONTAINER")
        logs_container = get_env("LOGS_CONTAINER")

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
            rows=rows
        )

        execute_upsert_from_staging(run_id)

        processed_name = f"processed/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.path.basename(blob_name)}"
        upload_text_blob(processed_container, processed_name, csv_text)

        log_payload = {
            "run_id": run_id,
            "source_blob": blob_name,
            "processed_blob": processed_name,
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
