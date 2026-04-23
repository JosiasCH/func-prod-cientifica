import json
import logging
import os
from datetime import datetime, timezone

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


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


def insert_pipeline_run(
    trigger_type: str,
    status: str,
    source_name: str = "FUNCTION_SMOKE"
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
                "N/A",
                "N/A",
                status,
                0,
                0,
                0,
                0
            )
        )
        conn.commit()

        cursor.execute(
            "SELECT TOP 1 run_id FROM meta.pipeline_runs ORDER BY run_id DESC"
        )
        row = cursor.fetchone()
        cursor.close()

    try:
        return int(row.run_id)
    except AttributeError:
        return int(row[0])


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container_status = validate_containers()

        payload = {
            "status": "ok",
            "utc_now": datetime.now(timezone.utc).isoformat(),
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
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="run-smoke", methods=["GET", "POST"])
def run_smoke(req: func.HttpRequest) -> func.HttpResponse:
    try:
        container_status = validate_containers()
        run_id = insert_pipeline_run(trigger_type="MANUAL", status="SUCCESS")

        payload = {
            "status": "ok",
            "message": "Smoke test executed successfully.",
            "run_id": run_id,
            "containers": container_status,
            "utc_now": datetime.now(timezone.utc).isoformat(),
        }

        return func.HttpResponse(
            json.dumps(payload, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json",
        )

    except Exception as exc:
        logging.exception("Smoke run failed")
        return func.HttpResponse(
            json.dumps(
                {
                    "status": "error",
                    "message": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            status_code=500,
            mimetype="application/json",
        )
