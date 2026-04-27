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
    value = value.replace("–", " ").replace("—", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_person_name(value: str | None) -> str:
    if not value:
        return ""
    norm = normalize_generic_text(value)
    norm = norm.replace(",", " ")
    norm = norm.replace("/", " ")
    norm = norm.replace(".", " ")
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


def row_attr(row, attr: str, idx: int):
    try:
        return getattr(row, attr)
    except AttributeError:
        return row[idx]


def normalize_url_or_doi(value: str | None) -> str | None:
    if not value:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.lower().startswith("http://") or raw.lower().startswith("https://"):
        return raw

    if raw.lower().startswith("doi:"):
        raw = raw[4:].strip()

    return raw


def build_doi_url(doi_value: str | None) -> str | None:
    normalized = normalize_url_or_doi(doi_value)
    if not normalized:
        return None

    if normalized.lower().startswith("http://") or normalized.lower().startswith("https://"):
        return normalized

    return f"https://doi.org/{normalized}"


def is_conference_document_type(document_type_value: str | None) -> bool:
    norm = normalize_generic_text(document_type_value)
    return "conference" in norm


def derive_conference_journal_value(document_type_value: str | None) -> str | None:
    if not document_type_value:
        return None
    return "Conference" if is_conference_document_type(document_type_value) else "Journal"


def derive_revista_and_conference(
    source_title_value: str | None,
    document_type_value: str | None,
) -> tuple[str | None, str | None]:
    if not source_title_value:
        return None, None

    if is_conference_document_type(document_type_value):
        return None, source_title_value

    return source_title_value, None


def build_authors_display(
    author_full_names_value: str | None,
    authors_value: str | None,
) -> str | None:
    full_names = [clean_author_full_name(x) for x in split_semicolon_values(author_full_names_value)]
    full_names = [x for x in full_names if x]
    if full_names:
        return "; ".join(full_names)

    short_names = [clean_author_full_name(x) for x in split_semicolon_values(authors_value)]
    short_names = [x for x in short_names if x]
    if short_names:
        return "; ".join(short_names)

    return None


def unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def build_text_corpus(*values: str | None) -> str:
    parts: list[str] = []
    for value in values:
        norm = normalize_generic_text(value)
        if norm:
            parts.append(norm)
    return " | ".join(parts)


def phrase_in_text(text: str, phrase: str | None) -> bool:
    norm_phrase = normalize_generic_text(phrase)
    if not norm_phrase:
        return False
    return norm_phrase in text


def choose_best_scored_candidate(score_map: dict[str, int], min_score: int = 1) -> str | None:
    positives = [(k, v) for k, v in score_map.items() if v >= min_score]
    if not positives:
        return None

    positives.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score = positives[0]

    tied = [k for k, v in positives if v == best_score]
    if len(tied) > 1:
        return None

    return best_key


def choose_best_scored_candidate_relaxed(score_map: dict[str, int], min_score: int = 1) -> str | None:
    positives = [(k, v) for k, v in score_map.items() if v >= min_score]
    if not positives:
        return None

    positives.sort(key=lambda x: (-x[1], normalize_generic_text(x[0])))
    return positives[0][0]


def choose_best_scored_candidate_with_margin(
    score_map: dict[str, int],
    min_score: int = 1,
    min_margin: int = 1,
) -> str | None:
    positives = [(k, v) for k, v in score_map.items() if v >= min_score]
    if not positives:
        return None

    positives.sort(key=lambda x: (-x[1], normalize_generic_text(x[0])))
    best_key, best_score = positives[0]

    if len(positives) == 1:
        return best_key

    second_score = positives[1][1]
    if (best_score - second_score) < min_margin:
        return None

    return best_key


THEMATIC_FIELD_WEIGHTS = {
    "abstract": 10,
    "title": 4,
    "author_keywords": 3,
    "index_keywords": 2,
    "source_title": 1,
}

THEMATIC_STRICT_MIN_SCORE = 7
THEMATIC_STRICT_MIN_MARGIN = 2
THEMATIC_APPROX_MIN_SCORE = 4


def build_thematic_text_fields(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict[str, str]:
    return {
        "title": normalize_generic_text(title_value),
        "abstract": normalize_generic_text(abstract_value),
        "author_keywords": normalize_generic_text(author_keywords_value),
        "index_keywords": normalize_generic_text(index_keywords_value),
        "source_title": normalize_generic_text(source_title_value),
    }


def score_alias_in_thematic_fields(
    text_fields: dict[str, str],
    alias: str | None,
    bonus: int = 0,
) -> int:
    alias_norm = normalize_generic_text(alias)
    if not alias_norm:
        return 0

    score = 0
    for field_name, text_value in text_fields.items():
        if text_value and phrase_in_text(text_value, alias_norm):
            score += THEMATIC_FIELD_WEIGHTS.get(field_name, 1) + bonus
    return score


def build_weighted_candidate_score(
    text_fields: dict[str, str],
    primary_aliases: list[str],
    support_aliases: list[str],
) -> int:
    score = 0
    seen: set[str] = set()

    for alias in primary_aliases:
        alias_norm = normalize_generic_text(alias)
        if alias_norm and alias_norm not in seen:
            score += score_alias_in_thematic_fields(text_fields, alias_norm, bonus=2)
            seen.add(alias_norm)

    for alias in support_aliases:
        alias_norm = normalize_generic_text(alias)
        if alias_norm and alias_norm not in seen:
            score += score_alias_in_thematic_fields(text_fields, alias_norm, bonus=0)
            seen.add(alias_norm)

    return score


def clip_text(value: str | None, max_chars: int = 7000) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars]

CSV_DELIMITER_CANDIDATES = [",", ";", "\t", "|"]
CSV_PARSE_PREVIEW_ROWS = 25

SQL_SAFE_MAX_LENGTHS = {
    "eid": 64,
    "publication_year_raw": 16,
    "scopus_year_raw": 16,
    "conference_journal_raw": 32,
    "indexation_raw": 64,
    "issn_raw": 64,
    "isbn_raw": 64,
    "es_scopus_raw": 16,
    "retractado_raw": 16,
    "descontinuado_scopus_raw": 64,
    "first_author_ulima_raw": 16,
    "metodo_cruce_scopus_raw": 64,
}

ISSN_REGEX = re.compile(r"\b\d{4}-?\d{3}[\dXx]\b")
ISBN_REGEX = re.compile(r"\b(?:97[89][-\s]?)?(?:\d[-\s]?){9,12}[\dXx]\b")
EID_REGEX = re.compile(r"\b2-s2\.0-\d+\b", flags=re.IGNORECASE)
YEAR_REGEX = re.compile(r"^\d{4}$")


def clean_csv_header_value(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def strip_excel_separator_hint(csv_text: str) -> tuple[str, str | None]:
    if not csv_text:
        return csv_text, None

    lines = csv_text.splitlines()
    if not lines:
        return csv_text, None

    first_line = lines[0].strip()
    match = re.match(r"^sep=(.)$", first_line, flags=re.IGNORECASE)
    if not match:
        return csv_text, None

    delimiter = match.group(1)
    remaining_text = "\n".join(lines[1:])
    return remaining_text, delimiter


def parse_csv_rows_with_delimiter(csv_text: str, delimiter: str) -> list[list[str]]:
    reader = csv.reader(
        io.StringIO(csv_text),
        delimiter=delimiter,
        quotechar='"',
        doublequote=True,
        skipinitialspace=False,
    )

    rows: list[list[str]] = []
    for row in reader:
        cleaned_row = [str(cell) if cell is not None else "" for cell in row]
        if any(str(cell).strip() for cell in cleaned_row):
            rows.append(cleaned_row)

    return rows


def get_expected_csv_header_aliases() -> set[str]:
    aliases: set[str] = set()
    for alias_list in COLUMN_ALIASES.values():
        for alias in alias_list:
            aliases.add(normalize_generic_text(alias))

    extra_headers = [
        "Author full names",
        "Authors with affiliations",
        "Publisher",
        "Publication Stage",
        "Source title",
        "Document Type",
        "Affiliations",
    ]
    for header in extra_headers:
        aliases.add(normalize_generic_text(header))

    return aliases


def score_csv_header_match(headers: list[str]) -> int:
    expected_aliases = get_expected_csv_header_aliases()
    normalized_headers = [normalize_generic_text(clean_csv_header_value(h)) for h in headers]
    return sum(1 for h in normalized_headers if h in expected_aliases)


def score_csv_candidate(rows: list[list[str]]) -> int:
    if not rows:
        return -10_000

    headers = rows[0]
    header_count = len(headers)
    if header_count == 0:
        return -10_000

    score = 0
    score += score_csv_header_match(headers) * 100

    preview_rows = rows[1: 1 + CSV_PARSE_PREVIEW_ROWS]
    if not preview_rows:
        return score

    for row in preview_rows:
        if len(row) == header_count:
            score += 8
        elif abs(len(row) - header_count) == 1:
            score += 2
        else:
            score -= 10

    return score


def normalize_csv_row_length(row: list[str], headers: list[str]) -> list[str]:
    expected = len(headers)
    values = list(row)

    if len(values) < expected:
        values.extend([""] * (expected - len(values)))
        return values

    if len(values) > expected:
        prefix = values[: expected - 1]
        merged_tail = ",".join(v for v in values[expected - 1:] if v is not None)
        return prefix + [merged_tail]

    return values


def build_dict_rows_from_csv_rows(rows: list[list[str]]) -> list[dict]:
    if not rows:
        return []

    headers = [clean_csv_header_value(h) for h in rows[0]]
    dict_rows: list[dict] = []

    for raw_row in rows[1:]:
        normalized_row = normalize_csv_row_length(raw_row, headers)
        row_dict = {headers[idx]: normalized_row[idx].strip() for idx in range(len(headers))}
        dict_rows.append(row_dict)

    return dict_rows


def extract_first_regex_match(value: str | None, pattern: re.Pattern[str]) -> str | None:
    if not value:
        return None
    match = pattern.search(str(value))
    if not match:
        return None
    return match.group(0).strip()


def extract_all_regex_matches(value: str | None, pattern: re.Pattern[str]) -> list[str]:
    if not value:
        return []
    matches = [m.group(0).strip() for m in pattern.finditer(str(value))]
    return unique_keep_order(matches)


def sanitize_short_field(value: str | None, max_length: int) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length]


def sanitize_year_field(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip()
    if YEAR_REGEX.match(cleaned):
        return cleaned

    extracted = extract_first_regex_match(cleaned, re.compile(r"\b\d{4}\b"))
    return extracted


def sanitize_identifier_fields(mapped: dict) -> dict:
    sanitized = dict(mapped)

    sanitized["eid"] = extract_first_regex_match(sanitized.get("eid"), EID_REGEX) or sanitize_short_field(
        sanitized.get("eid"), SQL_SAFE_MAX_LENGTHS["eid"]
    )

    sanitized["publication_year_raw"] = sanitize_year_field(sanitized.get("publication_year_raw"))
    sanitized["scopus_year_raw"] = sanitize_year_field(sanitized.get("scopus_year_raw"))

    issn_matches = extract_all_regex_matches(sanitized.get("issn_raw"), ISSN_REGEX)
    sanitized["issn_raw"] = "; ".join(issn_matches) if issn_matches else sanitize_short_field(
        sanitized.get("issn_raw"), SQL_SAFE_MAX_LENGTHS["issn_raw"]
    )

    isbn_matches = extract_all_regex_matches(sanitized.get("isbn_raw"), ISBN_REGEX)
    sanitized["isbn_raw"] = "; ".join(isbn_matches) if isbn_matches else sanitize_short_field(
        sanitized.get("isbn_raw"), SQL_SAFE_MAX_LENGTHS["isbn_raw"]
    )

    for field_name, max_length in SQL_SAFE_MAX_LENGTHS.items():
        if field_name in {"eid", "publication_year_raw", "scopus_year_raw", "issn_raw", "isbn_raw"}:
            continue
        sanitized[field_name] = sanitize_short_field(sanitized.get(field_name), max_length)

    return sanitized



def coerce_choice(value: str | None, allowed_values: list[str]) -> str | None:
    if not value:
        return None

    norm_value = normalize_generic_text(value)
    if not norm_value:
        return None

    for allowed in allowed_values:
        if normalize_generic_text(allowed) == norm_value:
            return allowed

    return None


def parse_float_or_none(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def extract_json_object(text: str | None) -> dict | None:
    if not text:
        return None

    candidate = str(text).strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate.strip(), flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate.strip()).strip()

    try:
        loaded = json.loads(candidate)
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None

    try:
        loaded = json.loads(match.group(0))
        return loaded if isinstance(loaded, dict) else None
    except Exception:
        return None


def merge_non_null_fields(base: dict, filler: dict, fields: list[str]) -> dict:
    result = dict(base)
    for field in fields:
        if not result.get(field) and filler.get(field):
            result[field] = filler[field]
    return result


def has_any_thematic_field(payload: dict) -> bool:
    return any(
        [
            payload.get("area_carrera_raw"),
            payload.get("linea_carrera_raw"),
            payload.get("category_tematica_raw"),
            payload.get("area_idic_raw"),
            payload.get("linea_idic_raw"),
        ]
    )


def missing_thematic_fields(payload: dict) -> list[str]:
    fields = [
        "area_carrera_raw",
        "linea_carrera_raw",
        "category_tematica_raw",
        "area_idic_raw",
        "linea_idic_raw",
    ]
    return [f for f in fields if not payload.get(f)]


# =========================
# CAREER HINTS FROM TEXT
# =========================
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


def infer_careers_from_text(value: str | None) -> list[str]:
    norm = normalize_generic_text(value)
    found: list[str] = []
    for career, patterns in CAREER_PATTERNS.items():
        if any(pattern in norm for pattern in patterns):
            found.append(career)
    return found


def infer_career_from_text(value: str | None) -> str | None:
    careers = infer_careers_from_text(value)
    return careers[0] if careers else None


VALID_ENGINEERING_CAREERS = [
    "Ingeniería Industrial",
    "Ingeniería Civil",
    "Ingeniería de Sistemas",
]

VALID_SCOPUS_DOCUMENT_TYPES = {
    "article",
    "conference paper",
}

ULIMA_ENGINEERING_CONTEXT_HINTS = [
    "facultad de ingenieria",
    "faculty of engineering",
    "school of engineering",
    "engineering school",
    "escuela de ingenieria",
    "department of engineering",
    "departamento de ingenieria",
]

EXTERNAL_INSTITUTION_BOUNDARY_HINTS = [
    "universidad",
    "university",
    "college",
    "polytechnic",
    "politecnico",
]


def filter_valid_engineering_careers(careers: list[str] | None) -> list[str]:
    if not careers:
        return []
    return unique_keep_order([c for c in careers if c in VALID_ENGINEERING_CAREERS])


def normalize_document_type_for_filter(value: str | None) -> str:
    return normalize_generic_text(value)


def is_valid_scopus_document_type(value: str | None) -> bool:
    return normalize_document_type_for_filter(value) in VALID_SCOPUS_DOCUMENT_TYPES


def split_affiliation_clauses(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part and str(part).strip()]



def has_external_institution_boundary(value: str | None) -> bool:
    norm = normalize_generic_text(value)
    if not norm or is_ulima_text(value):
        return False
    return any(hint in norm for hint in EXTERNAL_INSTITUTION_BOUNDARY_HINTS)



def extract_ulima_local_contexts(
    value: str | None,
    max_lookback: int = 3,
    max_lookahead: int = 6,
) -> list[str]:
    """
    Extrae el contexto local alrededor de una afiliación ULima.

    Importante para Scopus: algunas afiliaciones vienen como:
        "Universidad de Lima, ..., Carrera de Ingeniería Industrial, Facultad de Ingeniería"
    Es decir, la carrera/facultad aparece DESPUÉS de "Universidad de Lima".

    La versión anterior solo miraba hacia atrás, por lo que podía rechazar falsamente
    artículos válidos de Ingeniería ULima cuando la carrera aparecía hacia adelante
    dentro del mismo bloque de autor/afiliación.
    """
    if not value or not is_ulima_text(value):
        return []

    clauses = split_affiliation_clauses(value)
    if not clauses:
        return []

    contexts: list[str] = []
    for idx, clause in enumerate(clauses):
        if not is_ulima_text(clause):
            continue

        # Mirada hacia atrás: útil para patrones tipo
        # "Carrera de Ingeniería Industrial, Universidad de Lima".
        start_idx = idx
        steps = 0
        while start_idx > 0 and steps < max_lookback:
            previous_clause = clauses[start_idx - 1]
            if has_external_institution_boundary(previous_clause):
                break
            start_idx -= 1
            steps += 1

        # Mirada hacia adelante: necesaria para patrones tipo
        # "Universidad de Lima, ..., Carrera de Ingeniería Industrial, Facultad de Ingeniería".
        end_idx = idx
        steps = 0
        while end_idx + 1 < len(clauses) and steps < max_lookahead:
            next_clause = clauses[end_idx + 1]
            if has_external_institution_boundary(next_clause):
                break
            end_idx += 1
            steps += 1

        context = ", ".join(clauses[start_idx : end_idx + 1]).strip(" ,")
        if context:
            contexts.append(context)

    return unique_keep_order(contexts)



def extract_ulima_local_contexts_from_values(*values: str | None) -> list[str]:
    contexts: list[str] = []
    for value in values:
        if not value:
            continue
        parts = split_semicolon_values(value)
        if not parts:
            parts = [str(value)]
        for part in parts:
            contexts.extend(extract_ulima_local_contexts(part))
    return unique_keep_order(contexts)



def is_ulima_engineering_context(value: str | None) -> bool:
    if not value or not is_ulima_text(value):
        return False

    if filter_valid_engineering_careers(infer_careers_from_text(value)):
        return True

    norm = normalize_generic_text(value)
    return any(hint in norm for hint in ULIMA_ENGINEERING_CONTEXT_HINTS)



def resolve_engineering_affiliation_details(*values: str | None) -> dict:
    ulima_contexts = extract_ulima_local_contexts_from_values(*values)
    detected_careers: list[str] = []
    has_engineering_context = False

    for context in ulima_contexts:
        detected_careers.extend(filter_valid_engineering_careers(infer_careers_from_text(context)))
        if is_ulima_engineering_context(context):
            has_engineering_context = True

    return {
        "ulima_contexts": ulima_contexts,
        "careers": unique_keep_order(detected_careers),
        "has_ulima_affiliation": bool(ulima_contexts),
        "has_ulima_engineering_context": has_engineering_context,
    }



def resolve_engineering_careers_from_affiliation_text(*values: str | None) -> list[str]:
    return resolve_engineering_affiliation_details(*values).get("careers", [])



def determine_row_engineering_eligibility(
    document_type_value: str | None,
    authors_with_affiliations_value: str | None,
    affiliations_value: str | None,
    enrichment: dict,
) -> dict:
    normalized_doc_type = normalize_document_type_for_filter(document_type_value)
    if normalized_doc_type not in VALID_SCOPUS_DOCUMENT_TYPES:
        return {
            "eligible": False,
            "carrera_raw": None,
            "reason": f"Invalid document type: {document_type_value or 'UNKNOWN'}",
        }

    careers_from_enrichment = filter_valid_engineering_careers(
        split_semicolon_values(enrichment.get("carrera_raw"))
    )
    if careers_from_enrichment and enrichment.get("has_ulima_engineering_affiliation_raw"):
        return {
            "eligible": True,
            "carrera_raw": "; ".join(careers_from_enrichment),
            "reason": None,
        }

    affiliation_details = resolve_engineering_affiliation_details(
        authors_with_affiliations_value,
        affiliations_value,
    )
    careers_from_affiliation = filter_valid_engineering_careers(affiliation_details.get("careers"))
    if careers_from_affiliation and affiliation_details.get("has_ulima_engineering_context"):
        return {
            "eligible": True,
            "carrera_raw": "; ".join(careers_from_affiliation),
            "reason": None,
        }

    return {
        "eligible": False,
        "carrera_raw": None,
        "reason": "Row excluded: no valid ULima engineering affiliation for Industrial/Civil/Systems.",
    }


# =========================
# CATALOGS
# =========================
CAREER_AREA_LINE_CATALOG = {
    "Ingeniería Industrial": {
        "Work Design & Human Factors": [
            "Diseño de sistemas de trabajo",
            "Evaluación de factores físicos",
            "Evaluación ergonómica",
        ],
        "Operations Research & Analysis": [
            "Modelamiento matemático a la mejora de procesos como soporte a la toma de decisiones",
            "Simulación para la mejora del diseño de procesos",
            "Diseño y desarrollo de modelos para el análisis y predicción de las variables de un proceso",
        ],
        "Operations Engineering & Management": [
            "Planeamiento y Gestión de Operaciones",
            "Planeamiento, programación y control de proyectos",
            "Gestión de mantenimiento",
        ],
        "Supply Chain Management": [
            "Gestión de la cadena de suministro",
            "Gestión de Logística Inversa",
            "Gestión de Inventarios, Almacenes y Transportes",
            "Gestión de compras y proveedores, Nivel de servicio y Satisfacción al cliente",
        ],
        "Safety": [
            "Gestión de riesgos ocupacionales",
            "Identificación, análisis, evaluación y control de riesgos en seguridad y salud ocupacional",
        ],
        "Product Design & Development": [
            "Diseño de producto",
            "Desarrollo de producto",
        ],
    },
    "Ingeniería Civil": {
        "Ciencias de la Ingeniería": [
            "Ecuaciones diferenciales aplicadas al análisis estructural",
            "Física de materiales",
        ],
        "Construcción": [
            "Metodología BIM",
            "Normativa BIM",
            "Materiales de Construcción",
            "Sostenibilidad",
            "Innovación en Proceso constructivos",
            "Tecnología",
        ],
        "Estructuras": [
            "Técnicas de experimentación en estructuras",
            "Vulnerabilidad sísmica de estructuras",
            "Sistemas de protección sísmica de estructuras",
        ],
        "Innovación Empresarial": [
            "Obras de Construcción y su relación con el medio ambiente",
        ],
        "Hidráulica": [
            "Hidrología e Hidráulica",
            "Riego y Drenaje",
            "Calidad del Agua",
            "Cambio Climático en Recursos Hídricos",
            "Transporte de Sedimentos",
            "Hidrogeología",
        ],
        "Transporte y Geotecnia": [
            "Geotecnia computacional",
            "Geotecnia ambiental e hidrogeología",
            "Geotecnia experimental",
            "Geotecnia minera",
            "Mecánica de rocas e ingeniería geológica",
            "Tecnologia de mezclas asfalticas",
            "Diseño estructural de Pavimentos",
            "Mejora y estabilización del suelo",
            "Cambio climatico en la infraestructura vial",
            "Gestion de riesgos geológicos",
        ],
        "Gestión de Proyectos": [
            "Gestión de riesgos",
            "Gestión estratégica de contratos",
            "Gestión de las comunicaciones",
            "Gestión de recursos",
        ],
    },
    "Ingeniería de Sistemas": {
        "Aplicaciones en inteligencia artificial": [
            "Visión computacional",
            "NLP",
            "Aprendizaje automático",
            "Minería de datos",
        ],
        "Sistemas de Tecnologías de Información (TI)": [
            "Seguridad de sistemas y aplicaciones",
            "IoT",
            "Computación de alto rendimiento",
            "Redes y ciberseguridad",
            "Sostenibilidad en TI",
        ],
        "Interacción humano-computadora": [
            "HCI",
            "Realidad virtual y aumentada",
            "Construcción de juegos y gamificación",
            "Agentes virtuales",
        ],
        "Algoritmos y sistemas computacionales": [
            "Optimización computacional",
            "Ingeniería de software",
            "Diseño de algoritmos",
        ],
        "Tecnologías y gestión de la información": [
            "Gestión de procesos tecnológicos",
            "Liderazgo, género y tecnología",
            "Sistemas de gestión del conocimiento",
            "Minería de procesos",
            "Computación aplicada",
            "Simulación de procesos",
        ],
    },
}

IDIC_CATEGORY_AREA_LINE_CATALOG = {
    "Innovación y tecnología digital": {
        "Inteligencia artificial y computación avanzada": [
            "Machine learning y deep learning",
            "Procesamiento de lenguaje natural",
            "Visión computacional",
            "Sistemas autónomos y robótica",
        ],
        "Transformación digital": [
            "Tecnologías emergentes",
            "Ciberseguridad y privacidad",
            "Internet de las cosas (IoT)",
            "Computación cuántica",
            "Diseño y construcción virtual",
        ],
        "Experiencia digital humana": [
            "Interacción humano-computadora",
            "Realidad virtual y aumentada",
            "Diseño de interfaces adaptativas",
        ],
    },
    "Desarrollo sostenible y medioambiente": {
        "Sostenibilidad y cambio climático": [
            "Energías renovables",
            "Economía circular",
            "Gestión sostenible de recursos",
            "Adaptación al cambio climático",
        ],
        "Ciudades inteligentes y sostenibles": [
            "Urbanismo sostenible",
            "Movilidad urbana",
            "Infraestructura sostenible",
            "Gestión inteligente de recursos",
        ],
        "Tecnología y ecosistemas": [
            "Tecnologías limpias",
            "Biodiversidad y conservación",
            "Gestión de residuos",
            "Materiales avanzados",
        ],
    },
    "Sociedad y comportamiento humano": {
        "Bienestar y desarrollo humano": [
            "Salud mental y bienestar",
            "Educación, desarrollo cognitivo y socioafectivo",
            "Comportamiento social",
            "Mujer, cultura y sociedad",
            "Pobreza e informalidad",
        ],
        "Comunicación y cultura digital": [
            "Medios digitales y sociedad",
            "Comunicación intercultural",
            "Narrativas transmedia",
            "Comportamiento digital",
        ],
        "Ética, gobernanza y responsabilidad social": [
            "Ética y gobernanza",
            "Responsabilidad social",
            "Derechos humanos y tecnología",
        ],
    },
    "Gestión y economía del conocimiento": {
        "Innovación empresarial": [
            "Modelos de negocio digitales",
            "Emprendimiento tecnológico",
            "Gestión de la innovación",
            "Transformación organizacional",
        ],
        "Economía digital": [
            "Fintech y servicios financieros",
            "Mercados globales",
            "Análisis de datos económicos",
            "Economía de plataformas",
        ],
        "Gestión del conocimiento": [
            "Gestión del capital intelectual",
            "Aprendizaje organizacional",
            "Transferencia de conocimiento",
            "Inteligencia de negocios",
        ],
    },
}


def build_career_line_to_area_map() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for carrera, areas in CAREER_AREA_LINE_CATALOG.items():
        result[carrera] = {}
        for area, lineas in areas.items():
            for linea in lineas:
                result[carrera][normalize_generic_text(linea)] = area
    return result


def build_idic_maps() -> tuple[dict[str, str], dict[str, str]]:
    linea_to_area: dict[str, str] = {}
    area_to_category: dict[str, str] = {}

    for category, areas in IDIC_CATEGORY_AREA_LINE_CATALOG.items():
        for area, lineas in areas.items():
            area_to_category[normalize_generic_text(area)] = category
            for linea in lineas:
                linea_to_area[normalize_generic_text(linea)] = area

    return linea_to_area, area_to_category


CAREER_LINE_TO_AREA_MAP = build_career_line_to_area_map()
IDIC_LINE_TO_AREA_MAP, IDIC_AREA_TO_CATEGORY_MAP = build_idic_maps()


def get_allowed_career_areas(carrera: str | None) -> list[str]:
    if not carrera:
        return []
    return list(CAREER_AREA_LINE_CATALOG.get(carrera, {}).keys())


def get_allowed_career_lines(carrera: str | None, area_carrera: str | None = None) -> list[str]:
    if not carrera:
        return []

    areas = CAREER_AREA_LINE_CATALOG.get(carrera, {})
    if not areas:
        return []

    if area_carrera:
        return list(areas.get(area_carrera, []))

    result: list[str] = []
    for lineas in areas.values():
        result.extend(lineas)
    return result


def get_allowed_idic_categories() -> list[str]:
    return list(IDIC_CATEGORY_AREA_LINE_CATALOG.keys())


def get_allowed_idic_areas(category_tematica: str | None = None) -> list[str]:
    if category_tematica:
        return list(IDIC_CATEGORY_AREA_LINE_CATALOG.get(category_tematica, {}).keys())

    result: list[str] = []
    for areas in IDIC_CATEGORY_AREA_LINE_CATALOG.values():
        result.extend(list(areas.keys()))
    return result


def get_allowed_idic_lines(
    category_tematica: str | None = None,
    area_idic: str | None = None,
) -> list[str]:
    if category_tematica and area_idic:
        return list(IDIC_CATEGORY_AREA_LINE_CATALOG.get(category_tematica, {}).get(area_idic, []))

    if category_tematica:
        result: list[str] = []
        for lineas in IDIC_CATEGORY_AREA_LINE_CATALOG.get(category_tematica, {}).values():
            result.extend(lineas)
        return result

    result: list[str] = []
    for areas in IDIC_CATEGORY_AREA_LINE_CATALOG.values():
        for lineas in areas.values():
            result.extend(lineas)
    return result


def coerce_area_carrera_from_linea(carrera: str | None, linea_carrera: str | None) -> str | None:
    if not carrera or not linea_carrera:
        return None
    return CAREER_LINE_TO_AREA_MAP.get(carrera, {}).get(normalize_generic_text(linea_carrera))


def coerce_category_tematica_from_area(area_idic: str | None) -> str | None:
    if not area_idic:
        return None
    return IDIC_AREA_TO_CATEGORY_MAP.get(normalize_generic_text(area_idic))


def coerce_area_idic_from_linea(linea_idic: str | None) -> str | None:
    if not linea_idic:
        return None
    return IDIC_LINE_TO_AREA_MAP.get(normalize_generic_text(linea_idic))


def is_valid_career_area_line(
    carrera: str | None,
    area_carrera: str | None,
    linea_carrera: str | None,
) -> bool:
    if not carrera or not area_carrera or not linea_carrera:
        return False
    return linea_carrera in CAREER_AREA_LINE_CATALOG.get(carrera, {}).get(area_carrera, [])


def is_valid_idic_triplet(
    category_tematica: str | None,
    area_idic: str | None,
    linea_idic: str | None,
) -> bool:
    if not category_tematica or not area_idic or not linea_idic:
        return False
    return linea_idic in IDIC_CATEGORY_AREA_LINE_CATALOG.get(category_tematica, {}).get(area_idic, [])


# =========================
# CLASSIFICATION HINTS (fallback)
# =========================
CAREER_LINE_HINTS = {
    "Ingeniería Industrial": {
        "Diseño de sistemas de trabajo": [
            "diseno de sistemas de trabajo",
            "work design",
            "work system design",
        ],
        "Evaluación de factores físicos": [
            "factores fisicos",
            "physical factors",
        ],
        "Evaluación ergonómica": [
            "ergonomia",
            "ergonomic",
            "ergonomics",
        ],
        "Modelamiento matemático a la mejora de procesos como soporte a la toma de decisiones": [
            "modelamiento matematico",
            "mathematical modeling",
            "decision making",
        ],
        "Simulación para la mejora del diseño de procesos": [
            "simulacion",
            "simulation",
            "process simulation",
        ],
        "Diseño y desarrollo de modelos para el análisis y predicción de las variables de un proceso": [
            "prediccion",
            "prediction",
            "forecasting",
            "analisis de variables",
            "benchmarking",
            "multi model",
            "multi-model",
            "cross validation",
            "cross-validation",
            "particle swarm optimization",
            "pso",
            "machine learning",
        ],
        "Planeamiento y Gestión de Operaciones": [
            "operations management",
            "gestion de operaciones",
            "planeamiento de operaciones",
            "operational efficiency",
            "process improvement",
            "lean",
            "5s",
            "standard work",
            "otif",
            "productivity improvement",
        ],
        "Planeamiento, programación y control de proyectos": [
            "project scheduling",
            "project control",
            "programacion y control de proyectos",
        ],
        "Gestión de mantenimiento": [
            "mantenimiento",
            "maintenance management",
        ],
        "Gestión de la cadena de suministro": [
            "supply chain",
            "cadena de suministro",
        ],
        "Gestión de Logística Inversa": [
            "logistica inversa",
            "reverse logistics",
        ],
        "Gestión de Inventarios, Almacenes y Transportes": [
            "inventarios",
            "almacenes",
            "transportes",
            "inventory",
            "warehouse",
            "transportation",
            "inventory management",
            "warehouse management",
            "storage center",
            "packing process",
            "otif",
        ],
        "Gestión de compras y proveedores, Nivel de servicio y Satisfacción al cliente": [
            "compras",
            "proveedores",
            "nivel de servicio",
            "satisfaccion al cliente",
            "supplier",
            "service level",
        ],
        "Gestión de riesgos ocupacionales": [
            "riesgos ocupacionales",
            "occupational risks",
        ],
        "Identificación, análisis, evaluación y control de riesgos en seguridad y salud ocupacional": [
            "seguridad y salud ocupacional",
            "occupational health",
            "occupational safety",
        ],
        "Diseño de producto": [
            "diseno de producto",
            "product design",
        ],
        "Desarrollo de producto": [
            "desarrollo de producto",
            "product development",
            "nanoparticles",
            "silver nanoparticles",
            "green synthesis",
            "antimicrobial applications",
            "stone mastic asphalt",
            "sma mixtures",
            "asphalt mixtures",
            "sustainable fibres",
            "waste derived fibres",
            "material formulation",
            "composite materials",
        ],
    },
    "Ingeniería Civil": {
        "Ecuaciones diferenciales aplicadas al análisis estructural": [
            "analisis estructural",
            "structural analysis",
            "ecuaciones diferenciales",
        ],
        "Física de materiales": [
            "fisica de materiales",
            "materials physics",
        ],
        "Metodología BIM": [
            "bim",
            "building information modeling",
        ],
        "Normativa BIM": [
            "normativa bim",
            "bim standard",
            "bim standards",
        ],
        "Materiales de Construcción": [
            "materiales de construccion",
            "construction materials",
        ],
        "Sostenibilidad": [
            "sostenibilidad",
            "sustainability",
        ],
        "Innovación en Proceso constructivos": [
            "proceso constructivo",
            "construction process innovation",
        ],
        "Tecnología": [
            "tecnologia en construccion",
            "construction technology",
        ],
        "Técnicas de experimentación en estructuras": [
            "experimental structures",
            "experimentacion en estructuras",
        ],
        "Vulnerabilidad sísmica de estructuras": [
            "vulnerabilidad sismica",
            "seismic vulnerability",
        ],
        "Sistemas de protección sísmica de estructuras": [
            "proteccion sismica",
            "seismic protection",
            "base isolation",
        ],
        "Obras de Construcción y su relación con el medio ambiente": [
            "medio ambiente",
            "environmental impact",
            "construction and environment",
        ],
        "Hidrología e Hidráulica": [
            "hidrologia",
            "hidraulica",
            "hydrology",
            "hydraulics",
        ],
        "Riego y Drenaje": [
            "riego",
            "drenaje",
            "irrigation",
            "drainage",
        ],
        "Calidad del Agua": [
            "calidad del agua",
            "water quality",
        ],
        "Cambio Climático en Recursos Hídricos": [
            "cambio climatico",
            "climate change",
            "recursos hidricos",
            "water resources",
        ],
        "Transporte de Sedimentos": [
            "sedimentos",
            "sediment transport",
        ],
        "Hidrogeología": [
            "hidrogeologia",
            "hydrogeology",
        ],
        "Geotecnia computacional": [
            "geotecnia computacional",
            "computational geotechnics",
        ],
        "Geotecnia ambiental e hidrogeología": [
            "geotecnia ambiental",
            "environmental geotechnics",
        ],
        "Geotecnia experimental": [
            "geotecnia experimental",
            "experimental geotechnics",
        ],
        "Geotecnia minera": [
            "geotecnia minera",
            "mining geotechnics",
        ],
        "Mecánica de rocas e ingeniería geológica": [
            "mecanica de rocas",
            "rock mechanics",
            "ingenieria geologica",
        ],
        "Tecnologia de mezclas asfalticas": [
            "mezclas asfalticas",
            "asphalt mixtures",
        ],
        "Diseño estructural de Pavimentos": [
            "pavimentos",
            "pavement design",
        ],
        "Mejora y estabilización del suelo": [
            "estabilizacion del suelo",
            "soil stabilization",
            "mejora del suelo",
        ],
        "Cambio climatico en la infraestructura vial": [
            "infraestructura vial",
            "road infrastructure",
            "climate change in transport infrastructure",
        ],
        "Gestion de riesgos geológicos": [
            "riesgos geologicos",
            "geological risks",
        ],
        "Gestión de riesgos": [
            "gestion de riesgos",
            "risk management",
        ],
        "Gestión estratégica de contratos": [
            "contratos",
            "contract management",
        ],
        "Gestión de las comunicaciones": [
            "communications management",
            "gestion de las comunicaciones",
        ],
        "Gestión de recursos": [
            "resource management",
            "gestion de recursos",
        ],
    },
    "Ingeniería de Sistemas": {
        "Visión computacional": [
            "vision computacional",
            "computer vision",
        ],
        "NLP": [
            "nlp",
            "natural language processing",
            "procesamiento de lenguaje natural",
        ],
        "Aprendizaje automático": [
            "aprendizaje automatico",
            "machine learning",
        ],
        "Minería de datos": [
            "mineria de datos",
            "data mining",
        ],
        "Seguridad de sistemas y aplicaciones": [
            "seguridad de sistemas",
            "application security",
            "systems security",
        ],
        "IoT": [
            "iot",
            "internet of things",
        ],
        "Computación de alto rendimiento": [
            "high performance computing",
            "computacion de alto rendimiento",
            "hpc",
        ],
        "Redes y ciberseguridad": [
            "ciberseguridad",
            "cybersecurity",
            "networks",
            "redes",
        ],
        "Sostenibilidad en TI": [
            "sostenibilidad en ti",
            "green it",
        ],
        "HCI": [
            "hci",
            "human computer interaction",
            "interaccion humano computadora",
        ],
        "Realidad virtual y aumentada": [
            "realidad virtual",
            "realidad aumentada",
            "virtual reality",
            "augmented reality",
        ],
        "Construcción de juegos y gamificación": [
            "gamification",
            "gamificacion",
            "game development",
            "juegos",
        ],
        "Agentes virtuales": [
            "virtual agents",
            "agentes virtuales",
            "chatbot",
            "chatbots",
        ],
        "Optimización computacional": [
            "optimizacion computacional",
            "computational optimization",
        ],
        "Ingeniería de software": [
            "ingenieria de software",
            "software engineering",
        ],
        "Diseño de algoritmos": [
            "diseno de algoritmos",
            "algorithm design",
        ],
        "Gestión de procesos tecnológicos": [
            "procesos tecnologicos",
            "technology process management",
        ],
        "Liderazgo, género y tecnología": [
            "genero y tecnologia",
            "gender and technology",
            "liderazgo y tecnologia",
            "women in stem",
            "stem leadership",
            "gender gap",
        ],
        "Sistemas de gestión del conocimiento": [
            "gestion del conocimiento",
            "knowledge management systems",
        ],
        "Minería de procesos": [
            "mineria de procesos",
            "process mining",
        ],
        "Computación aplicada": [
            "computacion aplicada",
            "applied computing",
        ],
        "Simulación de procesos": [
            "simulacion de procesos",
            "process simulation",
        ],
    },
}

IDIC_LINE_HINTS = {
    "Machine learning y deep learning": [
        "machine learning",
        "deep learning",
        "aprendizaje automatico",
    ],
    "Procesamiento de lenguaje natural": [
        "natural language processing",
        "procesamiento de lenguaje natural",
        "nlp",
    ],
    "Visión computacional": [
        "computer vision",
        "vision computacional",
    ],
    "Sistemas autónomos y robótica": [
        "robotica",
        "robotics",
        "autonomous systems",
    ],
    "Tecnologías emergentes": [
        "tecnologias emergentes",
        "emerging technologies",
    ],
    "Ciberseguridad y privacidad": [
        "ciberseguridad",
        "cybersecurity",
        "privacy",
        "privacidad",
    ],
    "Internet de las cosas (IoT)": [
        "internet of things",
        "iot",
    ],
    "Computación cuántica": [
        "computacion cuantica",
        "quantum computing",
    ],
    "Diseño y construcción virtual": [
        "construccion virtual",
        "virtual construction",
        "bim",
    ],
    "Interacción humano-computadora": [
        "human computer interaction",
        "interaccion humano computadora",
        "hci",
    ],
    "Realidad virtual y aumentada": [
        "virtual reality",
        "augmented reality",
        "realidad virtual",
        "realidad aumentada",
    ],
    "Diseño de interfaces adaptativas": [
        "adaptive interfaces",
        "interfaces adaptativas",
    ],
    "Energías renovables": [
        "energias renovables",
        "renewable energy",
    ],
    "Economía circular": [
        "economia circular",
        "circular economy",
        "valorization",
        "waste valorization",
        "recycling",
        "reuse",
    ],
    "Gestión sostenible de recursos": [
        "gestion sostenible de recursos",
        "sustainable resource management",
        "resource optimization",
        "inventory management",
        "warehouse management",
        "storage center",
    ],
    "Adaptación al cambio climático": [
        "adaptacion al cambio climatico",
        "climate adaptation",
    ],
    "Urbanismo sostenible": [
        "urbanismo sostenible",
        "sustainable urbanism",
    ],
    "Movilidad urbana": [
        "movilidad urbana",
        "urban mobility",
    ],
    "Infraestructura sostenible": [
        "infraestructura sostenible",
        "sustainable infrastructure",
    ],
    "Gestión inteligente de recursos": [
        "smart resource management",
        "gestion inteligente de recursos",
        "inventory management",
        "resource allocation",
        "storage center",
        "warehouse",
        "inventory",
    ],
    "Tecnologías limpias": [
        "tecnologias limpias",
        "clean technologies",
        "green synthesis",
        "eco friendly",
        "eco-friendly",
        "sustainable valorization",
        "clean technology",
    ],
    "Biodiversidad y conservación": [
        "biodiversidad",
        "conservacion",
        "biodiversity",
        "conservation",
    ],
    "Gestión de residuos": [
        "gestion de residuos",
        "waste management",
        "waste",
        "paper industry waste",
        "agro-industrial waste",
        "sludge",
        "residuos",
    ],
    "Materiales avanzados": [
        "materiales avanzados",
        "advanced materials",
        "nanoparticles",
        "silver nanoparticles",
    ],
    "Salud mental y bienestar": [
        "salud mental",
        "mental health",
        "wellbeing",
        "bienestar",
    ],
    "Educación, desarrollo cognitivo y socioafectivo": [
        "educacion",
        "education",
        "desarrollo cognitivo",
        "socioafectivo",
    ],
    "Comportamiento social": [
        "comportamiento social",
        "social behavior",
    ],
    "Mujer, cultura y sociedad": [
        "mujer",
        "woman",
        "women",
        "culture and society",
        "cultura y sociedad",
        "women in stem",
        "gender gap",
        "leadership in stem",
    ],
    "Pobreza e informalidad": [
        "pobreza",
        "poverty",
        "informalidad",
        "informality",
    ],
    "Medios digitales y sociedad": [
        "medios digitales",
        "digital media",
        "society",
    ],
    "Comunicación intercultural": [
        "comunicacion intercultural",
        "intercultural communication",
    ],
    "Narrativas transmedia": [
        "narrativas transmedia",
        "transmedia narratives",
    ],
    "Comportamiento digital": [
        "comportamiento digital",
        "digital behavior",
    ],
    "Ética y gobernanza": [
        "etica",
        "ethics",
        "governance",
        "gobernanza",
    ],
    "Responsabilidad social": [
        "responsabilidad social",
        "social responsibility",
    ],
    "Derechos humanos y tecnología": [
        "derechos humanos",
        "human rights",
        "technology",
    ],
    "Modelos de negocio digitales": [
        "modelos de negocio digitales",
        "digital business models",
    ],
    "Emprendimiento tecnológico": [
        "emprendimiento tecnologico",
        "technology entrepreneurship",
    ],
    "Gestión de la innovación": [
        "gestion de la innovacion",
        "innovation management",
        "improvement model",
        "continuous improvement",
        "process improvement",
        "5s",
        "standard work",
        "operational efficiency",
        "productivity improvement",
        "otif",
    ],
    "Transformación organizacional": [
        "transformacion organizacional",
        "organizational transformation",
        "organizational improvement",
        "process redesign",
        "workflow redesign",
    ],
    "Fintech y servicios financieros": [
        "fintech",
        "financial services",
        "servicios financieros",
    ],
    "Mercados globales": [
        "mercados globales",
        "global markets",
    ],
    "Análisis de datos económicos": [
        "analisis de datos economicos",
        "economic data analysis",
    ],
    "Economía de plataformas": [
        "economia de plataformas",
        "platform economy",
    ],
    "Gestión del capital intelectual": [
        "capital intelectual",
        "intellectual capital",
    ],
    "Aprendizaje organizacional": [
        "aprendizaje organizacional",
        "organizational learning",
    ],
    "Transferencia de conocimiento": [
        "transferencia de conocimiento",
        "knowledge transfer",
    ],
    "Inteligencia de negocios": [
        "inteligencia de negocios",
        "business intelligence",
    ],
}


def build_classification_score(corpus: str, aliases: list[str]) -> int:
    score = 0
    for alias in aliases:
        if phrase_in_text(corpus, alias):
            score += 1
    return score


INDUSTRIAL_ORA_STRONG_SIGNALS = [
    "optimization",
    "optimisation",
    "optimizing",
    "optimising",
    "mathematical model",
    "mathematical modeling",
    "mathematical modelling",
    "modelamiento matematico",
    "prediction",
    "predictive",
    "forecast",
    "forecasting",
    "simulation",
    "simulacion",
    "benchmarking",
    "cross validation",
    "cross-validation",
    "particle swarm optimization",
    "genetic algorithm",
    "decision support",
    "machine learning",
    "deep learning",
    "algorithm",
    "algoritmo",
]

INDUSTRIAL_SCM_STRONG_SIGNALS = [
    "inventory",
    "inventory management",
    "warehouse",
    "warehouse management",
    "storage center",
    "logistics",
    "transportation",
    "transport",
    "supply chain",
    "inventarios",
    "almacenes",
    "almacen",
    "logistica",
    "transportes",
    "cadena de suministro",
    "supplier",
    "proveedores",
    "otif",
]

INDUSTRIAL_OEM_STRONG_SIGNALS = [
    "5s",
    "standard work",
    "lean",
    "kaizen",
    "operations management",
    "gestion de operaciones",
    "planeamiento de operaciones",
    "process improvement",
    "continuous improvement",
    "operational efficiency",
    "productivity improvement",
    "maintenance",
    "mantenimiento",
]

INDUSTRIAL_PDD_STRONG_SIGNALS = [
    "product development",
    "desarrollo de producto",
    "product design",
    "diseno de producto",
    "prototype",
    "prototipo",
    "materials",
    "material",
    "advanced material",
    "composite",
    "green synthesis",
    "nanoparticles",
    "silver nanoparticles",
    "fibres",
    "fibers",
    "asphalt",
    "stone mastic asphalt",
    "valorization",
    "valorisation",
]


IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS = [
    "5s",
    "standard work",
    "lean",
    "kaizen",
    "continuous improvement",
    "process improvement",
    "improvement model",
    "operational efficiency",
    "productivity improvement",
    "innovation management",
    "gestion de la innovacion",
    "workflow improvement",
    "mejora de procesos",
]

IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS = [
    "organizational transformation",
    "organisational transformation",
    "transformacion organizacional",
    "organizational change",
    "organisational change",
    "change management",
    "business transformation",
    "process redesign",
    "workflow redesign",
    "organizational redesign",
    "restructuring",
    "digital transformation",
]


IDIC_MACHINE_LEARNING_STRONG_SIGNALS = [
    "machine learning",
    "deep learning",
    "aprendizaje automatico",
    "aprendizaje automático",
    "predictive model",
    "predictive models",
    "prediction model",
    "prediction models",
    "benchmarking",
    "multi model",
    "multi-model",
    "model benchmarking",
]

IDIC_CLIMATE_ADAPTATION_EXACT_SIGNALS = [
    "adaptacion al cambio climatico",
    "adaptación al cambio climático",
    "climate change adaptation",
    "adaptation to climate change",
    "climate adaptation",
]

IDIC_CLIMATE_TERMS = [
    "climate change",
    "cambio climatico",
    "cambio climático",
    "climatic change",
    "climate variability",
    "variabilidad climatica",
    "variabilidad climática",
    "global warming",
    "calentamiento global",
]

IDIC_ADAPTATION_CONTEXT_TERMS = [
    "adaptation",
    "adaptacion",
    "adaptación",
    "resilience",
    "resiliencia",
    "vulnerability",
    "vulnerabilidad",
    "climate risk",
    "riesgo climatico",
    "riesgo climático",
    "climate hazard",
    "amenaza climatica",
    "amenaza climática",
    "climate impact",
    "impacto climatico",
    "impacto climático",
    "exposure",
    "exposicion",
    "exposición",
]

IDIC_CIRCULAR_ECONOMY_STRONG_SIGNALS = [
    "circular economy",
    "economia circular",
    "economía circular",
    "valorization",
    "valorisation",
    "valorizacion",
    "valorización",
    "waste valorization",
    "recycling",
    "reciclaje",
    "reuse",
    "reutilizacion",
    "reutilización",
]

IDIC_CLEAN_TECH_STRONG_SIGNALS = [
    "clean technology",
    "clean technologies",
    "tecnologias limpias",
    "tecnologías limpias",
    "green synthesis",
    "sintesis verde",
    "síntesis verde",
    "eco friendly",
    "eco-friendly",
    "sustainable valorization",
    "sustainable valorisation",
    "valorizacion sostenible",
    "valorización sostenible",
]

IDIC_WASTE_MANAGEMENT_STRONG_SIGNALS = [
    "waste management",
    "gestion de residuos",
    "gestión de residuos",
    "waste",
    "residuos",
    "sludge",
    "lodo",
    "adsorption",
    "adsorcion",
    "adsorción",
    "removal",
    "remocion",
    "remoción",
]

IDIC_ADVANCED_MATERIALS_STRONG_SIGNALS = [
    "advanced materials",
    "materiales avanzados",
    "nanoparticles",
    "nanoparticulas",
    "nanopartículas",
    "silver nanoparticles",
    "composite",
    "composites",
    "material formulation",
    "fiber reinforced",
    "fibre reinforced",
    "fiber-reinforced",
    "fibre-reinforced",
    "concrete",
    "concreto",
    "beam",
    "beams",
    "vigas",
    "cyclic loading",
    "carga ciclica",
    "carga cíclica",
    "structural response",
    "respuesta estructural",
    "asphalt",
    "asfalto",
]

IDIC_SUSTAINABLE_INFRASTRUCTURE_STRONG_SIGNALS = [
    "sustainable infrastructure",
    "infraestructura sostenible",
    "urban infrastructure",
    "road infrastructure",
    "infraestructura vial",
    "pavement",
    "pavimento",
    "construction infrastructure",
]

IDIC_EMERGING_TECH_STRONG_SIGNALS = [
    "emerging technologies",
    "tecnologias emergentes",
    "tecnologías emergentes",
    "computational characterization",
    "caracterizacion computacional",
    "caracterización computacional",
    "density functional theory",
    "dft",
    "molecular docking",
    "molecular dynamics",
    "spectroscopic",
    "spectroscopy",
    "spectrometric",
    "mass spectrometry",
    "esi ms",
    "esi-ms",
    "hplc",
    "uhplc",
    "orbitrap",
    "chromatography",
    "chemical characterization",
    "bioactive compounds",
    "secondary metabolites",
    "antitubercular",
    "isoniazid",
    "hydrazones",
    "drug derivatives",
    "novel derivatives",
]


def contains_any_phrase_in_text_fields(text_fields: dict[str, str], phrases: list[str]) -> bool:
    for phrase in phrases:
        norm_phrase = normalize_generic_text(phrase)
        if not norm_phrase:
            continue
        for text_value in text_fields.values():
            if text_value and phrase_in_text(text_value, norm_phrase):
                return True
    return False


def count_phrase_hits_in_text(value: str | None, phrases: list[str]) -> int:
    norm_value = normalize_generic_text(value)
    if not norm_value:
        return 0

    hits = 0
    seen: set[str] = set()
    for phrase in phrases:
        norm_phrase = normalize_generic_text(phrase)
        if not norm_phrase or norm_phrase in seen:
            continue
        if phrase_in_text(norm_value, norm_phrase):
            hits += 1
            seen.add(norm_phrase)
    return hits


def is_line_eligible_by_domain_rules(carrera: str, linea: str, text_fields: dict[str, str]) -> bool:
    if carrera != "Ingeniería Industrial":
        return True

    if linea in (
        "Modelamiento matemático a la mejora de procesos como soporte a la toma de decisiones",
        "Simulación para la mejora del diseño de procesos",
        "Diseño y desarrollo de modelos para el análisis y predicción de las variables de un proceso",
    ):
        return contains_any_phrase_in_text_fields(text_fields, INDUSTRIAL_ORA_STRONG_SIGNALS)

    return True


def apply_industrial_line_bonus(carrera: str, linea: str, text_fields: dict[str, str], base_score: int) -> int:
    if carrera != "Ingeniería Industrial":
        return base_score

    title_text = text_fields.get("title")
    abstract_text = text_fields.get("abstract")
    keyword_text = " | ".join(
        [
            text_fields.get("author_keywords", ""),
            text_fields.get("index_keywords", ""),
        ]
    )

    if linea in (
        "Modelamiento matemático a la mejora de procesos como soporte a la toma de decisiones",
        "Simulación para la mejora del diseño de procesos",
        "Diseño y desarrollo de modelos para el análisis y predicción de las variables de un proceso",
    ):
        bonus = 0
        bonus += 3 * count_phrase_hits_in_text(title_text, INDUSTRIAL_ORA_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, INDUSTRIAL_ORA_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(keyword_text, INDUSTRIAL_ORA_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Gestión de Inventarios, Almacenes y Transportes":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Gestión de la cadena de suministro":
        bonus = 0
        bonus += 3 * count_phrase_hits_in_text(title_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, INDUSTRIAL_SCM_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Planeamiento y Gestión de Operaciones":
        bonus = 0
        bonus += 3 * count_phrase_hits_in_text(title_text, INDUSTRIAL_OEM_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, INDUSTRIAL_OEM_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, INDUSTRIAL_OEM_STRONG_SIGNALS)
        return base_score + bonus

    if linea in ("Diseño de producto", "Desarrollo de producto"):
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, INDUSTRIAL_PDD_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, INDUSTRIAL_PDD_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, INDUSTRIAL_PDD_STRONG_SIGNALS)
        return base_score + bonus

    return base_score


def choose_best_line_with_rules(
    carrera: str,
    score_map: dict[str, int],
    text_fields: dict[str, str],
    min_score: int,
    min_margin: int = 1,
    relaxed: bool = False,
) -> str | None:
    filtered: list[tuple[str, int]] = []

    for linea, score in score_map.items():
        if score < min_score:
            continue
        if not is_line_eligible_by_domain_rules(carrera, linea, text_fields):
            continue
        adjusted_score = apply_industrial_line_bonus(carrera, linea, text_fields, score)
        filtered.append((linea, adjusted_score))

    if not filtered:
        return None

    filtered.sort(key=lambda x: (-x[1], normalize_generic_text(x[0])))

    if relaxed:
        return filtered[0][0]

    best_linea, best_score = filtered[0]
    tied = [linea for linea, score in filtered if score == best_score]
    if len(tied) > 1:
        return None

    if len(filtered) > 1:
        second_score = filtered[1][1]
        if (best_score - second_score) < min_margin:
            return None

    return best_linea



def has_climate_adaptation_evidence(text_fields: dict[str, str]) -> bool:
    if contains_any_phrase_in_text_fields(text_fields, IDIC_CLIMATE_ADAPTATION_EXACT_SIGNALS):
        return True

    climate_present = contains_any_phrase_in_text_fields(text_fields, IDIC_CLIMATE_TERMS)
    adaptation_context_present = contains_any_phrase_in_text_fields(
        text_fields,
        IDIC_ADAPTATION_CONTEXT_TERMS,
    )
    return climate_present and adaptation_context_present


def choose_non_climate_sustainability_idic_alternative(
    text_fields: dict[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    if contains_any_phrase_in_text_fields(text_fields, IDIC_CIRCULAR_ECONOMY_STRONG_SIGNALS):
        return (
            "Desarrollo sostenible y medioambiente",
            "Sostenibilidad y cambio climático",
            "Economía circular",
            "idic_guardrail_circular_economy_over_climate_adaptation",
        )

    if contains_any_phrase_in_text_fields(text_fields, IDIC_CLEAN_TECH_STRONG_SIGNALS):
        return (
            "Desarrollo sostenible y medioambiente",
            "Tecnología y ecosistemas",
            "Tecnologías limpias",
            "idic_guardrail_clean_tech_over_climate_adaptation",
        )

    if contains_any_phrase_in_text_fields(text_fields, IDIC_WASTE_MANAGEMENT_STRONG_SIGNALS):
        return (
            "Desarrollo sostenible y medioambiente",
            "Tecnología y ecosistemas",
            "Gestión de residuos",
            "idic_guardrail_waste_over_climate_adaptation",
        )

    if contains_any_phrase_in_text_fields(text_fields, IDIC_ADVANCED_MATERIALS_STRONG_SIGNALS):
        return (
            "Desarrollo sostenible y medioambiente",
            "Tecnología y ecosistemas",
            "Materiales avanzados",
            "idic_guardrail_materials_over_climate_adaptation",
        )

    if contains_any_phrase_in_text_fields(text_fields, IDIC_SUSTAINABLE_INFRASTRUCTURE_STRONG_SIGNALS):
        return (
            "Desarrollo sostenible y medioambiente",
            "Ciudades inteligentes y sostenibles",
            "Infraestructura sostenible",
            "idic_guardrail_infrastructure_over_climate_adaptation",
        )

    return None, None, None, None


def choose_general_idic_fallback(
    text_fields: dict[str, str],
) -> tuple[str, str, str, str]:
    """
    Fallback final obligatorio para evitar NULLs en IDIC.

    Esta función solo se usa cuando LLM + hints + guardrails no logran producir
    una clasificación IDIC completa. Mantiene el criterio general del proyecto:
    elegir la alternativa más cercana dentro del catálogo, sin usar
    'Adaptación al cambio climático' salvo que exista evidencia climática explícita.
    """
    alternative_category, alternative_area, alternative_line, source = choose_non_climate_sustainability_idic_alternative(text_fields)
    if alternative_category and alternative_area and alternative_line:
        return alternative_category, alternative_area, alternative_line, source or "idic_general_fallback_sustainability"

    if contains_any_phrase_in_text_fields(text_fields, IDIC_MACHINE_LEARNING_STRONG_SIGNALS):
        return (
            "Innovación y tecnología digital",
            "Inteligencia artificial y computación avanzada",
            "Machine learning y deep learning",
            "idic_general_fallback_machine_learning",
        )

    if contains_any_phrase_in_text_fields(text_fields, IDIC_EMERGING_TECH_STRONG_SIGNALS):
        return (
            "Innovación y tecnología digital",
            "Transformación digital",
            "Tecnologías emergentes",
            "idic_general_fallback_emerging_technologies",
        )

    # Último recurso institucional: nunca dejar IDIC en NULL.
    # Se elige una línea amplia, válida y menos engañosa que forzar adaptación climática.
    return (
        "Innovación y tecnología digital",
        "Transformación digital",
        "Tecnologías emergentes",
        "idic_general_fallback_default_emerging_technologies",
    )


def is_idic_line_eligible_by_domain_rules(linea: str, text_fields: dict[str, str]) -> bool:
    innovation_present = contains_any_phrase_in_text_fields(
        text_fields,
        IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS,
    )
    transformation_present = contains_any_phrase_in_text_fields(
        text_fields,
        IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS,
    )

    if linea == "Transformación organizacional" and innovation_present and not transformation_present:
        return False

    if linea == "Adaptación al cambio climático" and not has_climate_adaptation_evidence(text_fields):
        return False

    return True


def apply_idic_line_bonus(linea: str, text_fields: dict[str, str], base_score: int) -> int:
    title_text = text_fields.get("title")
    abstract_text = text_fields.get("abstract")
    keyword_text = " | ".join(
        [
            text_fields.get("author_keywords", ""),
            text_fields.get("index_keywords", ""),
        ]
    )

    if linea == "Gestión de la innovación":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Transformación organizacional":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Machine learning y deep learning":
        bonus = 0
        bonus += 6 * count_phrase_hits_in_text(title_text, IDIC_MACHINE_LEARNING_STRONG_SIGNALS)
        bonus += 3 * count_phrase_hits_in_text(keyword_text, IDIC_MACHINE_LEARNING_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(abstract_text, IDIC_MACHINE_LEARNING_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Adaptación al cambio climático":
        bonus = 0
        bonus += 5 * count_phrase_hits_in_text(title_text, IDIC_CLIMATE_ADAPTATION_EXACT_SIGNALS)
        bonus += 3 * count_phrase_hits_in_text(keyword_text, IDIC_CLIMATE_ADAPTATION_EXACT_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(abstract_text, IDIC_CLIMATE_ADAPTATION_EXACT_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(title_text, IDIC_CLIMATE_TERMS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_CLIMATE_TERMS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_ADAPTATION_CONTEXT_TERMS)
        return base_score + bonus

    if linea == "Economía circular":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_CIRCULAR_ECONOMY_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_CIRCULAR_ECONOMY_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_CIRCULAR_ECONOMY_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Tecnologías limpias":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_CLEAN_TECH_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_CLEAN_TECH_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_CLEAN_TECH_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Gestión de residuos":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_WASTE_MANAGEMENT_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_WASTE_MANAGEMENT_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_WASTE_MANAGEMENT_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Materiales avanzados":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_ADVANCED_MATERIALS_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_ADVANCED_MATERIALS_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_ADVANCED_MATERIALS_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Infraestructura sostenible":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_SUSTAINABLE_INFRASTRUCTURE_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_SUSTAINABLE_INFRASTRUCTURE_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_SUSTAINABLE_INFRASTRUCTURE_STRONG_SIGNALS)
        return base_score + bonus

    if linea == "Tecnologías emergentes":
        bonus = 0
        bonus += 4 * count_phrase_hits_in_text(title_text, IDIC_EMERGING_TECH_STRONG_SIGNALS)
        bonus += 2 * count_phrase_hits_in_text(keyword_text, IDIC_EMERGING_TECH_STRONG_SIGNALS)
        bonus += 1 * count_phrase_hits_in_text(abstract_text, IDIC_EMERGING_TECH_STRONG_SIGNALS)
        return base_score + bonus

    return base_score


def choose_best_idic_line_with_rules(
    score_map: dict[str, int],
    text_fields: dict[str, str],
    min_score: int,
    min_margin: int = 1,
    relaxed: bool = False,
) -> str | None:
    filtered: list[tuple[str, int]] = []

    for linea, score in score_map.items():
        if score < min_score:
            continue
        if not is_idic_line_eligible_by_domain_rules(linea, text_fields):
            continue
        adjusted_score = apply_idic_line_bonus(linea, text_fields, score)
        filtered.append((linea, adjusted_score))

    if not filtered:
        return None

    filtered.sort(key=lambda x: (-x[1], normalize_generic_text(x[0])))

    if relaxed:
        return filtered[0][0]

    best_linea, best_score = filtered[0]
    tied = [linea for linea, score in filtered if score == best_score]
    if len(tied) > 1:
        return None

    if len(filtered) > 1:
        second_score = filtered[1][1]
        if (best_score - second_score) < min_margin:
            return None

    return best_linea


def apply_final_idic_guardrails(
    category_tematica: str | None,
    area_idic: str | None,
    linea_idic: str | None,
    text_fields: dict[str, str],
) -> tuple[str | None, str | None, str | None, str | None]:
    if not category_tematica or not area_idic or not linea_idic:
        return category_tematica, area_idic, linea_idic, None

    innovation_present = contains_any_phrase_in_text_fields(
        text_fields,
        IDIC_INNOVATION_MANAGEMENT_STRONG_SIGNALS,
    )
    transformation_present = contains_any_phrase_in_text_fields(
        text_fields,
        IDIC_ORG_TRANSFORMATION_STRONG_SIGNALS,
    )

    if (
        category_tematica == "Gestión y economía del conocimiento"
        and area_idic == "Innovación empresarial"
        and linea_idic == "Transformación organizacional"
        and innovation_present
        and not transformation_present
    ):
        return (
            category_tematica,
            area_idic,
            "Gestión de la innovación",
            "idic_guardrail_innovation_over_transformation",
        )

    if linea_idic == "Adaptación al cambio climático" and not has_climate_adaptation_evidence(text_fields):
        alternative_category, alternative_area, alternative_line, source = choose_non_climate_sustainability_idic_alternative(
            text_fields
        )
        if alternative_category and alternative_area and alternative_line:
            return alternative_category, alternative_area, alternative_line, source

        fallback_category, fallback_area, fallback_line, fallback_source = choose_general_idic_fallback(text_fields)
        return fallback_category, fallback_area, fallback_line, fallback_source

    return category_tematica, area_idic, linea_idic, None


def append_classification_source(result: dict, source_name: str) -> None:
    current = result.get("classification_mode")
    if not current:
        result["classification_mode"] = source_name
        return

    parts = [p.strip() for p in str(current).split("|") if p.strip()]
    if source_name not in parts:
        parts.append(source_name)
    result["classification_mode"] = "|".join(parts)


def set_first_justification(result: dict, justification: str | None) -> None:
    if justification and not result.get("justification"):
        result["justification"] = justification


def classify_career_dimensions_force_best(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    line_scores: dict[str, int] = {}
    for area_name, lineas in CAREER_AREA_LINE_CATALOG[carrera].items():
        for linea in lineas:
            primary_aliases = [linea]
            support_aliases = [area_name]
            support_aliases.extend(CAREER_LINE_HINTS.get(carrera, {}).get(linea, []))
            score = build_weighted_candidate_score(
                text_fields=text_fields,
                primary_aliases=primary_aliases,
                support_aliases=support_aliases,
            )
            score = apply_industrial_line_bonus(carrera, linea, text_fields, score)
            line_scores[linea] = score

    eligible = [
        linea for linea in line_scores.keys()
        if is_line_eligible_by_domain_rules(carrera, linea, text_fields)
    ]
    candidate_pool = eligible or list(line_scores.keys())
    if not candidate_pool:
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    candidate_pool.sort(key=lambda x: (-line_scores.get(x, 0), normalize_generic_text(x)))
    best_linea = candidate_pool[0]
    best_area = coerce_area_carrera_from_linea(carrera, best_linea)
    return {"area_carrera_raw": best_area, "linea_carrera_raw": best_linea}


def classify_idic_dimensions_force_best(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    line_scores: dict[str, int] = {}
    for category_name, areas in IDIC_CATEGORY_AREA_LINE_CATALOG.items():
        for area_name, lineas in areas.items():
            for linea in lineas:
                primary_aliases = [linea]
                support_aliases = [area_name, category_name]
                support_aliases.extend(IDIC_LINE_HINTS.get(linea, []))
                base_score = build_weighted_candidate_score(
                    text_fields=text_fields,
                    primary_aliases=primary_aliases,
                    support_aliases=support_aliases,
                )
                if not is_idic_line_eligible_by_domain_rules(linea, text_fields):
                    continue
                line_scores[linea] = apply_idic_line_bonus(linea, text_fields, base_score)

    if not line_scores:
        return {
            "category_tematica_raw": None,
            "area_idic_raw": None,
            "linea_idic_raw": None,
        }

    ranked = sorted(line_scores.keys(), key=lambda x: (-line_scores.get(x, 0), normalize_generic_text(x)))
    best_linea = ranked[0]
    best_score = line_scores.get(best_linea, 0)

    if best_score < THEMATIC_APPROX_MIN_SCORE:
        fallback_category, fallback_area, fallback_line, _fallback_source = choose_general_idic_fallback(text_fields)
        return {
            "category_tematica_raw": fallback_category,
            "area_idic_raw": fallback_area,
            "linea_idic_raw": fallback_line,
        }

    best_area = coerce_area_idic_from_linea(best_linea)
    best_category = coerce_category_tematica_from_area(best_area)

    (
        best_category,
        best_area,
        best_linea,
        _guardrail_source,
    ) = apply_final_idic_guardrails(best_category, best_area, best_linea, text_fields)

    if not is_valid_idic_triplet(best_category, best_area, best_linea):
        fallback_category, fallback_area, fallback_line, _fallback_source = choose_general_idic_fallback(text_fields)
        best_category, best_area, best_linea = fallback_category, fallback_area, fallback_line

    return {
        "category_tematica_raw": best_category,
        "area_idic_raw": best_area,
        "linea_idic_raw": best_linea,
    }


def classify_career_dimensions_by_hints(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    if not any(text_fields.values()):
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    line_scores: dict[str, int] = {}
    for area_name, lineas in CAREER_AREA_LINE_CATALOG[carrera].items():
        for linea in lineas:
            primary_aliases = [linea]
            support_aliases = [area_name]
            support_aliases.extend(CAREER_LINE_HINTS.get(carrera, {}).get(linea, []))
            line_scores[linea] = build_weighted_candidate_score(
                text_fields=text_fields,
                primary_aliases=primary_aliases,
                support_aliases=support_aliases,
            )

    best_linea = choose_best_line_with_rules(
        carrera=carrera,
        score_map=line_scores,
        text_fields=text_fields,
        min_score=THEMATIC_STRICT_MIN_SCORE,
        min_margin=THEMATIC_STRICT_MIN_MARGIN,
        relaxed=False,
    )
    best_area = coerce_area_carrera_from_linea(carrera, best_linea) if best_linea else None

    return {"area_carrera_raw": best_area, "linea_carrera_raw": best_linea}


def classify_career_dimensions_by_hints_approx(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    if not any(text_fields.values()):
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    line_scores: dict[str, int] = {}
    for area_name, lineas in CAREER_AREA_LINE_CATALOG[carrera].items():
        for linea in lineas:
            primary_aliases = [linea]
            support_aliases = [area_name]
            support_aliases.extend(CAREER_LINE_HINTS.get(carrera, {}).get(linea, []))
            line_scores[linea] = build_weighted_candidate_score(
                text_fields=text_fields,
                primary_aliases=primary_aliases,
                support_aliases=support_aliases,
            )

    best_linea = choose_best_line_with_rules(
        carrera=carrera,
        score_map=line_scores,
        text_fields=text_fields,
        min_score=THEMATIC_APPROX_MIN_SCORE,
        min_margin=1,
        relaxed=True,
    )
    best_area = coerce_area_carrera_from_linea(carrera, best_linea) if best_linea else None

    return {"area_carrera_raw": best_area, "linea_carrera_raw": best_linea}


def classify_idic_dimensions_by_hints(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    if not any(text_fields.values()):
        return {
            "category_tematica_raw": None,
            "area_idic_raw": None,
            "linea_idic_raw": None,
        }

    line_scores: dict[str, int] = {}
    for category_name, areas in IDIC_CATEGORY_AREA_LINE_CATALOG.items():
        for area_name, lineas in areas.items():
            for linea in lineas:
                primary_aliases = [linea]
                support_aliases = [area_name, category_name]
                support_aliases.extend(IDIC_LINE_HINTS.get(linea, []))
                line_scores[linea] = build_weighted_candidate_score(
                    text_fields=text_fields,
                    primary_aliases=primary_aliases,
                    support_aliases=support_aliases,
                )

    best_linea = choose_best_idic_line_with_rules(
        score_map=line_scores,
        text_fields=text_fields,
        min_score=THEMATIC_STRICT_MIN_SCORE,
        min_margin=THEMATIC_STRICT_MIN_MARGIN,
        relaxed=False,
    )
    best_area = coerce_area_idic_from_linea(best_linea) if best_linea else None
    best_category = coerce_category_tematica_from_area(best_area) if best_area else None

    if best_category and best_area and best_linea:
        if not is_valid_idic_triplet(best_category, best_area, best_linea):
            best_category = None
            best_area = None
            best_linea = None

    return {
        "category_tematica_raw": best_category,
        "area_idic_raw": best_area,
        "linea_idic_raw": best_linea,
    }


def classify_idic_dimensions_by_hints_approx(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    if not any(text_fields.values()):
        return {
            "category_tematica_raw": None,
            "area_idic_raw": None,
            "linea_idic_raw": None,
        }

    line_scores: dict[str, int] = {}
    for category_name, areas in IDIC_CATEGORY_AREA_LINE_CATALOG.items():
        for area_name, lineas in areas.items():
            for linea in lineas:
                primary_aliases = [linea]
                support_aliases = [area_name, category_name]
                support_aliases.extend(IDIC_LINE_HINTS.get(linea, []))
                line_scores[linea] = build_weighted_candidate_score(
                    text_fields=text_fields,
                    primary_aliases=primary_aliases,
                    support_aliases=support_aliases,
                )

    best_linea = choose_best_idic_line_with_rules(
        score_map=line_scores,
        text_fields=text_fields,
        min_score=THEMATIC_APPROX_MIN_SCORE,
        min_margin=1,
        relaxed=True,
    )
    best_area = coerce_area_idic_from_linea(best_linea) if best_linea else None
    best_category = coerce_category_tematica_from_area(best_area) if best_area else None

    if best_category and best_area and best_linea:
        if not is_valid_idic_triplet(best_category, best_area, best_linea):
            best_category = None
            best_area = None
            best_linea = None

    return {
        "category_tematica_raw": best_category,
        "area_idic_raw": best_area,
        "linea_idic_raw": best_linea,
    }


# =========================
# AZURE OPENAI THEMATIC CLASSIFIER
# =========================
_AZURE_OPENAI_CLIENT = None


def is_thematic_llm_configured() -> bool:
    return bool(
        os.environ.get("AZURE_OPENAI_API_KEY")
        and (os.environ.get("AZURE_OPENAI_BASE_URL") or os.environ.get("AZURE_OPENAI_ENDPOINT"))
        and os.environ.get("AZURE_OPENAI_RESPONSES_MODEL")
    )


def get_azure_openai_base_url() -> str:
    explicit_base_url = os.environ.get("AZURE_OPENAI_BASE_URL")
    if explicit_base_url:
        return explicit_base_url.rstrip("/")

    endpoint = get_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    return f"{endpoint}/openai/v1"


def get_azure_openai_client():
    global _AZURE_OPENAI_CLIENT

    if _AZURE_OPENAI_CLIENT is None:
        from openai import OpenAI

        _AZURE_OPENAI_CLIENT = OpenAI(
            api_key=get_env("AZURE_OPENAI_API_KEY"),
            base_url=get_azure_openai_base_url(),
        )

    return _AZURE_OPENAI_CLIENT


def get_thematic_llm_min_confidence(mode: str = "strict") -> float:
    env_name = "THEMATIC_LLM_MIN_CONFIDENCE"
    default_value = "0.80"

    if mode == "approx":
        env_name = "THEMATIC_LLM_MIN_CONFIDENCE_APPROX"
        default_value = "0.55"

    raw = os.environ.get(env_name, default_value)
    try:
        value = float(raw)
    except Exception:
        value = float(default_value)

    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def build_thematic_empty_result() -> dict:
    return {
        "area_carrera_raw": None,
        "linea_carrera_raw": None,
        "category_tematica_raw": None,
        "area_idic_raw": None,
        "linea_idic_raw": None,
        "confidence": None,
        "justification": None,
        "classification_mode": None,
    }


def build_career_classification_prompt(
    carrera: str,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
    mode: str = "strict",
) -> str:
    career_catalog = CAREER_AREA_LINE_CATALOG.get(carrera, {})

    rules = [
        "Clasifica SOLO area_carrera_raw y linea_carrera_raw.",
        "No clasifiques campos IDIC en esta tarea.",
        "Usa abstract_scopus como evidencia principal y criterio dominante para decidir el foco real del artículo.",
        "Usa title, author_keywords e index_keywords solo como apoyo o corroboración.",
        "Usa source_title solo como evidencia débil.",
        "Debes elegir solo valores existentes en el catálogo proporcionado.",
        "No inventes áreas ni líneas.",
        "Si eliges linea_carrera_raw, debe pertenecer a la carrera dada y ser consistente con area_carrera_raw.",
        "Para Ingeniería Industrial, NO elijas 'Operations Research & Analysis' salvo que el aporte central sea modelamiento, optimización, predicción, benchmarking, simulación o algoritmos.",
        "Para Ingeniería Industrial, favorece 'Supply Chain Management' cuando el foco sea inventarios, almacenes, logística, transporte, proveedores, cadena de suministro u OTIF.",
        "Para Ingeniería Industrial, favorece 'Operations Engineering & Management' cuando el foco sea 5S, standard work, lean, mejora de procesos, eficiencia operativa o gestión de operaciones.",
        "Para Ingeniería Industrial, favorece 'Product Design & Development' cuando el foco sea síntesis, materiales, fibras, nanopartículas, formulación, caracterización o desarrollo de producto/material.",
        "Evita decidir solo por palabras amplias si el título y el resumen apuntan a un tema más específico.",
    ]

    if mode == "strict":
        rules.append("Si no hay evidencia suficiente, devuelve null en los campos inciertos.")
    else:
        rules.append("Si no hay evidencia exacta, elige la opción más cercana y defendible dentro del catálogo.")

    payload = {
        "mode": mode,
        "task": "career_only",
        "carrera": carrera,
        "career_catalog": career_catalog,
        "article": {
            "title": clip_text(title_value, 2000),
            "abstract_scopus": clip_text(abstract_value, 9000),
            "author_keywords": clip_text(author_keywords_value, 2000),
            "index_keywords": clip_text(index_keywords_value, 2000),
            "source_title": clip_text(source_title_value, 1000),
        },
        "output_schema": {
            "area_carrera_raw": "string|null",
            "linea_carrera_raw": "string|null",
            "confidence": "number 0..1",
            "justification": "short string",
        },
        "rules": rules,
    }

    return (
        "Eres un clasificador temático académico. Analiza el artículo y clasifica solo la dimensión Carrera.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_idic_classification_prompt(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
    mode: str = "strict",
) -> str:
    rules = [
        "Clasifica SOLO category_tematica_raw, area_idic_raw y linea_idic_raw.",
        "No clasifiques campos de Carrera en esta tarea.",
        "Usa abstract_scopus como evidencia principal y criterio dominante para decidir el foco real del artículo.",
        "Usa title, author_keywords e index_keywords solo como apoyo o corroboración.",
        "Usa source_title solo como evidencia débil.",
        "Debes elegir solo valores existentes en el catálogo proporcionado.",
        "No inventes categorías, áreas ni líneas.",
        "Si eliges linea_idic_raw, debe ser consistente con area_idic_raw y category_tematica_raw.",
        "Dentro de 'Gestión y economía del conocimiento' > 'Innovación empresarial', favorece 'Gestión de la innovación' cuando el foco sea 5S, standard work, lean, mejora continua, mejora de procesos, eficiencia operativa o productividad.",
        "Dentro de 'Gestión y economía del conocimiento' > 'Innovación empresarial', usa 'Transformación organizacional' solo cuando el aporte central trate explícitamente de transformación, rediseño o cambio organizacional.",
        "Distingue claramente entre enfoques de tecnología digital, sostenibilidad y comportamiento humano según el aporte central del artículo.",
        "Usa 'Adaptación al cambio climático' solo si el artículo menciona explícitamente cambio climático, adaptación, resiliencia, vulnerabilidad, riesgo climático o impactos climáticos.",
        "Si el artículo trata materiales, síntesis, nanopartículas, concreto, adsorción, valorización de residuos o tecnologías limpias sin evidencia climática explícita, no uses 'Adaptación al cambio climático'.",
        "Evita decidir solo por palabras amplias si el título y el resumen apuntan a un tema más específico.",
    ]

    if mode == "strict":
        rules.append("Si no hay evidencia suficiente, devuelve null en los campos inciertos.")
    else:
        rules.append("Si no hay evidencia exacta, elige la opción más cercana y defendible dentro del catálogo.")

    payload = {
        "mode": mode,
        "task": "idic_only",
        "idic_catalog": IDIC_CATEGORY_AREA_LINE_CATALOG,
        "article": {
            "title": clip_text(title_value, 2000),
            "abstract_scopus": clip_text(abstract_value, 9000),
            "author_keywords": clip_text(author_keywords_value, 2000),
            "index_keywords": clip_text(index_keywords_value, 2000),
            "source_title": clip_text(source_title_value, 1000),
        },
        "output_schema": {
            "category_tematica_raw": "string|null",
            "area_idic_raw": "string|null",
            "linea_idic_raw": "string|null",
            "confidence": "number 0..1",
            "justification": "short string",
        },
        "rules": rules,
    }

    return (
        "Eres un clasificador temático académico. Analiza el artículo y clasifica solo la dimensión IDIC.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def validate_llm_career_output(
    carrera: str | None,
    raw_output: dict | None,
    mode: str = "strict",
) -> dict:
    result = build_thematic_empty_result()

    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG or not raw_output:
        return result

    confidence = parse_float_or_none(raw_output.get("confidence"))
    result["confidence"] = confidence
    result["justification"] = raw_output.get("justification")
    result["classification_mode"] = f"career_llm_{mode}"

    area_carrera = coerce_choice(raw_output.get("area_carrera_raw"), get_allowed_career_areas(carrera))
    linea_carrera = coerce_choice(raw_output.get("linea_carrera_raw"), get_allowed_career_lines(carrera))

    if linea_carrera and not area_carrera:
        area_carrera = coerce_area_carrera_from_linea(carrera, linea_carrera)

    if not is_valid_career_area_line(carrera, area_carrera, linea_carrera):
        area_carrera = None
        linea_carrera = None

    min_confidence = get_thematic_llm_min_confidence(mode=mode)
    if confidence is not None and confidence < min_confidence:
        return build_thematic_empty_result()

    result["area_carrera_raw"] = area_carrera
    result["linea_carrera_raw"] = linea_carrera
    return result


def validate_llm_idic_output(
    raw_output: dict | None,
    mode: str = "strict",
) -> dict:
    result = build_thematic_empty_result()

    if not raw_output:
        return result

    confidence = parse_float_or_none(raw_output.get("confidence"))
    result["confidence"] = confidence
    result["justification"] = raw_output.get("justification")
    result["classification_mode"] = f"idic_llm_{mode}"

    category_tematica = coerce_choice(raw_output.get("category_tematica_raw"), get_allowed_idic_categories())
    area_idic = coerce_choice(raw_output.get("area_idic_raw"), get_allowed_idic_areas(category_tematica))

    if not area_idic:
        area_idic = coerce_choice(raw_output.get("area_idic_raw"), get_allowed_idic_areas())

    linea_idic = coerce_choice(raw_output.get("linea_idic_raw"), get_allowed_idic_lines(category_tematica, area_idic))

    if not linea_idic:
        linea_idic = coerce_choice(raw_output.get("linea_idic_raw"), get_allowed_idic_lines())

    if linea_idic and not area_idic:
        area_idic = coerce_area_idic_from_linea(linea_idic)

    if area_idic and not category_tematica:
        category_tematica = coerce_category_tematica_from_area(area_idic)

    if not is_valid_idic_triplet(category_tematica, area_idic, linea_idic):
        category_tematica = None
        area_idic = None
        linea_idic = None

    min_confidence = get_thematic_llm_min_confidence(mode=mode)
    if confidence is not None and confidence < min_confidence:
        return build_thematic_empty_result()

    result["category_tematica_raw"] = category_tematica
    result["area_idic_raw"] = area_idic
    result["linea_idic_raw"] = linea_idic
    return result


def classify_career_with_llm(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
    mode: str = "strict",
) -> dict:
    empty_result = build_thematic_empty_result()

    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return empty_result

    if not is_thematic_llm_configured():
        return empty_result

    prompt = build_career_classification_prompt(
        carrera=carrera,
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
        mode=mode,
    )

    try:
        client = get_azure_openai_client()
        response = client.responses.create(
            model=get_env("AZURE_OPENAI_RESPONSES_MODEL"),
            input=prompt,
        )

        raw_output = extract_json_object(getattr(response, "output_text", None))
        if not raw_output:
            logging.warning("Career LLM (%s) returned non-JSON output.", mode)
            return empty_result

        return validate_llm_career_output(carrera, raw_output, mode=mode)

    except Exception as exc:
        logging.warning("Career LLM classification failed (%s): %s", mode, str(exc))
        return empty_result


def classify_idic_with_llm(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
    mode: str = "strict",
) -> dict:
    empty_result = build_thematic_empty_result()

    if not is_thematic_llm_configured():
        return empty_result

    prompt = build_idic_classification_prompt(
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
        mode=mode,
    )

    try:
        client = get_azure_openai_client()
        response = client.responses.create(
            model=get_env("AZURE_OPENAI_RESPONSES_MODEL"),
            input=prompt,
        )

        raw_output = extract_json_object(getattr(response, "output_text", None))
        if not raw_output:
            logging.warning("IDIC LLM (%s) returned non-JSON output.", mode)
            return empty_result

        return validate_llm_idic_output(raw_output, mode=mode)

    except Exception as exc:
        logging.warning("IDIC LLM classification failed (%s): %s", mode, str(exc))
        return empty_result


def classify_thematic_fields(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    merged = build_thematic_empty_result()

    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return merged

    career_fields = ["area_carrera_raw", "linea_carrera_raw"]
    idic_fields = ["category_tematica_raw", "area_idic_raw", "linea_idic_raw"]

    # -------------------------
    # A. CARRERA
    # -------------------------
    career_strict = classify_career_with_llm(
        carrera=carrera,
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
        mode="strict",
    )
    merged = merge_non_null_fields(merged, career_strict, career_fields)
    if any(career_strict.get(field) for field in career_fields):
        append_classification_source(merged, "career_llm_strict")
        if career_strict.get("confidence") is not None and merged.get("confidence") is None:
            merged["confidence"] = career_strict.get("confidence")
        set_first_justification(merged, career_strict.get("justification"))

    if any(not merged.get(field) for field in career_fields):
        career_approx = classify_career_with_llm(
            carrera=carrera,
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
            mode="approx",
        )
        merged = merge_non_null_fields(merged, career_approx, career_fields)
        if any(career_approx.get(field) for field in career_fields):
            append_classification_source(merged, "career_llm_approx")
            if career_approx.get("confidence") is not None and merged.get("confidence") is None:
                merged["confidence"] = career_approx.get("confidence")
            set_first_justification(merged, career_approx.get("justification"))

    if any(not merged.get(field) for field in career_fields):
        career_hint = classify_career_dimensions_by_hints(
            carrera=carrera,
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, career_hint, career_fields)
        if any(career_hint.get(field) for field in career_fields):
            append_classification_source(merged, "career_hints_strict")

    if any(not merged.get(field) for field in career_fields):
        career_hint_approx = classify_career_dimensions_by_hints_approx(
            carrera=carrera,
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, career_hint_approx, career_fields)
        if any(career_hint_approx.get(field) for field in career_fields):
            append_classification_source(merged, "career_hints_approx")

    # -------------------------
    # B. IDIC
    # -------------------------
    idic_strict = classify_idic_with_llm(
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
        mode="strict",
    )
    merged = merge_non_null_fields(merged, idic_strict, idic_fields)
    if any(idic_strict.get(field) for field in idic_fields):
        append_classification_source(merged, "idic_llm_strict")
        if idic_strict.get("confidence") is not None and merged.get("confidence") is None:
            merged["confidence"] = idic_strict.get("confidence")
        set_first_justification(merged, idic_strict.get("justification"))

    if any(not merged.get(field) for field in idic_fields):
        idic_approx = classify_idic_with_llm(
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
            mode="approx",
        )
        merged = merge_non_null_fields(merged, idic_approx, idic_fields)
        if any(idic_approx.get(field) for field in idic_fields):
            append_classification_source(merged, "idic_llm_approx")
            if idic_approx.get("confidence") is not None and merged.get("confidence") is None:
                merged["confidence"] = idic_approx.get("confidence")
            set_first_justification(merged, idic_approx.get("justification"))

    if any(not merged.get(field) for field in idic_fields):
        idic_hint = classify_idic_dimensions_by_hints(
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, idic_hint, idic_fields)
        if any(idic_hint.get(field) for field in idic_fields):
            append_classification_source(merged, "idic_hints_strict")

    if any(not merged.get(field) for field in idic_fields):
        idic_hint_approx = classify_idic_dimensions_by_hints_approx(
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, idic_hint_approx, idic_fields)
        if any(idic_hint_approx.get(field) for field in idic_fields):
            append_classification_source(merged, "idic_hints_approx")

    text_fields = build_thematic_text_fields(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )

    (
        merged["category_tematica_raw"],
        merged["area_idic_raw"],
        merged["linea_idic_raw"],
        idic_guardrail_source,
    ) = apply_final_idic_guardrails(
        merged.get("category_tematica_raw"),
        merged.get("area_idic_raw"),
        merged.get("linea_idic_raw"),
        text_fields,
    )

    if idic_guardrail_source:
        append_classification_source(merged, idic_guardrail_source)

    # Validación final
    if not is_valid_career_area_line(
        carrera,
        merged.get("area_carrera_raw"),
        merged.get("linea_carrera_raw"),
    ):
        merged["area_carrera_raw"] = None
        merged["linea_carrera_raw"] = None

    if not is_valid_idic_triplet(
        merged.get("category_tematica_raw"),
        merged.get("area_idic_raw"),
        merged.get("linea_idic_raw"),
    ):
        merged["category_tematica_raw"] = None
        merged["area_idic_raw"] = None
        merged["linea_idic_raw"] = None

    # Fallback final obligatorio: no dejar NULLs temáticos
    if any(not merged.get(field) for field in career_fields):
        career_forced = classify_career_dimensions_force_best(
            carrera=carrera,
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, career_forced, career_fields)
        if any(career_forced.get(field) for field in career_fields):
            append_classification_source(merged, "career_force_best")
            set_first_justification(merged, "Fallback final obligatorio de carrera para evitar NULLs.")

    if any(not merged.get(field) for field in idic_fields):
        idic_forced = classify_idic_dimensions_force_best(
            title_value=title_value,
            abstract_value=abstract_value,
            author_keywords_value=author_keywords_value,
            index_keywords_value=index_keywords_value,
            source_title_value=source_title_value,
        )
        merged = merge_non_null_fields(merged, idic_forced, idic_fields)
        if any(idic_forced.get(field) for field in idic_fields):
            append_classification_source(merged, "idic_force_best")
            set_first_justification(merged, "Fallback final obligatorio de IDIC para evitar NULLs.")

    # Safety net: por regla del proyecto, ningún registro aceptado debe salir con IDIC en NULL.
    if any(not merged.get(field) for field in idic_fields):
        fallback_category, fallback_area, fallback_line, fallback_source = choose_general_idic_fallback(text_fields)
        merged["category_tematica_raw"] = fallback_category
        merged["area_idic_raw"] = fallback_area
        merged["linea_idic_raw"] = fallback_line
        append_classification_source(merged, fallback_source)
        set_first_justification(merged, "Fallback institucional obligatorio de IDIC para evitar NULLs.")

    return merged


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
    source_file_path: str,
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
                0,
            ),
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
    error_message: str | None = None,
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
                run_id,
            ),
        )
        conn.commit()
        cursor.close()


def insert_pipeline_run_smoke(trigger_type: str, status: str, source_name: str = "FUNCTION_SMOKE") -> int:
    return create_pipeline_run(
        trigger_type=trigger_type,
        source_name=source_name,
        source_file_name="N/A",
        source_file_path="N/A",
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
                values_only=True,
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
                    values_only=True,
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
                ),
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
            (periodo_academico,),
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
                ),
            )

        conn.commit()
        cursor.close()

    return len(deduped_rows)


# =========================
# DOCENTES REF MATCHING / NORMALIZATION
# =========================
def build_docente_display_name(docente: dict) -> str:
    family = " ".join([x for x in [docente.get("apellido_1"), docente.get("apellido_2")] if x])
    names = docente.get("nombres")
    if family and names:
        return f"{family.title()}, {names.title()}"
    return (docente.get("nombre_original") or "").replace("/", " ").title()


def first_given_name(nombres: str | None) -> str:
    tokens = [t for t in normalize_generic_text(nombres).split() if t]
    return tokens[0] if tokens else ""


def initials_variants_from_tokens(tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    initials = "".join(token[0] for token in tokens if token)
    variants = {initials}
    if initials:
        variants.add(initials[0])
    return {v for v in variants if v}


def prepare_docente_reference_entry(docente: dict) -> dict:
    apellido_1_norm = normalize_generic_text(docente.get("apellido_1"))
    apellido_2_norm = normalize_generic_text(docente.get("apellido_2"))
    nombres_norm = normalize_generic_text(docente.get("nombres"))
    nombre_normalizado = normalize_person_name(docente.get("nombre_normalizado") or docente.get("nombre_original"))

    family_signature = " ".join([x for x in [apellido_1_norm, apellido_2_norm] if x]).strip()
    first_name = first_given_name(docente.get("nombres"))
    initials_full = normalize_generic_text(docente.get("iniciales"))

    strong_aliases: set[str] = set()
    weak_aliases: set[str] = set()

    if nombre_normalizado:
        strong_aliases.add(nombre_normalizado)

    initials_variants = {initials_full}
    if initials_full:
        initials_variants.add(initials_full[:1])

    if family_signature:
        for iv in initials_variants:
            if iv:
                strong_aliases.add(f"{family_signature} {iv}".strip())
        if first_name:
            strong_aliases.add(f"{family_signature} {first_name}".strip())

    if apellido_1_norm:
        for iv in initials_variants:
            if iv:
                weak_aliases.add(f"{apellido_1_norm} {iv}".strip())
        if first_name:
            weak_aliases.add(f"{apellido_1_norm} {first_name}".strip())

    docente_prepared = dict(docente)
    docente_prepared["apellido_1_norm"] = apellido_1_norm
    docente_prepared["apellido_2_norm"] = apellido_2_norm
    docente_prepared["nombres_norm"] = nombres_norm
    docente_prepared["family_signature"] = family_signature
    docente_prepared["first_name_norm"] = first_name
    docente_prepared["strong_aliases"] = strong_aliases
    docente_prepared["weak_aliases"] = weak_aliases

    return docente_prepared


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
            (periodo_academico,),
        )

        rows = cursor.fetchall()
        cursor.close()

    for row in rows:
        docente = {
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
        docentes.append(prepare_docente_reference_entry(docente))

    return docentes


def split_authors_with_affiliations_blocks(
    author_full_names_value: str | None,
    authors_with_affiliations_value: str | None,
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


def extract_author_name_from_block(block: str | None) -> str | None:
    if not block:
        return None

    first_part = str(block).split(",", 1)[0].strip()
    if not first_part:
        return None

    return clean_author_full_name(first_part)


def get_preferred_author_name_for_block(
    idx: int,
    full_authors: list[str],
    short_authors: list[str],
    block: str | None,
) -> str | None:
    if idx < len(full_authors) and full_authors[idx]:
        return clean_author_full_name(full_authors[idx])

    if idx < len(short_authors) and short_authors[idx]:
        return clean_author_full_name(short_authors[idx])

    return extract_author_name_from_block(block)


def parse_scopus_author_name(scopus_author_name: str) -> dict:
    raw = clean_author_full_name(scopus_author_name)

    family_tokens: list[str] = []
    given_tokens: list[str] = []

    if "," in raw:
        family_part, given_part = [p.strip() for p in raw.split(",", 1)]
        family_tokens = [t for t in normalize_generic_text(family_part).split() if t]
        given_tokens = [t for t in normalize_generic_text(given_part.replace(".", " ")).split() if t]
    else:
        tokens = [t.strip() for t in raw.replace(".", " ").split() if t.strip()]
        normalized_tokens = [normalize_generic_text(t) for t in tokens]

        trailing_initials: list[str] = []
        idx = len(normalized_tokens) - 1
        while idx >= 0 and len(normalized_tokens[idx]) == 1:
            trailing_initials.insert(0, normalized_tokens[idx])
            idx -= 1

        if trailing_initials and idx >= 0:
            family_tokens = normalized_tokens[: idx + 1]
            given_tokens = trailing_initials
        elif len(normalized_tokens) >= 2:
            family_tokens = normalized_tokens[:-1]
            given_tokens = [normalized_tokens[-1]]
        else:
            family_tokens = normalized_tokens
            given_tokens = []

    apellido_1 = family_tokens[0] if family_tokens else None
    apellido_2 = " ".join(family_tokens[1:]) if len(family_tokens) > 1 else None
    initials = "".join(token[0] for token in given_tokens if token)
    normalized_full = normalize_person_name(raw)
    family_signature = " ".join(family_tokens).strip()
    first_name = given_tokens[0] if given_tokens else ""

    return {
        "raw": raw,
        "apellido_1": apellido_1,
        "apellido_2": apellido_2,
        "family_tokens": family_tokens,
        "family_signature": family_signature,
        "given_tokens": given_tokens,
        "initials": initials,
        "first_name": first_name,
        "normalized_full": normalized_full,
    }


def build_scopus_author_aliases(parsed: dict) -> dict:
    strong_aliases: set[str] = set()
    weak_aliases: set[str] = set()

    if parsed["normalized_full"]:
        strong_aliases.add(parsed["normalized_full"])

    initials_variants = initials_variants_from_tokens(parsed["given_tokens"])

    if parsed["family_signature"]:
        for iv in initials_variants:
            strong_aliases.add(f"{parsed['family_signature']} {iv}".strip())
        if parsed["first_name"]:
            strong_aliases.add(f"{parsed['family_signature']} {parsed['first_name']}".strip())

    if parsed["apellido_1"]:
        for iv in initials_variants:
            weak_aliases.add(f"{parsed['apellido_1']} {iv}".strip())
        if parsed["first_name"]:
            weak_aliases.add(f"{parsed['apellido_1']} {parsed['first_name']}".strip())

    return {
        "strong_aliases": {a for a in strong_aliases if a},
        "weak_aliases": {a for a in weak_aliases if a},
    }


def given_names_match(ref_nombres: str | None, scopus_given_tokens: list[str], scopus_initials: str) -> bool:
    ref_tokens = [t for t in normalize_generic_text(ref_nombres).split() if t]

    if not scopus_given_tokens and not scopus_initials:
        return True

    if scopus_initials and (not scopus_given_tokens or all(len(tok) == 1 for tok in scopus_given_tokens)):
        ref_initials = "".join(token[0] for token in ref_tokens if token)
        return ref_initials.startswith(scopus_initials)

    if scopus_given_tokens:
        if len(ref_tokens) < len(scopus_given_tokens):
            return False
        for idx, token in enumerate(scopus_given_tokens):
            if ref_tokens[idx] != token:
                return False
        return True

    return False


def filter_candidates_by_career_hint(candidates: list[dict], career_hint: str | None) -> list[dict]:
    if not career_hint:
        return candidates
    filtered = [c for c in candidates if c.get("carrera") == career_hint]
    return filtered if filtered else candidates


def unique_alias_match(
    scopus_aliases: set[str],
    docentes_ref: list[dict],
    alias_key: str,
    career_hint: str | None = None,
) -> dict:
    candidates = []
    for docente in docentes_ref:
        if scopus_aliases.intersection(docente.get(alias_key, set())):
            candidates.append(docente)

    candidates = filter_candidates_by_career_hint(candidates, career_hint)

    if len(candidates) == 1:
        return {"matched": True, "ambiguous": False, "docente": candidates[0]}

    if len(candidates) > 1:
        return {"matched": False, "ambiguous": True, "docente": None}

    return {"matched": False, "ambiguous": False, "docente": None}


def match_scopus_author_to_docente(
    scopus_author_name: str,
    docentes_ref: list[dict],
    career_hint: str | None = None,
) -> dict:
    parsed = parse_scopus_author_name(scopus_author_name)
    scopus_aliases = build_scopus_author_aliases(parsed)

    exact_matches = [d for d in docentes_ref if d["nombre_normalizado"] == parsed["normalized_full"]]
    exact_matches = filter_candidates_by_career_hint(exact_matches, career_hint)

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

    structured_matches = []
    for docente in docentes_ref:
        ref_ap1 = docente.get("apellido_1_norm")
        ref_ap2 = docente.get("apellido_2_norm")

        if parsed["apellido_1"] and ref_ap1 != parsed["apellido_1"]:
            continue
        if parsed["apellido_2"] and ref_ap2 != parsed["apellido_2"]:
            continue
        if not parsed["apellido_1"]:
            continue
        if not given_names_match(docente.get("nombres"), parsed["given_tokens"], parsed["initials"]):
            continue

        structured_matches.append(docente)

    structured_matches = filter_candidates_by_career_hint(structured_matches, career_hint)

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

    strong_match = unique_alias_match(
        scopus_aliases=scopus_aliases["strong_aliases"],
        docentes_ref=docentes_ref,
        alias_key="strong_aliases",
        career_hint=career_hint,
    )

    if strong_match["matched"]:
        return {
            "matched": True,
            "ambiguous": False,
            "match_method": "DOCENTES_REF_ALIAS_STRONG",
            "docente": strong_match["docente"],
        }

    if strong_match["ambiguous"]:
        return {
            "matched": False,
            "ambiguous": True,
            "match_method": "DOCENTES_REF_ALIAS_STRONG_AMBIGUOUS",
            "docente": None,
        }

    return {
        "matched": False,
        "ambiguous": False,
        "match_method": "DOCENTES_REF_NO_MATCH",
        "docente": None,
    }


# =========================
# AFFILIATION-FIRST + AUTHOR FALLBACK
# =========================
def get_publication_level_engineering_careers(affiliations_value: str | None) -> list[str]:
    publication_details = resolve_engineering_affiliation_details(affiliations_value)
    return filter_valid_engineering_careers(publication_details.get("careers"))



def get_block_engineering_careers(
    block_value: str | None,
    publication_level_careers: list[str],
) -> list[str]:
    block_details = resolve_engineering_affiliation_details(block_value)
    block_careers = filter_valid_engineering_careers(block_details.get("careers"))
    if block_careers:
        return block_careers

    if block_details.get("has_ulima_engineering_context") and len(publication_level_careers) == 1:
        return publication_level_careers

    return []



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
    if not author_blocks and affiliations_value:
        author_blocks = split_semicolon_values(affiliations_value)
        if not author_blocks:
            author_blocks = [str(affiliations_value)]

    publication_details = resolve_engineering_affiliation_details(affiliations_value)
    publication_level_careers = filter_valid_engineering_careers(publication_details.get("careers"))

    ulima_authors_detected: list[str] = []
    publication_careers: list[str] = []
    first_author_ulima = False
    matched_any = False
    affiliation_ulima_detected = False
    has_ulima_engineering_affiliation = False

    for idx, block in enumerate(author_blocks):
        block_details = resolve_engineering_affiliation_details(block)
        if not block_details.get("has_ulima_affiliation"):
            continue

        affiliation_ulima_detected = True
        if block_details.get("has_ulima_engineering_context"):
            has_ulima_engineering_affiliation = True

        preferred_author_name = get_preferred_author_name_for_block(
            idx=idx,
            full_authors=full_authors,
            short_authors=short_authors,
            block=block,
        )
        if preferred_author_name:
            ulima_authors_detected.append(preferred_author_name)

        if idx == 0:
            first_author_ulima = True

        block_careers = get_block_engineering_careers(
            block_value=block,
            publication_level_careers=publication_level_careers,
        )

        if idx < len(full_authors):
            scopus_author_name = full_authors[idx]
        elif idx < len(short_authors):
            scopus_author_name = short_authors[idx]
        else:
            scopus_author_name = extract_author_name_from_block(block) or ""

        resolved_career_hint = block_careers[0] if len(block_careers) == 1 else None

        match_result = match_scopus_author_to_docente(
            scopus_author_name=scopus_author_name,
            docentes_ref=docentes_ref,
            career_hint=resolved_career_hint,
        )

        if match_result["matched"]:
            matched_any = True
            docente = match_result["docente"]

            if (
                not block_careers
                and block_details.get("has_ulima_engineering_context")
                and docente.get("carrera") in VALID_ENGINEERING_CAREERS
            ):
                block_careers = [docente.get("carrera")]

        publication_careers.extend(filter_valid_engineering_careers(block_careers))

    if not publication_careers and publication_level_careers and publication_details.get("has_ulima_engineering_context"):
        publication_careers.extend(publication_level_careers)
        has_ulima_engineering_affiliation = True

    publication_careers = unique_keep_order([c for c in publication_careers if c])
    ulima_authors_detected = unique_keep_order([a for a in ulima_authors_detected if a])

    if matched_any and publication_careers:
        metodo = "AFFILIATION+DOCENTES_REF"
    elif publication_careers:
        metodo = "AFFILIATION_ONLY"
    elif matched_any:
        metodo = "DOCENTES_REF_ONLY"
    else:
        metodo = None

    return {
        "ulima_docentes_raw": "; ".join(ulima_authors_detected) if ulima_authors_detected else None,
        "first_author_ulima_raw": "True" if first_author_ulima else "False",
        "carrera_raw": "; ".join(publication_careers) if publication_careers else None,
        "metodo_cruce_scopus_raw": metodo,
        "es_ulima_raw_detected": affiliation_ulima_detected,
        "has_ulima_engineering_affiliation_raw": has_ulima_engineering_affiliation,
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
    "revista_raw": ["Revista"],
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
    "retractado_raw": ["Retractado"],
    "descontinuado_scopus_raw": ["Descontinuado de SCOPUS", "Descontinuado de Scopus"],
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
    source_title_value = safe_get(row, ["Source title", "Source Title"])
    document_type_value = safe_get(row, ["Document Type", "Document Type (Scopus)", "Source & document type"])
    publication_stage_value = safe_get(row, ["Publication Stage", "Publication Status"])
    publisher_value = safe_get(row, ["Publisher"])
    doi_value = safe_get(row, ["DOI /link", "DOI"])

    enrichment = enrich_ulima_fields_from_ref(
        authors_value=authors_value,
        author_full_names_value=author_full_names_value,
        authors_with_affiliations_value=authors_with_affiliations_value,
        affiliations_value=affiliations_value,
        docentes_ref=docentes_ref,
    )

    authors_display = build_authors_display(author_full_names_value, authors_value)
    conference_journal_value = derive_conference_journal_value(document_type_value)
    revista_value, conference_value = derive_revista_and_conference(source_title_value, document_type_value)
    alternative_link_value = build_doi_url(doi_value)

    mapped["authors_raw"] = authors_display or mapped.get("authors_raw")
    mapped["affiliation_raw"] = authors_with_affiliations_value or affiliations_value or mapped.get("affiliation_raw")
    mapped["ulima_docentes_raw"] = enrichment["ulima_docentes_raw"]
    mapped["first_author_ulima_raw"] = enrichment["first_author_ulima_raw"]

    if enrichment["carrera_raw"]:
        mapped["carrera_raw"] = enrichment["carrera_raw"]

    if not mapped.get("conference_journal_raw"):
        mapped["conference_journal_raw"] = conference_journal_value

    mapped["revista_raw"] = revista_value
    mapped["conference_raw"] = conference_value

    if not mapped.get("editorial_publication_raw"):
        mapped["editorial_publication_raw"] = publisher_value

    if not mapped.get("publication_status_raw"):
        mapped["publication_status_raw"] = publication_stage_value

    if not mapped.get("source_title_raw"):
        mapped["source_title_raw"] = source_title_value

    if not mapped.get("document_type_scopus_raw"):
        mapped["document_type_scopus_raw"] = document_type_value

    if not mapped.get("doi_link_raw"):
        mapped["doi_link_raw"] = normalize_url_or_doi(doi_value)

    if not mapped.get("alternative_link_raw"):
        mapped["alternative_link_raw"] = alternative_link_value

    if not mapped.get("indexation_raw"):
        mapped["indexation_raw"] = "Scopus"

    if not mapped.get("es_scopus_raw"):
        mapped["es_scopus_raw"] = "True"

    if not mapped.get("retractado_raw"):
        mapped["retractado_raw"] = "No"

    if not mapped.get("descontinuado_scopus_raw"):
        mapped["descontinuado_scopus_raw"] = "No"

    if enrichment["metodo_cruce_scopus_raw"]:
        mapped["metodo_cruce_scopus_raw"] = enrichment["metodo_cruce_scopus_raw"]
    elif mapped.get("eid") and not mapped.get("metodo_cruce_scopus_raw"):
        mapped["metodo_cruce_scopus_raw"] = "EID"

    eligibility = determine_row_engineering_eligibility(
        document_type_value=document_type_value,
        authors_with_affiliations_value=authors_with_affiliations_value,
        affiliations_value=affiliations_value,
        enrichment=enrichment,
    )

    if eligibility.get("carrera_raw"):
        mapped["carrera_raw"] = eligibility.get("carrera_raw")

    if not eligibility.get("eligible"):
        mapped["area_carrera_raw"] = None
        mapped["linea_carrera_raw"] = None
        mapped["category_tematica_raw"] = None
        mapped["area_idic_raw"] = None
        mapped["linea_idic_raw"] = None
        mapped = sanitize_identifier_fields(mapped)
        mapped["record_hash"] = compute_record_hash(
            mapped.get("eid"),
            mapped.get("doi_link_raw"),
            mapped.get("publication_title_raw"),
        )
        mapped["is_valid_for_curated"] = 0
        mapped["rejection_reason"] = eligibility.get("reason")
        mapped["__skip_insert__"] = True
        return mapped

    # -------------------------
    # Thematic classification
    # -------------------------
    carrera_for_classification = mapped.get("carrera_raw")
    if carrera_for_classification and ";" in str(carrera_for_classification):
        carrera_for_classification = str(carrera_for_classification).split(";", 1)[0].strip()

    if carrera_for_classification:
        thematic = classify_thematic_fields(
            carrera=carrera_for_classification,
            title_value=mapped.get("publication_title_raw"),
            abstract_value=mapped.get("abstract_scopus_raw"),
            author_keywords_value=mapped.get("author_keywords_raw"),
            index_keywords_value=mapped.get("index_keywords_raw"),
            source_title_value=mapped.get("source_title_raw"),
        )

        if not mapped.get("area_carrera_raw") and thematic.get("area_carrera_raw"):
            mapped["area_carrera_raw"] = thematic["area_carrera_raw"]

        if not mapped.get("linea_carrera_raw") and thematic.get("linea_carrera_raw"):
            mapped["linea_carrera_raw"] = thematic["linea_carrera_raw"]

        if not mapped.get("category_tematica_raw") and thematic.get("category_tematica_raw"):
            mapped["category_tematica_raw"] = thematic["category_tematica_raw"]

        if not mapped.get("area_idic_raw") and thematic.get("area_idic_raw"):
            mapped["area_idic_raw"] = thematic["area_idic_raw"]

        if not mapped.get("linea_idic_raw") and thematic.get("linea_idic_raw"):
            mapped["linea_idic_raw"] = thematic["linea_idic_raw"]

    if not is_valid_career_area_line(
        carrera_for_classification,
        mapped.get("area_carrera_raw"),
        mapped.get("linea_carrera_raw"),
    ):
        mapped["area_carrera_raw"] = None
        mapped["linea_carrera_raw"] = None

    if not is_valid_idic_triplet(
        mapped.get("category_tematica_raw"),
        mapped.get("area_idic_raw"),
        mapped.get("linea_idic_raw"),
    ):
        mapped["category_tematica_raw"] = None
        mapped["area_idic_raw"] = None
        mapped["linea_idic_raw"] = None

    mapped = sanitize_identifier_fields(mapped)

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
    mapped["__skip_insert__"] = False

    return mapped


def parse_csv_text(csv_text: str) -> list[dict]:
    normalized_text = (csv_text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized_text, hinted_delimiter = strip_excel_separator_hint(normalized_text)

    candidate_delimiters = [hinted_delimiter] if hinted_delimiter else []
    for delimiter in CSV_DELIMITER_CANDIDATES:
        if delimiter not in candidate_delimiters:
            candidate_delimiters.append(delimiter)

    best_rows: list[list[str]] | None = None
    best_score: int | None = None
    best_delimiter: str | None = None

    for delimiter in candidate_delimiters:
        try:
            parsed_rows = parse_csv_rows_with_delimiter(normalized_text, delimiter)
        except Exception:
            continue

        candidate_score = score_csv_candidate(parsed_rows)
        if best_score is None or candidate_score > best_score:
            best_rows = parsed_rows
            best_score = candidate_score
            best_delimiter = delimiter

    if not best_rows:
        raise RuntimeError("Unable to parse CSV with supported delimiters.")

    logging.info(
        "CSV parsed using delimiter %r with score %s and %s data rows.",
        best_delimiter,
        best_score,
        max(len(best_rows) - 1, 0),
    )

    return build_dict_rows_from_csv_rows(best_rows)


def insert_rows_to_staging(
    run_id: int,
    source_file_name: str,
    rows: list[dict],
    docentes_ref: list[dict],
) -> tuple[int, int]:
    from mssql_python import connect

    sql_conn_str = get_sql_connection_string()
    inserted = 0
    rejected = 0

    with connect(sql_conn_str) as conn:
        cursor = conn.cursor()

        for idx, row in enumerate(rows, start=1):
            mapped = map_row_to_staging(row, docentes_ref=docentes_ref)

            if mapped.get("__skip_insert__"):
                rejected += 1
                continue

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
                    retractado_raw,
                    descontinuado_scopus_raw,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    mapped.get("retractado_raw"),
                    mapped.get("descontinuado_scopus_raw"),
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
                ),
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
            "thematic_llm_configured": is_thematic_llm_configured(),
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
            source_file_path=f"{docentes_container}/{blob_name}",
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

        processed_name = f"docentes/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{os.path.basename(blob_name)}"
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
            error_message=None,
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
                error_message=str(exc),
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
            source_file_path=f"{raw_container}/{blob_name}",
        )

        csv_text = download_blob_text(raw_container, blob_name)
        rows = parse_csv_text(csv_text)

        records_read = len(rows)
        records_inserted, records_rejected = insert_rows_to_staging(
            run_id=run_id,
            source_file_name=blob_name,
            rows=rows,
            docentes_ref=docentes_ref,
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
            error_message=None,
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
                error_message=str(exc),
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
