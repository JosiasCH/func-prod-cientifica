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


def clip_text(value: str | None, max_chars: int = 7000) -> str | None:
    if not value:
        return None
    value = str(value).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars]


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
        "Diseño de sistemas de trabajo": ["diseno de sistemas de trabajo", "work design", "work system design"],
        "Evaluación de factores físicos": ["factores fisicos", "physical factors"],
        "Evaluación ergonómica": ["ergonomia", "ergonomic", "ergonomics"],
        "Modelamiento matemático a la mejora de procesos como soporte a la toma de decisiones": [
            "modelamiento matematico", "mathematical modeling", "decision making"
        ],
        "Simulación para la mejora del diseño de procesos": ["simulacion", "simulation", "process simulation"],
        "Diseño y desarrollo de modelos para el análisis y predicción de las variables de un proceso": [
            "prediccion", "prediction", "forecasting", "analisis de variables"
        ],
        "Planeamiento y Gestión de Operaciones": ["operations management", "gestion de operaciones", "planeamiento de operaciones"],
        "Planeamiento, programación y control de proyectos": ["project scheduling", "project control", "programacion y control de proyectos"],
        "Gestión de mantenimiento": ["mantenimiento", "maintenance management"],
        "Gestión de la cadena de suministro": ["supply chain", "cadena de suministro"],
        "Gestión de Logística Inversa": ["logistica inversa", "reverse logistics"],
        "Gestión de Inventarios, Almacenes y Transportes": ["inventarios", "almacenes", "transportes", "inventory", "warehouse", "transportation"],
        "Gestión de compras y proveedores, Nivel de servicio y Satisfacción al cliente": [
            "compras", "proveedores", "nivel de servicio", "satisfaccion al cliente", "supplier", "service level"
        ],
        "Gestión de riesgos ocupacionales": ["riesgos ocupacionales", "occupational risks"],
        "Identificación, análisis, evaluación y control de riesgos en seguridad y salud ocupacional": [
            "seguridad y salud ocupacional", "occupational health", "occupational safety"
        ],
        "Diseño de producto": ["diseno de producto", "product design"],
        "Desarrollo de producto": ["desarrollo de producto", "product development"],
    },
    "Ingeniería Civil": {
        "Ecuaciones diferenciales aplicadas al análisis estructural": ["analisis estructural", "structural analysis", "ecuaciones diferenciales"],
        "Física de materiales": ["fisica de materiales", "materials physics"],
        "Metodología BIM": ["bim", "building information modeling"],
        "Normativa BIM": ["normativa bim", "bim standard", "bim standards"],
        "Materiales de Construcción": ["materiales de construccion", "construction materials"],
        "Sostenibilidad": ["sostenibilidad", "sustainability"],
        "Innovación en Proceso constructivos": ["proceso constructivo", "construction process innovation"],
        "Tecnología": ["tecnologia en construccion", "construction technology"],
        "Técnicas de experimentación en estructuras": ["experimental structures", "experimentacion en estructuras"],
        "Vulnerabilidad sísmica de estructuras": ["vulnerabilidad sismica", "seismic vulnerability"],
        "Sistemas de protección sísmica de estructuras": ["proteccion sismica", "seismic protection", "base isolation"],
        "Obras de Construcción y su relación con el medio ambiente": ["medio ambiente", "environmental impact", "construction and environment"],
        "Hidrología e Hidráulica": ["hidrologia", "hidraulica", "hydrology", "hydraulics"],
        "Riego y Drenaje": ["riego", "drenaje", "irrigation", "drainage"],
        "Calidad del Agua": ["calidad del agua", "water quality"],
        "Cambio Climático en Recursos Hídricos": ["cambio climatico", "climate change", "recursos hidricos", "water resources"],
        "Transporte de Sedimentos": ["sedimentos", "sediment transport"],
        "Hidrogeología": ["hidrogeologia", "hydrogeology"],
        "Geotecnia computacional": ["geotecnia computacional", "computational geotechnics"],
        "Geotecnia ambiental e hidrogeología": ["geotecnia ambiental", "environmental geotechnics"],
        "Geotecnia experimental": ["geotecnia experimental", "experimental geotechnics"],
        "Geotecnia minera": ["geotecnia minera", "mining geotechnics"],
        "Mecánica de rocas e ingeniería geológica": ["mecanica de rocas", "rock mechanics", "ingenieria geologica"],
        "Tecnologia de mezclas asfalticas": ["mezclas asfalticas", "asphalt mixtures"],
        "Diseño estructural de Pavimentos": ["pavimentos", "pavement design"],
        "Mejora y estabilización del suelo": ["estabilizacion del suelo", "soil stabilization", "mejora del suelo"],
        "Cambio climatico en la infraestructura vial": ["infraestructura vial", "road infrastructure", "climate change in transport infrastructure"],
        "Gestion de riesgos geológicos": ["riesgos geologicos", "geological risks"],
        "Gestión de riesgos": ["gestion de riesgos", "risk management"],
        "Gestión estratégica de contratos": ["contratos", "contract management"],
        "Gestión de las comunicaciones": ["communications management", "gestion de las comunicaciones"],
        "Gestión de recursos": ["resource management", "gestion de recursos"],
    },
    "Ingeniería de Sistemas": {
        "Visión computacional": ["vision computacional", "computer vision"],
        "NLP": ["nlp", "natural language processing", "procesamiento de lenguaje natural"],
        "Aprendizaje automático": ["aprendizaje automatico", "machine learning"],
        "Minería de datos": ["mineria de datos", "data mining"],
        "Seguridad de sistemas y aplicaciones": ["seguridad de sistemas", "application security", "systems security"],
        "IoT": ["iot", "internet of things"],
        "Computación de alto rendimiento": ["high performance computing", "computacion de alto rendimiento", "hpc"],
        "Redes y ciberseguridad": ["ciberseguridad", "cybersecurity", "networks", "redes"],
        "Sostenibilidad en TI": ["sostenibilidad en ti", "green it"],
        "HCI": ["hci", "human computer interaction", "interaccion humano computadora"],
        "Realidad virtual y aumentada": ["realidad virtual", "realidad aumentada", "virtual reality", "augmented reality"],
        "Construcción de juegos y gamificación": ["gamification", "gamificacion", "game development", "juegos"],
        "Agentes virtuales": ["virtual agents", "agentes virtuales", "chatbot", "chatbots"],
        "Optimización computacional": ["optimizacion computacional", "computational optimization"],
        "Ingeniería de software": ["ingenieria de software", "software engineering"],
        "Diseño de algoritmos": ["diseno de algoritmos", "algorithm design"],
        "Gestión de procesos tecnológicos": ["procesos tecnologicos", "technology process management"],
        "Liderazgo, género y tecnología": ["genero y tecnologia", "gender and technology", "liderazgo y tecnologia"],
        "Sistemas de gestión del conocimiento": ["gestion del conocimiento", "knowledge management systems"],
        "Minería de procesos": ["mineria de procesos", "process mining"],
        "Computación aplicada": ["computacion aplicada", "applied computing"],
        "Simulación de procesos": ["simulacion de procesos", "process simulation"],
    },
}

IDIC_LINE_HINTS = {
    "Machine learning y deep learning": ["machine learning", "deep learning", "aprendizaje automatico"],
    "Procesamiento de lenguaje natural": ["natural language processing", "procesamiento de lenguaje natural", "nlp"],
    "Visión computacional": ["computer vision", "vision computacional"],
    "Sistemas autónomos y robótica": ["robotica", "robotics", "autonomous systems"],
    "Tecnologías emergentes": ["tecnologias emergentes", "emerging technologies"],
    "Ciberseguridad y privacidad": ["ciberseguridad", "cybersecurity", "privacy", "privacidad"],
    "Internet de las cosas (IoT)": ["internet of things", "iot"],
    "Computación cuántica": ["computacion cuantica", "quantum computing"],
    "Diseño y construcción virtual": ["construccion virtual", "virtual construction", "bim"],
    "Interacción humano-computadora": ["human computer interaction", "interaccion humano computadora", "hci"],
    "Realidad virtual y aumentada": ["virtual reality", "augmented reality", "realidad virtual", "realidad aumentada"],
    "Diseño de interfaces adaptativas": ["adaptive interfaces", "interfaces adaptativas"],
    "Energías renovables": ["energias renovables", "renewable energy"],
    "Economía circular": ["economia circular", "circular economy"],
    "Gestión sostenible de recursos": ["gestion sostenible de recursos", "sustainable resource management"],
    "Adaptación al cambio climático": ["adaptacion al cambio climatico", "climate adaptation"],
    "Urbanismo sostenible": ["urbanismo sostenible", "sustainable urbanism"],
    "Movilidad urbana": ["movilidad urbana", "urban mobility"],
    "Infraestructura sostenible": ["infraestructura sostenible", "sustainable infrastructure"],
    "Gestión inteligente de recursos": ["smart resource management", "gestion inteligente de recursos"],
    "Tecnologías limpias": ["tecnologias limpias", "clean technologies"],
    "Biodiversidad y conservación": ["biodiversidad", "conservacion", "biodiversity", "conservation"],
    "Gestión de residuos": ["gestion de residuos", "waste management"],
    "Materiales avanzados": ["materiales avanzados", "advanced materials"],
    "Salud mental y bienestar": ["salud mental", "mental health", "wellbeing", "bienestar"],
    "Educación, desarrollo cognitivo y socioafectivo": ["educacion", "education", "desarrollo cognitivo", "socioafectivo"],
    "Comportamiento social": ["comportamiento social", "social behavior"],
    "Mujer, cultura y sociedad": ["mujer", "woman", "women", "culture and society", "cultura y sociedad"],
    "Pobreza e informalidad": ["pobreza", "poverty", "informalidad", "informality"],
    "Medios digitales y sociedad": ["medios digitales", "digital media", "society"],
    "Comunicación intercultural": ["comunicacion intercultural", "intercultural communication"],
    "Narrativas transmedia": ["narrativas transmedia", "transmedia narratives"],
    "Comportamiento digital": ["comportamiento digital", "digital behavior"],
    "Ética y gobernanza": ["etica", "ethics", "governance", "gobernanza"],
    "Responsabilidad social": ["responsabilidad social", "social responsibility"],
    "Derechos humanos y tecnología": ["derechos humanos", "human rights", "technology"],
    "Modelos de negocio digitales": ["modelos de negocio digitales", "digital business models"],
    "Emprendimiento tecnológico": ["emprendimiento tecnologico", "technology entrepreneurship"],
    "Gestión de la innovación": ["gestion de la innovacion", "innovation management"],
    "Transformación organizacional": ["transformacion organizacional", "organizational transformation"],
    "Fintech y servicios financieros": ["fintech", "financial services", "servicios financieros"],
    "Mercados globales": ["mercados globales", "global markets"],
    "Análisis de datos económicos": ["analisis de datos economicos", "economic data analysis"],
    "Economía de plataformas": ["economia de plataformas", "platform economy"],
    "Gestión del capital intelectual": ["capital intelectual", "intellectual capital"],
    "Aprendizaje organizacional": ["aprendizaje organizacional", "organizational learning"],
    "Transferencia de conocimiento": ["transferencia de conocimiento", "knowledge transfer"],
    "Inteligencia de negocios": ["inteligencia de negocios", "business intelligence"],
}


def build_classification_score(corpus: str, aliases: list[str]) -> int:
    score = 0
    for alias in aliases:
        if phrase_in_text(corpus, alias):
            score += 1
    return score


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

    corpus = build_text_corpus(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )
    if not corpus:
        return {"area_carrera_raw": None, "linea_carrera_raw": None}

    line_scores: dict[str, int] = {}
    for area_name, lineas in CAREER_AREA_LINE_CATALOG[carrera].items():
        for linea in lineas:
            aliases = [linea, area_name]
            aliases.extend(CAREER_LINE_HINTS.get(carrera, {}).get(linea, []))
            line_scores[linea] = build_classification_score(corpus, aliases)

    best_linea = choose_best_scored_candidate(line_scores, min_score=1)
    best_area = coerce_area_carrera_from_linea(carrera, best_linea) if best_linea else None

    return {"area_carrera_raw": best_area, "linea_carrera_raw": best_linea}


def classify_idic_dimensions_by_hints(
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    corpus = build_text_corpus(
        title_value,
        abstract_value,
        author_keywords_value,
        index_keywords_value,
        source_title_value,
    )
    if not corpus:
        return {
            "category_tematica_raw": None,
            "area_idic_raw": None,
            "linea_idic_raw": None,
        }

    line_scores: dict[str, int] = {}
    for category_name, areas in IDIC_CATEGORY_AREA_LINE_CATALOG.items():
        for area_name, lineas in areas.items():
            for linea in lineas:
                aliases = [linea, area_name, category_name]
                aliases.extend(IDIC_LINE_HINTS.get(linea, []))
                line_scores[linea] = build_classification_score(corpus, aliases)

    best_linea = choose_best_scored_candidate(line_scores, min_score=1)
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


def get_thematic_llm_min_confidence() -> float:
    raw = os.environ.get("THEMATIC_LLM_MIN_CONFIDENCE", "0.80")
    try:
        value = float(raw)
    except Exception:
        value = 0.80

    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def build_thematic_classification_prompt(
    carrera: str,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> str:
    career_catalog = CAREER_AREA_LINE_CATALOG.get(carrera, {})

    payload = {
        "carrera": carrera,
        "career_catalog": career_catalog,
        "idic_catalog": IDIC_CATEGORY_AREA_LINE_CATALOG,
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
            "category_tematica_raw": "string|null",
            "area_idic_raw": "string|null",
            "linea_idic_raw": "string|null",
            "confidence": "number 0..1",
            "justification": "short string",
        },
        "rules": [
            "Usa principalmente title y abstract_scopus. Usa keywords y source_title solo como apoyo.",
            "Debes elegir solo valores existentes en los catálogos proporcionados.",
            "No inventes áreas, líneas o categorías.",
            "Si no hay evidencia suficiente, devuelve null en los campos inciertos.",
            "Si eliges linea_carrera_raw, debe pertenecer a la carrera dada y ser consistente con area_carrera_raw.",
            "Si eliges linea_idic_raw, debe ser consistente con area_idic_raw y category_tematica_raw.",
            "Devuelve únicamente JSON válido, sin markdown ni comentarios.",
        ],
    }

    return (
        "Eres un clasificador temático estricto para publicaciones académicas de ingeniería. "
        "Analiza el artículo y clasifica usando únicamente el catálogo cerrado proporcionado.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def validate_llm_thematic_output(carrera: str | None, raw_output: dict | None) -> dict:
    result = {
        "area_carrera_raw": None,
        "linea_carrera_raw": None,
        "category_tematica_raw": None,
        "area_idic_raw": None,
        "linea_idic_raw": None,
        "confidence": None,
        "justification": None,
    }

    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG or not raw_output:
        return result

    confidence = parse_float_or_none(raw_output.get("confidence"))
    result["confidence"] = confidence
    result["justification"] = raw_output.get("justification")

    area_carrera = coerce_choice(raw_output.get("area_carrera_raw"), get_allowed_career_areas(carrera))
    linea_carrera = coerce_choice(raw_output.get("linea_carrera_raw"), get_allowed_career_lines(carrera))

    if linea_carrera and not area_carrera:
        area_carrera = coerce_area_carrera_from_linea(carrera, linea_carrera)

    if not is_valid_career_area_line(carrera, area_carrera, linea_carrera):
        area_carrera = None
        linea_carrera = None

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

    min_confidence = get_thematic_llm_min_confidence()
    if confidence is not None and confidence < min_confidence:
        return result

    result["area_carrera_raw"] = area_carrera
    result["linea_carrera_raw"] = linea_carrera
    result["category_tematica_raw"] = category_tematica
    result["area_idic_raw"] = area_idic
    result["linea_idic_raw"] = linea_idic
    return result


def classify_thematic_fields_with_llm(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    empty_result = {
        "area_carrera_raw": None,
        "linea_carrera_raw": None,
        "category_tematica_raw": None,
        "area_idic_raw": None,
        "linea_idic_raw": None,
        "confidence": None,
        "justification": None,
    }

    if not carrera or carrera not in CAREER_AREA_LINE_CATALOG:
        return empty_result

    if not is_thematic_llm_configured():
        return empty_result

    prompt = build_thematic_classification_prompt(
        carrera=carrera,
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
    )

    try:
        client = get_azure_openai_client()
        response = client.responses.create(
            model=get_env("AZURE_OPENAI_RESPONSES_MODEL"),
            input=prompt,
        )

        raw_output = extract_json_object(getattr(response, "output_text", None))
        if not raw_output:
            logging.warning("Thematic LLM returned non-JSON output.")
            return empty_result

        return validate_llm_thematic_output(carrera, raw_output)

    except Exception as exc:
        logging.warning("Thematic LLM classification failed: %s", str(exc))
        return empty_result


def classify_thematic_fields(
    carrera: str | None,
    title_value: str | None,
    abstract_value: str | None,
    author_keywords_value: str | None,
    index_keywords_value: str | None,
    source_title_value: str | None,
) -> dict:
    llm_result = classify_thematic_fields_with_llm(
        carrera=carrera,
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
    )

    has_any_llm_value = any(
        [
            llm_result.get("area_carrera_raw"),
            llm_result.get("linea_carrera_raw"),
            llm_result.get("category_tematica_raw"),
            llm_result.get("area_idic_raw"),
            llm_result.get("linea_idic_raw"),
        ]
    )

    if has_any_llm_value:
        return llm_result

    career_hint_result = classify_career_dimensions_by_hints(
        carrera=carrera,
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
    )
    idic_hint_result = classify_idic_dimensions_by_hints(
        title_value=title_value,
        abstract_value=abstract_value,
        author_keywords_value=author_keywords_value,
        index_keywords_value=index_keywords_value,
        source_title_value=source_title_value,
    )

    return {
        "area_carrera_raw": career_hint_result.get("area_carrera_raw"),
        "linea_carrera_raw": career_hint_result.get("linea_carrera_raw"),
        "category_tematica_raw": idic_hint_result.get("category_tematica_raw"),
        "area_idic_raw": idic_hint_result.get("area_idic_raw"),
        "linea_idic_raw": idic_hint_result.get("linea_idic_raw"),
        "confidence": None,
        "justification": None,
    }


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
    if not is_ulima_text(affiliations_value):
        return []
    return unique_keep_order(infer_careers_from_text(affiliations_value))


def get_block_engineering_careers(
    block_value: str | None,
    publication_level_careers: list[str],
) -> list[str]:
    if not is_ulima_text(block_value):
        return []

    block_careers = unique_keep_order(infer_careers_from_text(block_value))
    if block_careers:
        return block_careers

    if len(publication_level_careers) == 1:
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

    publication_level_careers = get_publication_level_engineering_careers(affiliations_value)

    ulima_authors_detected: list[str] = []
    publication_careers: list[str] = []
    first_author_ulima = False
    matched_any = False
    affiliation_ulima_detected = False

    for idx, block in enumerate(author_blocks):
        if not is_ulima_text(block):
            continue

        affiliation_ulima_detected = True

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

            if not block_careers and docente.get("carrera"):
                block_careers = [docente.get("carrera")]

        publication_careers.extend(block_careers)

    if not publication_careers and publication_level_careers:
        publication_careers.extend(publication_level_careers)

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
