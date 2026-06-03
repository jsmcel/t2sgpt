#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"
CODEX = shutil.which("codex.cmd") or shutil.which("codex") or "codex"
MODEL_PRESETS = {
    "codex_high": {"label": "Codex High", "reasoning": "high"},
    "codex_fast": {"label": "Codex Fast", "reasoning": "low"},
    "local_rag": {"label": "Solo RAG", "reasoning": "none"},
}
GENERATION_CONTEXT_HITS = 24
GENERATION_RETRIEVAL_HITS = 40
MAX_CONTEXT_HITS = 64
DOMAIN_EVIDENCE_SCORE_THRESHOLD = 0.18
RELEASE_RE = re.compile(r"R(20\d{2})[._ -]?(NOV|OCT|JUN|MAR)", re.I)
MESSAGE_RE = re.compile(r"\b(acmt|admi|camt|pacs|reda|semt|sese|seev)\.(\d{3})(?:\.(\d{3}))?\b", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,12}\b")
EASTER_EGG_TRIGGER = "galleta vietnamita"
EASTER_EGG_RESPONSES = (
    "hay mucho rubio en suecia",
    "hace frio en suecia",
    "César Conesa",
    "Didieur te llevara al eur",
)

SPANISH_LANGUAGE_HINTS = {
    "como",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "dame",
    "de",
    "del",
    "el",
    "en",
    "es",
    "explica",
    "hay",
    "la",
    "las",
    "los",
    "que",
    "qué",
    "son",
    "una",
}
ENGLISH_LANGUAGE_HINTS = {"are", "does", "explain", "how", "in", "is", "of", "the", "what", "which", "who"}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "como",
    "con",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "de",
    "del",
    "dame",
    "el",
    "en",
    "es",
    "explica",
    "for",
    "is",
    "la",
    "las",
    "los",
    "para",
    "que",
    "qué",
    "sobre",
    "the",
    "what",
    "which",
    "y",
}
DOMAIN_LEXICON_STOPWORDS = QUESTION_STOPWORDS | {
    "all",
    "also",
    "annex",
    "chapter",
    "clean",
    "document",
    "documents",
    "english",
    "from",
    "general",
    "html",
    "index",
    "page",
    "pages",
    "part",
    "pdf",
    "release",
    "shared",
    "table",
    "target",
    "target2",
    "version",
    "with",
}
FAMILY_HINTS = {
    "t2s_sdd": ["sdd", "scope defining", "scope", "legal basis", "service description"],
    "t2s_udfs": ["udfs", "functional specification", "message", "schema", "xsd", "xml"],
    "t2s_uhb": ["uhb", "user handbook", "gui", "screen", "u2a", "user interface"],
    "change_requests": ["change request", "cr", "crs", "release", "impact", "status"],
    "connectivity": ["connectivity", "esmig", "nsp", "swift", "message exchange", "technical"],
    "business_processes": ["business process", "bpd", "settlement", "matching", "lifecycle"],
    "migration_and_testing": ["migration", "testing", "readiness", "test", "certification"],
    "pricing": ["pricing", "fee", "tariff", "billing", "cost"],
    "legal": ["legal", "framework agreement", "guideline", "terms"],
    "participation": ["csd", "dcp", "participant", "onboarding", "account", "party"],
    "shared_features": ["shared", "common", "crdm", "billing", "reference data"],
    "messages_and_schemas": ["iso 20022", "message", "schema", "xsd", "semt", "sese", "camt", "reda"],
}

MESSAGE_DEFINITIONS = {
    "sese.023": {
        "name": "Securities Settlement Transaction Instruction",
        "short_es": (
            "la instruccion ISO 20022 que un CSD o un DCP envia a T2S para crear o mantener "
            "una instruccion de liquidacion de valores"
        ),
        "short_en": (
            "the ISO 20022 instruction that a CSD or DCP sends to T2S to create or maintain "
            "a securities settlement instruction"
        ),
        "sender_es": "Lo envia normalmente un CSD o un DCP a T2S.",
        "sender_en": "It is normally sent by a CSD or DCP to T2S.",
        "purpose_es": (
            "Sirve para decirle a T2S que debe casar y liquidar una operacion de valores: "
            "quien entrega, quien recibe, que instrumento se liquida, que cantidad, en que fecha prevista "
            "y bajo que referencias e indicadores operativos."
        ),
        "purpose_en": (
            "It tells T2S to match and settle a securities transaction: delivering party, receiving party, "
            "instrument, quantity, intended settlement date, references and operational indicators."
        ),
        "flow_es": (
            "Despues de recibir un sese.023, T2S contesta con mensajes de estado, sobre todo sese.024: "
            "validacion, matching y estado de liquidacion. Si la operacion liquida, el flujo se completa "
            "con confirmaciones como sese.025; si falta la contrapartida, aparecen allegements como sese.028 "
            "y su retirada mediante sese.029."
        ),
        "flow_en": (
            "After receiving a sese.023, T2S replies with status messages, mainly sese.024: validation, "
            "matching and settlement status. If settlement takes place the flow is completed by confirmations "
            "such as sese.025; if the counterparty leg is missing, allegement messages such as sese.028 and "
            "their removal via sese.029 appear."
        ),
        "not_es": (
            "No es una confirmacion ni un informe de estado: es la instruccion de entrada que arranca "
            "el ciclo operativo de matching y liquidacion."
        ),
        "not_en": (
            "It is not a confirmation or a status report: it is the inbound instruction that starts "
            "the matching and settlement lifecycle."
        ),
        "related": [
            ("sese.024", "Securities Settlement Transaction Status Advice"),
            ("sese.025", "Securities Settlement Transaction Confirmation"),
            ("sese.028", "Securities Settlement Transaction Allegement Notification"),
            ("sese.029", "Securities Settlement Allegement Removal"),
            ("semt.026", "Securities Settlement Transaction Query"),
            ("semt.027", "Securities Settlement Transaction Query Response"),
        ],
    },
    "sese.024": {
        "name": "Securities Settlement Transaction Status Advice",
        "short_es": "el mensaje de estado que T2S usa para informar sobre validacion, matching o liquidacion de una instruccion",
        "short_en": "the status message used by T2S to report validation, matching or settlement status of an instruction",
        "sender_es": "Lo envia T2S hacia el CSD o DCP afectado.",
        "sender_en": "It is sent by T2S to the relevant CSD or DCP.",
        "purpose_es": "Sirve para comunicar como va una instruccion: aceptada, rechazada, casada, pendiente o con incidencias.",
        "purpose_en": "It reports how an instruction is progressing: accepted, rejected, matched, pending or affected by issues.",
        "flow_es": "Suele aparecer como respuesta o actualizacion posterior a una instruccion sese.023.",
        "flow_en": "It commonly appears as a response or later update after a sese.023 instruction.",
        "not_es": "No crea la instruccion: informa de su estado.",
        "not_en": "It does not create the instruction: it reports its status.",
        "related": [("sese.023", "Securities Settlement Transaction Instruction"), ("sese.025", "Securities Settlement Transaction Confirmation")],
    },
    "sese.025": {
        "name": "Securities Settlement Transaction Confirmation",
        "short_es": "la confirmacion de liquidacion de una transaccion de valores",
        "short_en": "the confirmation of a securities settlement transaction",
        "sender_es": "Lo envia T2S cuando debe confirmar el resultado de liquidacion.",
        "sender_en": "It is sent by T2S when it must confirm the settlement result.",
        "purpose_es": "Sirve para confirmar que la instruccion ha liquidado, total o parcialmente segun el caso operativo.",
        "purpose_en": "It confirms that the instruction has settled, fully or partially depending on the operational case.",
        "flow_es": "Aparece despues del matching y del proceso de settlement iniciado por mensajes como sese.023.",
        "flow_en": "It appears after matching and settlement processing initiated by messages such as sese.023.",
        "not_es": "No es la orden de liquidar: es la confirmacion del resultado.",
        "not_en": "It is not the order to settle: it confirms the result.",
        "related": [("sese.023", "Securities Settlement Transaction Instruction"), ("sese.024", "Securities Settlement Transaction Status Advice")],
    },
    "sese.028": {
        "name": "Securities Settlement Transaction Allegement Notification",
        "short_es": "la notificacion de allegement cuando T2S ve una instruccion de una parte y falta o no casa la contrapartida",
        "short_en": "the allegement notification used when T2S sees one party's instruction and the counterparty leg is missing or not matched",
        "sender_es": "Lo envia T2S a la parte que debe ser informada del allegement.",
        "sender_en": "It is sent by T2S to the party that must be informed of the allegement.",
        "purpose_es": "Sirve para alertar a la posible contrapartida de que existe una instruccion que requiere actuacion o matching.",
        "purpose_en": "It alerts the possible counterparty that an instruction exists and requires action or matching.",
        "flow_es": "En el flujo de matching aparece despues del sese.023 y antes de su retirada mediante sese.029 si procede.",
        "flow_en": "In the matching flow it appears after the sese.023 and before removal via sese.029 where applicable.",
        "not_es": "No liquida la operacion: avisa de una situacion de matching pendiente.",
        "not_en": "It does not settle the transaction: it warns about a pending matching situation.",
        "related": [("sese.023", "Securities Settlement Transaction Instruction"), ("sese.029", "Securities Settlement Allegement Removal")],
    },
    "sese.029": {
        "name": "Securities Settlement Allegement Removal",
        "short_es": "el mensaje que retira un allegement previamente notificado",
        "short_en": "the message that removes a previously notified allegement",
        "sender_es": "Lo envia T2S cuando el allegement deja de aplicar.",
        "sender_en": "It is sent by T2S when the allegement no longer applies.",
        "purpose_es": "Sirve para cerrar la alerta de allegement, por ejemplo porque la situacion de matching ha cambiado.",
        "purpose_en": "It closes the allegement alert, for example because the matching situation has changed.",
        "flow_es": "Esta relacionado con sese.028 y con el ciclo iniciado por sese.023.",
        "flow_en": "It is related to sese.028 and to the lifecycle initiated by sese.023.",
        "not_es": "No es una nueva instruccion: limpia una notificacion anterior.",
        "not_en": "It is not a new instruction: it clears a previous notification.",
        "related": [("sese.028", "Securities Settlement Transaction Allegement Notification"), ("sese.023", "Securities Settlement Transaction Instruction")],
    },
}

TERM_DEFINITIONS = {
    "esmig": {
        "triggers": ["esmig", "connectivity", "conectividad"],
        "name": "ESMIG",
        "short_es": "la capa de conectividad de Eurosystem Market Infrastructure Gateway usada para intercambiar mensajes con T2S y otros servicios TARGET",
        "short_en": "the Eurosystem Market Infrastructure Gateway connectivity layer used to exchange messages with T2S and other TARGET Services",
        "role_es": "Concentra el acceso tecnico: los participantes no hablan con T2S de forma aislada, sino a traves del canal y reglas de conectividad definidos para ESMIG.",
        "role_en": "It concentrates technical access: participants do not connect to T2S in isolation, but through the ESMIG channel and connectivity rules.",
        "flow_es": "En la practica, ESMIG transporta la mensajeria A2A/U2A entre CSDs, DCPs y T2S; los detalles operativos dependen del actor, del NSP y del tipo de servicio.",
        "flow_en": "In practice, ESMIG transports A2A/U2A messaging between CSDs, DCPs and T2S; operational details depend on the actor, NSP and service type.",
    },
    "dvp": {
        "triggers": ["dvp", "delivery versus payment", "liquidacion", "liquidación"],
        "name": "DvP",
        "short_es": "el mecanismo de liquidacion entrega contra pago: los valores y el efectivo se liquidan de forma vinculada",
        "short_en": "the delivery-versus-payment settlement mechanism: securities and cash settle in a linked way",
        "role_es": "Reduce riesgo principal porque la entrega de valores y el pago se condicionan mutuamente dentro del proceso de liquidacion.",
        "role_en": "It reduces principal risk because securities delivery and cash payment are mutually conditioned in settlement processing.",
        "flow_es": "En T2S, la instruccion entra, se valida, se casa, se comprueba la disponibilidad de valores y efectivo, y despues se liquida o queda pendiente.",
        "flow_en": "In T2S, the instruction is received, validated, matched, checked for securities and cash availability, then settled or left pending.",
    },
    "sdd": {
        "triggers": ["sdd", "scope defining", "scope defining document"],
        "name": "Scope Defining Documents",
        "short_es": "los documentos que delimitan el alcance funcional de T2S antes de bajar al detalle tecnico",
        "short_en": "the documents that define the functional scope of T2S before detailed technical specification",
        "role_es": "Sirven para fijar que entra y que queda fuera del servicio: procesos, actores, responsabilidades y limites funcionales.",
        "role_en": "They set what is inside and outside the service: processes, actors, responsibilities and functional boundaries.",
        "flow_es": "Son una referencia de alcance; para implementacion detallada se baja a UDFS, UHB, esquemas ISO 20022 y documentacion de conectividad.",
        "flow_en": "They are scope references; implementation detail is found in UDFS, UHB, ISO 20022 schemas and connectivity documentation.",
    },
    "uhb": {
        "triggers": ["uhb", "user handbook", "pantalla", "pantallas", "gui", "u2a"],
        "name": "UHB",
        "short_es": "el User Handbook de T2S: la guia de uso de pantallas, funciones U2A y operativa de usuario",
        "short_en": "the T2S User Handbook: the guide for screens, U2A functions and user operations",
        "role_es": "Sirve para saber como opera un usuario en la interfaz: consultas, pantallas, acciones y datos visibles.",
        "role_en": "It explains how a user operates through the interface: queries, screens, actions and visible data.",
        "flow_es": "Complementa a la UDFS: la UDFS describe funcionalidad y mensajes; la UHB aterriza la experiencia de usuario.",
        "flow_en": "It complements the UDFS: UDFS describes functionality and messages; UHB describes the user experience.",
    },
    "udfs": {
        "triggers": ["udfs", "user detailed functional specification", "functional specification"],
        "name": "UDFS",
        "short_es": "la User Detailed Functional Specification: la especificacion funcional detallada de T2S",
        "short_en": "the User Detailed Functional Specification: the detailed functional specification of T2S",
        "role_es": "Es la referencia para procesos, reglas funcionales, mensajeria A2A, campos, validaciones e interacciones principales.",
        "role_en": "It is the reference for processes, functional rules, A2A messaging, fields, validations and core interactions.",
        "flow_es": "Cuando se implementa o se analiza un flujo T2S, la UDFS suele ser la fuente base junto con esquemas ISO 20022 y change requests.",
        "flow_en": "When implementing or analysing a T2S flow, UDFS is usually the baseline source together with ISO 20022 schemas and change requests.",
    },
    "dcp": {
        "triggers": ["dcp", "directly connected party"],
        "name": "DCP",
        "short_es": "una Directly Connected Party: entidad autorizada a conectarse directamente con T2S para intercambiar mensajes",
        "short_en": "a Directly Connected Party: an entity authorised to connect directly to T2S to exchange messages",
        "role_es": "Permite operar por canal directo, normalmente para mensajeria y seguimiento de instrucciones bajo el marco del CSD correspondiente.",
        "role_en": "It enables direct-channel operation, normally for messaging and monitoring instructions under the relevant CSD framework.",
        "flow_es": "Un DCP puede enviar o recibir mensajes T2S como instrucciones, estados, consultas y notificaciones segun su perfil y autorizaciones.",
        "flow_en": "A DCP may send or receive T2S messages such as instructions, status messages, queries and notifications according to its profile and permissions.",
    },
    "csd": {
        "triggers": ["csd", "central securities depository"],
        "name": "CSD",
        "short_es": "un Central Securities Depository: el depositario central de valores que participa en T2S",
        "short_en": "a Central Securities Depository participating in T2S",
        "role_es": "Es el actor de infraestructura que mantiene cuentas de valores y canaliza la liquidacion de sus participantes en T2S.",
        "role_en": "It is the infrastructure actor that maintains securities accounts and channels its participants' settlement in T2S.",
        "flow_es": "En los flujos T2S, el CSD aparece como parte clave para instrucciones, cuentas, participantes, elegibilidad y reporting.",
        "flow_en": "In T2S flows, the CSD is central for instructions, accounts, participants, eligibility and reporting.",
    },
    "dca": {
        "triggers": ["dca", "dedicated cash account"],
        "name": "DCA",
        "short_es": "una Dedicated Cash Account: cuenta de efectivo dedicada usada para la pata de efectivo de la liquidacion",
        "short_en": "a Dedicated Cash Account used for the cash leg of settlement",
        "role_es": "Sirve para liquidar el efectivo vinculado a operaciones de valores, especialmente en flujos DvP.",
        "role_en": "It settles the cash linked to securities transactions, especially in DvP flows.",
        "flow_es": "La liquidacion comprueba disponibilidad de valores y de efectivo; la DCA participa en esa comprobacion y movimiento de efectivo.",
        "flow_en": "Settlement checks securities and cash availability; the DCA is involved in that cash check and movement.",
    },
    "allegement": {
        "triggers": ["allegement", "allegements", "allgmt", "sctiessttlmtxallgmt", "notificacion de allegement", "notificación de allegement"],
        "name": "Allegement",
        "short_es": "una alerta de matching: T2S informa de que existe una instruccion de una parte y falta, no llega o no casa la instruccion de la contrapartida",
        "short_en": "a matching alert: T2S informs that one party's instruction exists and the counterparty instruction is missing, not received or not matched",
        "role_es": "Sirve para avisar a la posible contrapartida de que debe revisar o introducir su instruccion para que la operacion pueda casar.",
        "role_en": "It alerts the possible counterparty that it should review or enter its instruction so the transaction can match.",
        "flow_es": "En el flujo T2S suele vincularse a `sese.028` como notificacion de allegement; si deja de aplicar, T2S lo retira con mensajes como `sese.029`.",
        "flow_en": "In the T2S flow it is typically linked to `sese.028` as allegement notification; when it no longer applies, T2S removes it with messages such as `sese.029`.",
    },
    "change_request": {
        "triggers": ["change request", "change requests", "cr ", "crs", "release"],
        "name": "Change Request",
        "short_es": "una peticion formal de cambio sobre T2S que describe necesidad, impacto, documentacion afectada y release objetivo",
        "short_en": "a formal T2S change request describing need, impact, affected documentation and target release",
        "role_es": "Sirve para gobernar cambios funcionales o tecnicos: mensajes, pantallas, reglas, datos, reporting o conectividad.",
        "role_en": "It governs functional or technical changes: messages, screens, rules, data, reporting or connectivity.",
        "flow_es": "Normalmente se identifica con codigo T2S-xxxx, estado, justificacion, impactos y referencias a UDFS, UHB, esquemas o releases.",
        "flow_en": "It is normally identified by a T2S-xxxx code, status, rationale, impacts and references to UDFS, UHB, schemas or releases.",
    },
}


@dataclass
class Hit:
    rank: int
    score: float
    chunk: dict[str, Any]
    reason: str = ""

    @property
    def citation(self) -> str:
        return cite_label(self)


def _normalize_easter_text(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = folded.lower()
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return re.sub(r"\s+", " ", folded).strip()


def _similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def is_easter_egg_query(query: str) -> bool:
    normalized = _normalize_easter_text(query)
    target = _normalize_easter_text(EASTER_EGG_TRIGGER)
    if not normalized:
        return False
    if target in normalized:
        return True
    compact = normalized.replace(" ", "")
    target_compact = target.replace(" ", "")
    if target_compact in compact:
        return True
    if len(normalized) <= len(target) + 8 and _similarity(normalized, target) >= 0.82:
        return True
    words = normalized.split()
    for size in (2, 3):
        for start in range(0, max(0, len(words) - size + 1)):
            window = " ".join(words[start : start + size])
            window_compact = window.replace(" ", "")
            if max(_similarity(window, target), _similarity(window_compact, target_compact)) >= 0.78:
                return True
    return False


def detect_question_language(query: str) -> str:
    lower = query.lower()
    if re.search(r"[¿¡áéíóúñ]", lower):
        return "es"
    tokens = set(re.findall(r"\b[\wáéíóúñ]+\b", lower, flags=re.I))
    spanish_score = len(tokens & SPANISH_LANGUAGE_HINTS)
    english_score = len(tokens & ENGLISH_LANGUAGE_HINTS)
    return "en" if english_score > spanish_score else "es"


@lru_cache(maxsize=2)
def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run `python t2s_ingest.py` first.")
    with path.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=8)
def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query(query: str) -> str:
    query = query.strip()
    replacements = {
        "liquidacion": "settlement securities settlement delivery versus payment dvp",
        "liquidación": "settlement securities settlement delivery versus payment dvp",
        "valores": "securities settlement t2s financial instruments",
        "conectividad": "connectivity ESMIG network service provider NSP",
        "pantallas": "UHB GUI screen user interface",
        "manual": "user handbook UHB",
        "requisitos": "requirements specifications SDD UDFS",
        "mensajes": "messages schemas ISO 20022",
        "mensaje": "message schema ISO 20022",
        "esquema": "schema xsd usage guideline",
        "esquemas": "schemas xsd usage guidelines",
        "campo": "field element path minOccurs maxOccurs",
        "campos": "fields elements paths minOccurs maxOccurs",
        "cr": "change request release impact status",
        "crs": "change requests release impact status",
        "dvp": "delivery versus payment securities settlement",
        "dca": "dedicated cash account cash settlement",
        "csd": "central securities depository participant",
        "dcp": "directly connected party participant connectivity",
        "autocolateralizacion": "auto-collateralisation auto collateralisation",
        "autocolateralización": "auto-collateralisation auto collateralisation",
        "allegement": "allegement matching counterparty instruction sese.028 sese.029 semt.019",
        "allegements": "allegement matching counterparty instruction sese.028 sese.029 semt.019",
    }
    tokens = re.findall(r"[\w./-]+", query, flags=re.UNICODE)
    cleaned = " ".join(token for token in tokens if token.lower() not in QUESTION_STOPWORDS)
    expanded = [cleaned or query, "T2S TARGET2-Securities securities settlement CSD DCP UDFS UHB SDD ISO 20022"]
    low = query.lower()
    for key, value in replacements.items():
        if key in low:
            expanded.append(value)
    for code in query_message_codes(query):
        expanded.append(code)
        expanded.append(code.replace(".", "_"))
    release = query_release(query)
    if release:
        expanded.append(release)
    return " ".join(expanded)


def query_release(query: str) -> str:
    match = RELEASE_RE.search(query)
    if match:
        return f"R{match.group(1)}.{match.group(2).upper()}"
    reverse = re.search(r"\b(NOV|OCT|JUN|MAR)[._ -]?(20\d{2})\b", query, re.I)
    if reverse:
        return f"R{reverse.group(2)}.{reverse.group(1).upper()}"
    return ""


def query_message_codes(query: str) -> list[str]:
    codes: list[str] = []
    for match in MESSAGE_RE.finditer(query):
        code = f"{match.group(1).lower()}.{match.group(2)}"
        if match.group(3):
            code = f"{code}.{match.group(3)}"
        if code not in codes:
            codes.append(code)
    return codes


def query_acronyms(query: str) -> list[str]:
    return [item for item in ACRONYM_RE.findall(query.upper()) if item not in {"THE", "AND", "FOR"}]


def is_domain_query(query: str) -> bool:
    low = query.lower()
    if "t2s" in low or "target2-securities" in low or "target2 securities" in low:
        return True
    if query_release(query) or query_message_codes(query):
        return True
    domain_terms = [
        "securities settlement",
        "settlement",
        "dvp",
        "csd",
        "dcp",
        "dca",
        "uhb",
        "udfs",
        "sdd",
        "esmig",
        "auto-collateralisation",
        "autocolateral",
        "matching",
        "partial settlement",
        "corporate action",
        "allegement",
        "allegements",
        "allgmt",
        "liquidacion",
        "liquidación",
        "valores",
    ]
    return any(term in low for term in domain_terms) or any(acr in {"T2S", "CSD", "DCP", "DCA", "UDFS", "UHB", "SDD"} for acr in query_acronyms(query))


def _bm25_scores(index: dict[str, Any], query: str) -> np.ndarray:
    vectorizer = index.get("bm25_vectorizer")
    matrix = index.get("bm25_matrix")
    idf = index.get("bm25_idf")
    doc_len = index.get("bm25_doc_len")
    avgdl = float(index.get("bm25_avgdl") or 1.0)
    if vectorizer is None or matrix is None or idf is None or doc_len is None:
        return np.zeros(len(index.get("chunks", [])), dtype=float)
    q = vectorizer.transform([query])
    if q.nnz == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    k1 = 1.5
    b = 0.75
    scores = np.zeros(matrix.shape[0], dtype=float)
    for term_idx in q.indices:
        col = matrix[:, term_idx].tocoo()
        if col.nnz == 0:
            continue
        freq = col.data.astype(float)
        denom = freq + k1 * (1 - b + b * (doc_len[col.row] / avgdl))
        scores[col.row] += idf[term_idx] * (freq * (k1 + 1) / denom)
    max_score = float(scores.max() or 0.0)
    if max_score:
        scores = scores / max_score
    return scores


def _chunk_text_for_ranking(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(chunk.get(key) or "")
        for key in [
            "title",
            "family",
            "category",
            "release",
            "unit_type",
            "unit",
            "message_id",
            "usage_guideline_name",
            "collection",
            "local_path",
            "source_url",
            "text",
        ]
    )


def query_content_terms(query: str) -> list[str]:
    normalized = _normalize_easter_text(query)
    terms: list[str] = []
    for term in re.findall(r"[a-z0-9./-]{3,}", normalized):
        if term in QUESTION_STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def has_domain_evidence(query: str, hits: list[Hit]) -> bool:
    if not hits or hits[0].score < DOMAIN_EVIDENCE_SCORE_THRESHOLD:
        return False
    terms = query_content_terms(query)
    if not terms:
        return False
    top_hits = hits[: min(8, len(hits))]
    for hit in top_hits:
        if hit.score < DOMAIN_EVIDENCE_SCORE_THRESHOLD:
            continue
        hay = _normalize_easter_text(_chunk_text_for_ranking(hit.chunk))
        matched_terms = [term for term in terms if term in hay]
        if matched_terms:
            return True
    return False


def _domain_terms_from_text(text: str) -> set[str]:
    normalized = _normalize_easter_text(text)
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]{3,}", normalized)
        if token not in DOMAIN_LEXICON_STOPWORDS and not token.isdigit()
    ]
    terms: set[str] = set(tokens)
    for size in (2, 3):
        for start in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[start : start + size])
            if len(phrase) <= 80:
                terms.add(phrase)
    return terms


def _metadata_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "category", "family", "release", "unit_type", "unit", "local_path", "source_url"):
        value = item.get(key)
        if value:
            parts.append(str(value))
    context_path = item.get("context_path")
    if isinstance(context_path, list):
        parts.extend(str(part) for part in context_path if part)
    elif context_path:
        parts.append(str(context_path))
    return " ".join(parts)


@lru_cache(maxsize=2)
def load_domain_lexicon(index_path: str = str(INDEX_PATH)) -> frozenset[str]:
    index = load_index(Path(index_path))
    terms: set[str] = set()
    for doc in index.get("docs") or []:
        if isinstance(doc, dict):
            terms.update(_domain_terms_from_text(_metadata_text(doc)))
    seen_docs: set[str] = set()
    for chunk in index.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        doc_id = str(chunk.get("doc_id") or "")
        if doc_id and doc_id in seen_docs:
            continue
        if doc_id:
            seen_docs.add(doc_id)
        terms.update(_domain_terms_from_text(_metadata_text(chunk)))
    return frozenset(terms)


def is_index_domain_query(query: str, index_path: str = str(INDEX_PATH)) -> bool:
    terms = query_content_terms(query)
    if not terms:
        return False
    lexicon = load_domain_lexicon(index_path)
    if any(term in lexicon for term in terms):
        return True
    for size in (2, 3):
        for start in range(0, max(0, len(terms) - size + 1)):
            if " ".join(terms[start : start + size]) in lexicon:
                return True
    return False


def metadata_bonus(chunk: dict[str, Any], query: str) -> float:
    low = query.lower()
    hay = _chunk_text_for_ranking(chunk).lower()
    bonus = 0.0
    family = str(chunk.get("family") or "")
    for hinted_family, hints in FAMILY_HINTS.items():
        if family == hinted_family and any(hint in low for hint in hints):
            bonus += 0.18
    release = query_release(query)
    if release:
        bonus += 0.35 if chunk.get("release") == release else -0.05
    for code in query_message_codes(query):
        aliases = {code, code.replace(".", "_"), code.replace(".", " ")}
        if any(alias in hay for alias in aliases):
            bonus += 0.35
    for acronym in query_acronyms(query):
        if re.search(rf"\b{re.escape(acronym.lower())}\b", hay):
            bonus += 0.08
    if chunk.get("revision_status") == "clean":
        bonus += 0.04
    return bonus


def retrieve(index: dict[str, Any], query: str, top_k: int = 8, pool: int = 180) -> list[Hit]:
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    normalized = normalize_query(query)
    word_scores = linear_kernel(index["word_vectorizer"].transform([normalized]), index["word_matrix"]).ravel()
    char_scores = linear_kernel(index["char_vectorizer"].transform([normalized]), index["char_matrix"]).ravel()
    bm25_scores = _bm25_scores(index, normalized)
    scores = 0.48 * word_scores + 0.25 * char_scores + 0.27 * bm25_scores
    pool_size = min(max(pool, top_k * 6), len(chunks))
    candidate_idx = np.argpartition(scores, -pool_size)[-pool_size:]
    ranked = sorted(candidate_idx, key=lambda idx: scores[idx] + metadata_bonus(chunks[idx], query), reverse=True)
    hits: list[Hit] = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        score = float(scores[idx] + metadata_bonus(chunks[idx], query))
        hits.append(Hit(rank=rank, score=score, chunk=chunks[int(idx)], reason="hybrid"))
    return hits


def augment_hits(index: dict[str, Any], query: str, hits: list[Hit]) -> list[Hit]:
    return hits


def expand_neighbor_hits(index: dict[str, Any], hits: list[Hit], max_neighbors: int = 1, max_total: int = MAX_CONTEXT_HITS) -> list[Hit]:
    chunks = index.get("chunks") or []
    chunk_id_to_pos = index.get("chunk_id_to_pos") or {}
    selected: list[Hit] = []
    seen: set[str] = set()

    def add(hit: Hit) -> None:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        if chunk_id not in seen and len(selected) < max_total:
            selected.append(hit)
            seen.add(chunk_id)

    for hit in hits:
        add(hit)
        pos = chunk_id_to_pos.get(hit.chunk.get("chunk_id"))
        if pos is None:
            continue
        for offset in range(1, max_neighbors + 1):
            for neighbor_pos in (pos - offset, pos + offset):
                if 0 <= neighbor_pos < len(chunks):
                    neighbor = chunks[neighbor_pos]
                    if neighbor.get("doc_id") == hit.chunk.get("doc_id"):
                        add(Hit(rank=len(selected) + 1, score=max(hit.score - 0.08 * offset, 0.01), chunk=neighbor, reason="neighbor"))
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def context_pointer_priority(query: str, hit: Hit) -> tuple[float, float]:
    chunk = hit.chunk
    hay = _chunk_text_for_ranking(chunk).lower()
    priority = metadata_bonus(chunk, query)
    if hit.reason == "neighbor":
        priority -= 0.08
    query_terms = [term for term in re.findall(r"[\w./-]{3,}", query.lower()) if term not in QUESTION_STOPWORDS]
    priority += min(sum(0.025 for term in query_terms if term in hay), 0.5)
    return priority, hit.score


def rerank_context_hits(query: str, hits: list[Hit], max_total: int | None = None) -> list[Hit]:
    ranked = sorted(enumerate(hits), key=lambda item: (*context_pointer_priority(query, item[1]), -item[0]), reverse=True)
    selected: list[Hit] = []
    seen_chunks: set[str] = set()
    seen_texts: set[str] = set()
    for _, hit in ranked:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        text_key = re.sub(r"\W+", " ", str(hit.chunk.get("text") or "").lower())[:700]
        if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
            continue
        seen_chunks.add(chunk_id)
        if text_key:
            seen_texts.add(text_key)
        selected.append(hit)
        if max_total and len(selected) >= max_total:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def trim_excerpt(text: str, max_chars: int = 950) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    if last_stop > 350:
        return cut[: last_stop + 1].strip()
    return cut.rstrip() + "..."


def cite_label(hit: Hit) -> str:
    chunk = hit.chunk
    title = chunk.get("title") or "Untitled"
    release = chunk.get("release")
    unit_type = chunk.get("unit_type") or "unit"
    unit = chunk.get("unit")
    release_part = f", {release}" if release else ""
    where = f"{unit_type} {unit}" if unit not in (None, "") else unit_type
    return f"{title}{release_part}, {where}"


def citations_from_hits(hits: list[Hit]) -> list[dict[str, Any]]:
    citations = []
    for n, hit in enumerate(hits, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    return citations


def source_block(citations: list[dict[str, Any]], language: str) -> str:
    if not citations:
        return ""
    title = "Sources" if language == "en" else "Fuentes"
    return "\n\n" + title + ":\n" + "\n".join(f"[{c['n']}] {c['label']}" for c in citations)


def infer_question_intent(query: str, chat_history: list[dict[str, str]] | None = None) -> str:
    low = query.lower()
    if any(term in low for term in ["que es", "qué es", "what is", "define", "defin"]):
        return "definition"
    if any(term in low for term in ["diferencia", "compara", "compare", "versus", " vs "]):
        return "comparison"
    if any(term in low for term in ["flujo", "flow", "paso", "step", "secuencia"]):
        return "flow"
    if any(term in low for term in ["impacto", "impact", "cambia", "change", "cr ", "change request"]):
        return "impact"
    if any(term in low for term in ["resumen", "summary", "sintetiza"]):
        return "summary"
    if chat_history and any(term in low for term in ["y ", "entonces", "mas", "más", "tambien", "también", "eso", "este", "esta"]):
        return "follow_up"
    return "answer"


def compact_evidence_sentence(text: str, max_chars: int = 360) -> str:
    text = trim_excerpt(text, max_chars=max_chars)
    text = re.sub(r"^[•\-\d.\s]+", "", text).strip()
    return text


def build_term_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    low = query.lower()
    definition = None
    key = ""
    for candidate_key, candidate in TERM_DEFINITIONS.items():
        if any(trigger in low for trigger in candidate.get("triggers", [])):
            key = candidate_key
            definition = candidate
            break
    if not definition:
        return None
    citations = citations_from_hits(hits[: min(4, len(hits))])
    if language == "en":
        answer = (
            f"{definition['name']} is {definition['short_en']}.\n\n"
            f"- **Operational role:** {definition['role_en']}\n"
            f"- **How it fits in T2S:** {definition['flow_en']}\n"
            f"- **Practical reading:** in an answer, explain its function, the actors involved and the point of the T2S flow where it matters."
        )
    else:
        answer = (
            f"{definition['name']} es {definition['short_es']}.\n\n"
            f"- **Papel operativo:** {definition['role_es']}\n"
            f"- **Como encaja en T2S:** {definition['flow_es']}\n"
            f"- **Lectura practica:** en una respuesta hay que explicar su funcion, los actores implicados y el punto del flujo T2S donde importa."
        )
    answer += source_block(citations, language)
    return {"answer": answer, "citations": citations, "confidence": "high", "answer_type": "term_definition"}


def build_synthetic_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any]:
    top = hits[: min(5, len(hits))]
    citations = citations_from_hits(top)
    intent = infer_question_intent(query)
    excerpts = [compact_evidence_sentence(str(hit.chunk.get("text") or "")) for hit in top]
    excerpts = [item for item in excerpts if item]
    subject = query.strip().rstrip("?") or ("the requested T2S topic" if language == "en" else "el tema T2S preguntado")

    if language == "en":
        if intent == "comparison":
            lead = f"The key comparison for `{subject}` is this:"
            sections = ["- **Difference:** " + (excerpts[0] if excerpts else "The retrieved local evidence is not explicit enough to state a clean difference."),
                        "- **Operational effect:** " + (excerpts[1] if len(excerpts) > 1 else "Use context mode to inspect the detailed evidence."),
                        "- **Bottom line:** treat the result as an operational distinction and verify implementation detail in the cited sources."]
        elif intent in {"flow", "follow_up"}:
            lead = f"The operational flow for `{subject}` is:"
            sections = [f"{i + 1}. {excerpt}" for i, excerpt in enumerate(excerpts[:4])]
            sections.append("Bottom line: the relevant answer is the processing sequence; the cited sources provide conditions and implementation detail.")
        else:
            lead = f"Short answer: `{subject}` is a T2S topic that must be explained by function, actors and operational effect."
            sections = ["- **What it means:** " + (excerpts[0] if excerpts else "The local index retrieved related T2S evidence, but not a single explicit definition."),
                        "- **Why it matters:** " + (excerpts[1] if len(excerpts) > 1 else "It affects how the relevant T2S process, actor, message or release is interpreted."),
                        "- **Practical conclusion:** the answer gives the operational synthesis; the sources are there to audit the detail."]
    else:
        if intent == "comparison":
            lead = f"La comparacion clave sobre `{subject}` es esta:"
            sections = ["- **Diferencia:** " + (excerpts[0] if excerpts else "La evidencia local recuperada no formula una diferencia unica y limpia."),
                        "- **Efecto operativo:** " + (excerpts[1] if len(excerpts) > 1 else "Usa el modo Contexto para revisar el detalle documental."),
                        "- **Conclusion practica:** tratala como una distincion operativa y valida el detalle de implementacion en las fuentes."]
        elif intent in {"flow", "follow_up"}:
            lead = f"El flujo operativo sobre `{subject}` es:"
            sections = [f"{i + 1}. {excerpt}" for i, excerpt in enumerate(excerpts[:4])]
            sections.append("Conclusion: la respuesta relevante es la secuencia de procesamiento; las fuentes aportan condiciones y detalle de implementacion.")
        else:
            lead = f"Respuesta corta: `{subject}` es un tema T2S que hay que explicar por funcion, actores y efecto operativo."
            sections = ["- **Que significa:** " + (excerpts[0] if excerpts else "El indice local recupero evidencia relacionada, pero no una definicion unica literal."),
                        "- **Por que importa:** " + (excerpts[1] if len(excerpts) > 1 else "Afecta a como se interpreta el proceso, actor, mensaje o release T2S relacionado."),
                        "- **Conclusion practica:** la respuesta da la sintesis operativa; las fuentes quedan para verificar el detalle."]

    answer = lead + "\n\n" + "\n".join(sections)
    answer += source_block(citations, language)
    confidence = "high" if hits[0].score >= 0.25 else "medium" if hits[0].score >= 0.12 else "low"
    return {"answer": answer, "citations": citations, "confidence": confidence, "answer_type": f"synthetic_{intent}"}


def _message_definition_key(code: str) -> str:
    parts = code.lower().split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else code.lower()


def _message_citation_score(hit: Hit, code: str, definition: dict[str, Any]) -> float:
    hay = _chunk_text_for_ranking(hit.chunk).lower()
    aliases = {code, code.replace(".", "_"), code.replace(".", " ")}
    score = hit.score
    if any(alias in hay for alias in aliases):
        score += 0.8
    name = str(definition.get("name") or "").lower()
    if name and name in hay:
        score += 0.6
    if "payment type" in hay and "settlement quantity" in hay:
        score += 0.35
    if "used to allow the instructing party" in hay or "request a transfer of securities" in hay:
        score += 0.55
    if "validation status" in hay or "matching status" in hay:
        score += 0.25
    if "allegement" in hay:
        score += 0.15
    return score


def _select_message_citation_hits(code: str, definition: dict[str, Any], hits: list[Hit], max_items: int = 4) -> list[Hit]:
    if not hits:
        return []
    ranked = sorted(hits, key=lambda hit: _message_citation_score(hit, code, definition), reverse=True)
    selected: list[Hit] = []
    seen_docs: set[tuple[str, Any]] = set()
    for hit in ranked:
        key = (str(hit.chunk.get("doc_id") or ""), hit.chunk.get("unit"))
        if key in seen_docs:
            continue
        selected.append(hit)
        seen_docs.add(key)
        if len(selected) >= max_items:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def build_message_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any] | None:
    codes = query_message_codes(query)
    if not codes:
        return None
    code = _message_definition_key(codes[0])
    definition = MESSAGE_DEFINITIONS.get(code)
    if not definition:
        return None

    citation_hits = _select_message_citation_hits(code, definition, hits)
    citations = citations_from_hits(citation_hits)

    if language == "en":
        related = "\n".join(f"- `{rel_code}`: {rel_name}" for rel_code, rel_name in definition.get("related", []))
        answer = (
            f"`{code}` is the ISO 20022 `{definition['name']}` message: {definition['short_en']}.\n\n"
            f"In T2S terms:\n"
            f"- **Operational role:** {definition['purpose_en']}\n"
            f"- **Sender/receiver:** {definition['sender_en']}\n"
            f"- **Lifecycle:** {definition['flow_en']}\n"
            f"- **Do not confuse it with:** {definition['not_en']}\n\n"
            f"Related messages:\n{related}"
        )
    else:
        related = "\n".join(f"- `{rel_code}`: {rel_name}" for rel_code, rel_name in definition.get("related", []))
        answer = (
            f"`{code}` es el mensaje ISO 20022 `{definition['name']}`: {definition['short_es']}.\n\n"
            f"En T2S, en concreto:\n"
            f"- **Papel operativo:** {definition['purpose_es']}\n"
            f"- **Quien lo envia:** {definition['sender_es']}\n"
            f"- **Flujo:** {definition['flow_es']}\n"
            f"- **No lo confundas con:** {definition['not_es']}\n\n"
            f"Mensajes relacionados:\n{related}"
        )

    answer += source_block(citations, language)
    return {"answer": answer, "citations": citations, "confidence": "high", "answer_type": "message_definition"}


def build_answer(query: str, hits: list[Hit], language: str = "es", corpus_domain: bool = False) -> dict[str, Any]:
    if not is_domain_query(query) and not corpus_domain and not has_domain_evidence(query, hits):
        msg = (
            "I cannot see a T2S term, message, acronym, release or process in that question. Ask with the specific T2S concept and I will answer from the local corpus."
            if language == "en"
            else "No veo un termino, mensaje, acronimo, release o proceso T2S en esa pregunta. Pon el concepto T2S concreto y respondo con el corpus local."
        )
        return {"answer": msg, "citations": [], "confidence": "low", "skip_generation": True}
    if not hits:
        msg = "I cannot find enough evidence in the local T2S index." if language == "en" else "No aparece evidencia suficiente en el indice local de T2S."
        return {"answer": msg, "citations": [], "confidence": "low"}
    message_answer = build_message_answer(query, hits, language=language)
    if message_answer:
        return message_answer
    term_answer = build_term_answer(query, hits, language=language)
    if term_answer:
        return term_answer
    return build_synthetic_answer(query, hits, language=language)


def prioritize_generation_hits(query: str, hits: list[Hit], max_hits: int) -> list[Hit]:
    reranked = rerank_context_hits(query, hits, max_total=max(len(hits), max_hits))
    return reranked[:max_hits]


def build_codex_context(query: str, hits: list[Hit], max_hits: int = GENERATION_CONTEXT_HITS) -> dict[str, Any]:
    hits = prioritize_generation_hits(query, hits, max_hits=max_hits)
    evidence = []
    for n, hit in enumerate(hits[:max_hits], start=1):
        evidence.append(
            {
                "ref": n,
                "score": round(hit.score, 4),
                "retrieval_reason": hit.reason or "hybrid",
                "citation": cite_label(hit),
                "title": hit.chunk.get("title"),
                "family": hit.chunk.get("family"),
                "release": hit.chunk.get("release"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=3000),
            }
        )
    return {
        "question": query,
        "retrieval_pipeline": "hybrid TF-IDF word + TF-IDF char + BM25 + metadata boosts + local rerank + neighbor expansion",
        "instructions_for_codex_high": (
            "Answer using only the local T2S evidence dossier. Cite every substantive claim with [n]. "
            "If evidence is weak or absent, say so."
        ),
        "evidence": evidence,
    }


def format_chat_history(chat_history: list[dict[str, str]] | None, language: str) -> str:
    if not chat_history:
        return ""
    labels = {"user": "Usuario" if language == "es" else "User", "assistant": "Asistente" if language == "es" else "Assistant"}
    lines: list[str] = []
    for turn in chat_history[-10:]:
        role = str(turn.get("role", "")).lower()
        if role not in labels:
            continue
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()
        if content:
            lines.append(f"{labels[role]}: {content[:1600]}")
    if not lines:
        return ""
    title = (
        "Conversacion reciente, solo para resolver referencias e intencion de seguimiento:"
        if language == "es"
        else "Recent conversation, only to resolve follow-up references and intent:"
    )
    return title + "\n" + "\n".join(lines)


def build_generation_prompt(
    query: str,
    hits: list[Hit],
    language: str = "es",
    max_hits: int = GENERATION_CONTEXT_HITS,
    chat_history: list[dict[str, str]] | None = None,
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if language == "auto":
        language = detect_question_language(query)
    lang_name = "Spanish" if language == "es" else "English"
    ranking_query = context_query or query
    context = build_codex_context(ranking_query, hits, max_hits=max_hits)
    evidence_lines = []
    for item in context["evidence"]:
        evidence_lines.append(
            "\n".join(
                [
                    f"[{item['ref']}] {item['citation']}",
                    f"Retrieval: {item['retrieval_reason']} | score={item['score']}",
                    f"Local path: {item['local_path']}",
                    f"Source URL: {item['source_url']}",
                    f"Excerpt: {item['excerpt']}",
                ]
            )
        )
    history_block = format_chat_history(chat_history, language)
    intent = infer_question_intent(query, chat_history)
    draft_block = ""
    if draft_answer:
        draft_block = (
            "Structured local draft. Use it as a safety rail for terminology and coverage. "
            "Improve it if it is too short, too list-like or too shallow; do not copy it mechanically:\n"
            + re.sub(r"\s+", " ", draft_answer).strip()[:5000]
            + "\n\n"
        )
    style_rules = (
        "Write in Spanish with a natural technical-assistant voice. Start with the answer, then develop it. "
        "Analyse the user's intention before writing: if the user asks 'que es', define; if they ask for impact, explain impact; if they ask for a flow, sequence the process; if it is a follow-up, use the conversation to resolve what 'eso', 'este' or 'lo anterior' refers to. "
        "Do not answer as a document search result. The sources support the answer; they are not the answer. "
        "Keep the main answer concise and concrete unless the user asks for detail. Put sources only at the end, not inline inside the explanatory paragraphs. "
        "Use official T2S names in English when the documents use them. If something is missing, say 'No aparece en la documentacion local recuperada'."
        if language == "es"
        else "Write in English with a natural, direct technical-assistant voice. Start with the answer, then develop it. "
        "Analyse the user's intention before writing: if they ask what something is, define it; if they ask for impact, explain impact; if they ask for a flow, sequence the process; if it is a follow-up, use the conversation to resolve references. "
        "Do not answer as a document search result. The sources support the answer; they are not the answer. "
        "Keep the main answer concise and concrete unless the user asks for detail. Put sources only at the end, not inline inside explanatory paragraphs. "
        "Use official T2S names. If something is missing, say 'I cannot find it in the retrieved local documentation'."
    )
    return f"""You are a senior T2S documentation assistant running inside a local T2S documentation repository.

Answer the user's question in {lang_name}, naturally and directly, like a high-quality ChatGPT answer.
Use only the local T2S corpus: the evidence dossier below and the listed local paths as read-only pointers. Do not invent facts and do not use the internet.
Do not dump raw excerpts. Synthesize the answer.
The retrieval layer is deliberately generous. Treat it as a set of pointers to the right documents, then use your own reasoning to connect the facts, resolve follow-up references and produce the best answer.
Intent detected by the product: {intent}
Put references at the end with title, page/unit, and local path. Do not make sources the answer.
{style_rules}

User question:
{query}

{history_block + chr(10) if history_block else ""}Local evidence dossier:

{draft_block}
{chr(10).join(evidence_lines)}
"""


def generate_with_codex(
    query: str,
    hits: list[Hit],
    language: str = "es",
    timeout: int | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
    draft_answer: str | None = None,
    context_query: str | None = None,
) -> str:
    if os.environ.get("T2S_DISABLE_CODEX", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Codex generation disabled by T2S_DISABLE_CODEX")
    if not shutil.which("codex.cmd") and not shutil.which("codex"):
        raise RuntimeError("codex CLI not found")
    prompt = build_generation_prompt(
        query,
        hits,
        language=language,
        max_hits=GENERATION_CONTEXT_HITS,
        chat_history=chat_history,
        draft_answer=draft_answer,
        context_query=context_query,
    )
    timeout = timeout or int(os.environ.get("T2S_CODEX_TIMEOUT", "180"))
    preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])
    reasoning = preset["reasoning"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        output_path = Path(tmp.name)
    cmd = [
        CODEX,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "read-only",
        "--color",
        "never",
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-o",
        str(output_path),
        "-",
    ]
    model = os.environ.get("T2S_CODEX_MODEL", "").strip()
    if model:
        cmd[2:2] = ["-m", model]
    try:
        result = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            output = (result.stderr or result.stdout or "").strip()
            if len(output) > 1800:
                output = output[:700] + "\n...\n" + output[-1000:]
            raise RuntimeError(output)
        answer = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not answer:
            raise RuntimeError("codex returned an empty answer")
        return answer
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _retrieve_context(query: str, top_k: int, generate: bool) -> list[Hit]:
    index = load_index()
    retrieval_k = max(top_k, GENERATION_RETRIEVAL_HITS) if generate else top_k
    hits = retrieve(index, query, top_k=retrieval_k)
    hits = augment_hits(index, query, hits)
    max_context = max(retrieval_k + 16, top_k, GENERATION_CONTEXT_HITS if generate else top_k)
    hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(max_context, MAX_CONTEXT_HITS))
    return rerank_context_hits(query, hits, max_total=min(max_context, MAX_CONTEXT_HITS))


def answer_question(
    query: str,
    top_k: int = 16,
    language: str = "auto",
    generate: bool = False,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
) -> dict[str, Any]:
    resolved_language = detect_question_language(query) if language == "auto" else language
    if is_easter_egg_query(query):
        return {
            "answer": random.choice(EASTER_EGG_RESPONSES),
            "citations": [],
            "confidence": "high",
            "answer_type": "easter_egg",
            "generated_by": "easter_egg",
            "question": query,
            "language": resolved_language,
            "model": model_preset,
            "hits": [],
        }
    search_query = retrieval_query or query
    if model_preset == "local_rag":
        generate = False
    corpus_domain = is_index_domain_query(query)
    hits = _retrieve_context(search_query, top_k=top_k, generate=generate)
    payload = build_answer(query, hits, language=resolved_language, corpus_domain=corpus_domain)
    if generate and hits and payload.get("confidence") != "low":
        draft = payload.get("answer") or None
        generation_hits = prioritize_generation_hits(search_query, hits, max_hits=GENERATION_CONTEXT_HITS)
        try:
            payload["answer"] = generate_with_codex(
                query,
                generation_hits,
                language=resolved_language,
                chat_history=chat_history,
                model_preset=model_preset,
                draft_answer=draft,
                context_query=search_query,
            )
            payload["citations"] = citations_from_hits(generation_hits[: min(8, len(generation_hits))])
            payload["generated_by"] = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])["label"].lower().replace(" ", "_")
            payload.pop("skip_generation", None)
        except Exception as exc:
            payload["generated_by"] = "fallback_extractivo"
            payload["generator_error"] = str(exc)
    elif payload.get("skip_generation"):
        payload["generated_by"] = "structured"
    elif model_preset == "local_rag":
        payload["generated_by"] = "local_rag"
    payload["question"] = query
    payload["language"] = resolved_language
    payload["model"] = model_preset
    payload["hits"] = [
        {
            "rank": hit.rank,
            "score": round(hit.score, 4),
            "reason": hit.reason,
            "citation": cite_label(hit),
            "chunk": {
                key: hit.chunk.get(key)
                for key in [
                    "chunk_id",
                    "doc_id",
                    "title",
                    "category",
                    "family",
                    "release",
                    "revision_status",
                    "unit_type",
                    "unit",
                    "local_path",
                    "source_url",
                    "context_path",
                ]
            },
            "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=900),
        }
        for hit in hits
    ]
    return payload


def read_question(args: argparse.Namespace) -> str:
    if args.question:
        return " ".join(args.question).strip()
    return sys.stdin.read().strip()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Ask questions against the local T2S index.")
    parser.add_argument("question", nargs="*", help="question; if omitted, stdin is used")
    parser.add_argument("--json", action="store_true", help="print full JSON answer")
    parser.add_argument("--context", action="store_true", help="print optimized evidence JSON for Codex High")
    parser.add_argument("--generate", action="store_true", help="generate a conversational answer with Codex High")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lang", choices=["auto", "es", "en"], default="auto")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="codex_high", help="generation preset")
    args = parser.parse_args(argv)
    query = read_question(args)
    if not query:
        print("ERROR: empty question", file=sys.stderr)
        return 2
    try:
        hits = _retrieve_context(query, top_k=args.top_k, generate=args.generate)
        if args.context:
            print(json.dumps(build_codex_context(query, hits, max_hits=args.top_k), ensure_ascii=False, indent=2))
            return 0
        payload = answer_question(query, top_k=args.top_k, language=args.lang, generate=args.generate, model_preset=args.model)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["answer"])
        return 0
    except Exception as exc:
        language = detect_question_language(query)
        message = (
            f"I could not complete this query, but the CLI stayed alive. Try a more specific T2S term or run with --context. Error: {exc}"
            if language == "en"
            else f"No he podido completar esta consulta, pero el CLI sigue vivo. Prueba con un termino T2S mas concreto o ejecuta con --context. Error: {exc}"
        )
        if args.json:
            print(json.dumps({"question": query, "answer": message, "error": str(exc), "confidence": "low"}, ensure_ascii=False, indent=2))
            return 0
        print(message)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
