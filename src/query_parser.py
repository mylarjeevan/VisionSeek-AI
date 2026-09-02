"""Boolean query parser for structured object search.

Supports AND, OR, NOT, parentheses, and multi-word class names.
Does not interpret clothing or color attributes as YOLO classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.coco_classes import (
    AMBIGUOUS_COLOR_CLASSES,
    CLASS_ALIASES,
    COCO_CLASSES,
    DESCRIPTIVE_HINTS,
    QUERY_STOPWORDS,
)


AND, OR, NOT, LPAREN, RPAREN, TERM = "AND", "OR", "NOT", "LPAREN", "RPAREN", "TERM"


@dataclass
class Token:
    kind: str
    value: str
    known: bool = True


@dataclass
class ParseResult:
    raw: str
    ast: dict[str, Any] | None
    mode_hint: str
    extracted_classes: list[str]
    unknown_terms: list[str]
    semantic_text: str
    error: str | None = None
    has_operators: bool = False


def _match_table(extra: list[str] | None) -> list[tuple[str, str]]:
    """Longest-first (surface form, canonical class) pairs."""
    mapping: dict[str, str] = {c.lower(): c.lower() for c in COCO_CLASSES}
    for alias, canonical in CLASS_ALIASES.items():
        mapping[alias.lower()] = canonical.lower()
    for item in extra or []:
        if item:
            mapping[str(item).lower()] = str(item).lower()
    return sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)


def tokenize(query: str, known_classes: list[str] | None = None) -> list[Token]:
    text = (query or "").strip()
    if not text:
        return []

    classes = _match_table(known_classes)
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text[i] == "(":
            tokens.append(Token(LPAREN, "("))
            i += 1
            continue
        if text[i] == ")":
            tokens.append(Token(RPAREN, ")"))
            i += 1
            continue

        rest = text[i:]
        upper = rest.upper()
        op_match = re.match(r"^(AND|OR|NOT|WITHOUT|EXCEPT)\b", upper)
        if op_match:
            op = op_match.group(1)
            if op in {"WITHOUT", "EXCEPT"}:
                tokens.append(Token(NOT, NOT))
            else:
                tokens.append(Token(op, op))
            i += op_match.end()
            continue

        matched = None
        rest_l = rest.lower()
        for surface, canonical in classes:
            if rest_l.startswith(surface):
                end = len(surface)
                nxt = rest[end : end + 1]
                if end == len(rest) or not (nxt.isalnum()):
                    matched = canonical
                    i += end
                    break
        if matched:
            tokens.append(Token(TERM, matched, known=True))
            continue

        word = re.match(r"^[A-Za-z0-9_\-']+", rest)
        if not word:
            i += 1
            continue
        value = word.group(0).lower()
        i += word.end()
        if value in QUERY_STOPWORDS:
            continue
        tokens.append(Token(TERM, value, known=False))
    return tokens


class QueryParser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token | None:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def eat(self, kind: str | None = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of query.")
        if kind and tok.kind != kind and tok.value != kind:
            raise ValueError(f"Expected {kind}, found '{tok.value}'.")
        self.pos += 1
        return tok

    def parse(self) -> dict[str, Any]:
        if not self.tokens:
            raise ValueError("Query is empty.")
        node = self.parse_or()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token '{self.peek().value}'.")
        return node

    def parse_or(self) -> dict[str, Any]:
        nodes = [self.parse_and()]
        while self.peek() and self.peek().value == OR:
            self.eat()
            nodes.append(self.parse_and())
        if len(nodes) == 1:
            return nodes[0]
        return {"operator": "OR", "conditions": nodes}

    def parse_and(self) -> dict[str, Any]:
        nodes = [self.parse_not()]
        while True:
            nxt = self.peek()
            if nxt is None:
                break
            if nxt.value == AND:
                self.eat()
                nodes.append(self.parse_not())
                continue
            # Implicit AND between adjacent terms / groups / NOT
            if nxt.kind in {TERM, LPAREN} or nxt.value == NOT:
                nodes.append(self.parse_not())
                continue
            break
        if len(nodes) == 1:
            return nodes[0]
        return {"operator": "AND", "conditions": nodes}

    def parse_not(self) -> dict[str, Any]:
        nxt = self.peek()
        if nxt and nxt.value == NOT:
            self.eat()
            return {"operator": "NOT", "conditions": [self.parse_not()]}
        return self.parse_primary()

    def parse_primary(self) -> dict[str, Any]:
        nxt = self.peek()
        if nxt is None:
            raise ValueError("Query ended too early.")
        if nxt.kind == LPAREN:
            self.eat()
            node = self.parse_or()
            closing = self.peek()
            if closing is None or closing.kind != RPAREN:
                raise ValueError("Missing closing parenthesis.")
            self.eat()
            return node
        if nxt.kind == TERM:
            tok = self.eat()
            return {"operator": "TERM", "value": tok.value, "known": tok.known}
        raise ValueError(f"Unexpected token '{nxt.value}'.")


def _collect_classes(node: dict[str, Any], out: list[str]) -> None:
    if node.get("operator") == "TERM":
        if node.get("known", True):
            out.append(node["value"])
        return
    for child in node.get("conditions") or []:
        _collect_classes(child, out)


def parse_query(query: str, known_classes: list[str] | None = None) -> ParseResult:
    raw = (query or "").strip()
    if not raw:
        return ParseResult(
            raw=raw,
            ast=None,
            mode_hint="semantic",
            extracted_classes=[],
            unknown_terms=[],
            semantic_text="",
            error="Enter a search query.",
        )

    tokens = tokenize(raw, known_classes)
    has_operators = any(t.value in {AND, OR, NOT} or t.kind in {LPAREN, RPAREN} for t in tokens)
    unknown = [t.value for t in tokens if t.kind == TERM and not t.known]
    known_terms = [t.value for t in tokens if t.kind == TERM and t.known]
    words = [t.value for t in tokens if t.kind == TERM]
    descriptive = any(w in DESCRIPTIVE_HINTS for w in words)

    if not tokens:
        return ParseResult(
            raw=raw,
            ast=None,
            mode_hint="semantic",
            extracted_classes=[],
            unknown_terms=[],
            semantic_text=raw,
            error="Could not read that query.",
        )

    try:
        ast = QueryParser(tokens).parse()
    except ValueError as exc:
        return ParseResult(
            raw=raw,
            ast=None,
            mode_hint="semantic" if not has_operators else "structured",
            extracted_classes=known_terms,
            unknown_terms=unknown,
            semantic_text=raw,
            error=str(exc),
            has_operators=has_operators,
        )

    extracted: list[str] = []
    _collect_classes(ast, extracted)

    colorish = [t for t in known_terms if t in AMBIGUOUS_COLOR_CLASSES]
    object_terms = [t for t in extracted if t not in AMBIGUOUS_COLOR_CLASSES]
    if colorish and object_terms and not has_operators:
        unknown = list(dict.fromkeys(unknown + colorish))
        known_terms = object_terms
        extracted = object_terms
        ast = {
            "operator": "AND",
            "conditions": [{"operator": "TERM", "value": c, "known": True} for c in object_terms],
        }
        if len(object_terms) == 1:
            ast = ast["conditions"][0]
        descriptive = True

    if has_operators:
        if unknown:
            return ParseResult(
                raw=raw,
                ast=ast,
                mode_hint="structured",
                extracted_classes=list(dict.fromkeys(known_terms)),
                unknown_terms=unknown,
                semantic_text=raw,
                error=(
                    "Unknown object class in a boolean query: "
                    + ", ".join(sorted(set(unknown)))
                    + ". Use Semantic or Auto for descriptions such as clothing or color."
                ),
                has_operators=True,
            )
        hint = "structured"
    elif descriptive or unknown:
        hint = "hybrid" if known_terms else "semantic"
        if known_terms:
            terms = [{"operator": "TERM", "value": c, "known": True} for c in dict.fromkeys(known_terms)]
            ast = terms[0] if len(terms) == 1 else {"operator": "AND", "conditions": terms}
        else:
            ast = None
    else:
        hint = "structured"

    return ParseResult(
        raw=raw,
        ast=ast,
        mode_hint=hint,
        extracted_classes=list(dict.fromkeys(extracted)),
        unknown_terms=unknown,
        semantic_text=raw,
        has_operators=has_operators,
    )


def ast_to_text(node: dict[str, Any] | None) -> str:
    if not node:
        return ""
    op = node.get("operator")
    if op == "TERM":
        return str(node.get("value", ""))
    if op == "NOT":
        inner = ast_to_text((node.get("conditions") or [None])[0])
        return f"NOT {inner}"
    parts = [ast_to_text(c) for c in node.get("conditions") or []]
    joiner = f" {op} "
    body = joiner.join(parts)
    if op in {"AND", "OR"} and len(parts) > 1:
        return f"({body})"
    return body


def builder_to_query(include: list[str], combine: str, exclude: list[str]) -> str:
    include = [c for c in include if c]
    exclude = [c for c in exclude if c]
    op = "AND" if str(combine).upper() == "AND" else "OR"
    left = f" {op} ".join(include) if include else ""
    nots = " AND ".join(f"NOT {c}" for c in exclude)
    if left and nots:
        if op == "OR" and len(include) > 1:
            return f"({left}) AND {nots}"
        return f"{left} AND {nots}"
    return left or nots
