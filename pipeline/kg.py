"""Bonus (§13) — Knowledge Graph construction from unstructured docs.

Vector RAG retrieves *chunks*; a knowledge graph retrieves *connected facts*,
which is what wins multi-hop and global-summary questions (GraphRAG-Bench,
ICLR'26). The construction pipeline is the same shape regardless of extractor:

    text -> extract (entity, relation, entity) triples -> upsert into a graph

Here the extractor is a small **deterministic** rule matcher so the lab stays
zero-key and reproducible. In production this node is an LLM (e.g. LangChain
`LLMGraphTransformer`) with entity resolution — swap `extract_triples` and the
rest of the pipeline is unchanged. That substitutability is the lesson.
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

_PRODUCTS = ["widget", "gadget", "sprocket"]

# Entity resolution: collapse surface forms (plural, the/a) to one canonical node
# so an IS_A object ("accessory") matches the SHIPS_FROM subject ("accessories").
# A real pipeline does this with an LLM or a fuzzy matcher; here it is a small map.
_CANON = {
    "widgets": "widget", "gadgets": "gadget", "sprockets": "sprocket",
    "accessories": "accessory", "industrial-parts": "industrial-part",
}


def _canon(entity: str) -> str:
    e = entity.strip().strip(".").lower()
    e = re.sub(r"^(the|a|an)\s+", "", e)
    return _CANON.get(e, e)


def extract_triples(text: str) -> list[tuple[str, str, str]]:
    """Deterministic (subject, relation, object) extraction. LLM slots in here.

    Produces two kinds of edge:
      * entity -> literal  (RETURNABLE_WITHIN, HAS_WARRANTY, NON_RETURNABLE)
      * entity -> entity   (IS_A, SHIPS_FROM) -- these are what make the graph
        *traversable*, so a multi-hop question (widget -> accessory -> warehouse)
        can be answered by walking edges.

    A real pipeline swaps this body for an LLM (e.g. LangChain
    ``LLMGraphTransformer``) plus entity resolution; the downstream graph code is
    unchanged. That substitutability is the lesson.
    """
    triples: list[tuple[str, str, str]] = []
    sentences: list[str] = []
    for line in text.splitlines():          # skip markdown headings + split per line
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sentences.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
    for sentence in sentences:
        low = sentence.lower()

        # per-product policy edges (entity -> literal)
        for prod in _PRODUCTS:
            if prod not in low:
                continue
            ret = re.search(r"within\s+(\d+)\s*-?\s*day", low)
            if "return" in low and ret:
                triples.append((prod, "RETURNABLE_WITHIN", f"{ret.group(1)} days"))
            war = re.search(r"(\d+)\s*-?\s*day", low)
            if "warranty" in low and war:
                triples.append((prod, "HAS_WARRANTY", f"{war.group(1)} days"))
            if "final sale" in low or "cannot be returned" in low:
                triples.append((prod, "NON_RETURNABLE", "final sale"))

        # category membership (entity -> entity): "X and Y belong to the Z category"
        m = re.search(r"^(.*?) belong to the ([\w-]+) category", low)
        if m:
            cat = _canon(m.group(2))
            for subj in re.split(r",|\band\b", m.group(1)):
                subj = _canon(subj)
                if subj:
                    triples.append((subj, "IS_A", cat))

        # logistics (entity -> entity): "Z ship from the <place>"
        m = re.search(r"^([\w-]+) ship from the (.+?)[.]?$", low)
        if m:
            triples.append((_canon(m.group(1)), "SHIPS_FROM", m.group(2).strip()))

    return triples


def build_graph(triples: list[tuple[str, str, str]]) -> dict[str, list[tuple[str, str]]]:
    """Adjacency list: node -> [(relation, neighbour), ...]. Pure-Python, no deps."""
    graph: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for subj, rel, obj in triples:
        graph[subj].append((rel, obj))
    return dict(graph)


def ingest_docs_to_graph(docs_dir: Path) -> dict:
    triples: list[tuple[str, str, str]] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        triples.extend(extract_triples(path.read_text(encoding="utf-8")))
    return build_graph(triples)


def query(graph: dict, subject: str, relation: str | None = None) -> list[tuple[str, str]]:
    """1-hop lookup: facts about `subject`, optionally filtered by relation."""
    edges = graph.get(subject, [])
    return [(r, o) for (r, o) in edges if relation is None or r == relation]


def returnable_products(graph: dict) -> list[str]:
    """Connected-fact query a flat vector lookup struggles with: 'what is returnable?'

    A product is returnable iff it has a RETURNABLE_WITHIN edge and is NOT marked
    final-sale. Note a warranty (HAS_WARRANTY) does NOT make something returnable —
    conflating the two is exactly the kind of relation error a graph lets you avoid.
    """
    out = []
    for node, edges in graph.items():
        rels = {r for r, _ in edges}
        if "RETURNABLE_WITHIN" in rels and "NON_RETURNABLE" not in rels:
            out.append(node)
    return sorted(out)


def traverse(graph: dict, start: str, target_relation: str, max_hops: int = 3) -> list[dict]:
    """Real multi-hop: BFS from `start`, following entity->entity edges, until a
    `target_relation` edge is found. Returns each path that reaches the answer.

    Example: traverse(g, "widget", "SHIPS_FROM") walks
        widget --IS_A--> accessory --SHIPS_FROM--> Hanoi fulfillment center
    i.e. a 2-hop answer no single edge (and no single document chunk) contains.
    """
    from collections import deque

    results: list[dict] = []
    seen = {start}
    q: deque = deque([(start, [start])])
    while q:
        node, path = q.popleft()
        if len(path) > max_hops + 1:
            continue
        for rel, obj in graph.get(node, []):
            if rel == target_relation:
                results.append({"path": path + [obj], "hops": len(path), "answer": obj})
            if obj in graph and obj not in seen:   # obj is itself an entity -> keep hopping
                seen.add(obj)
                q.append((obj, path + [obj]))
    return results


def _sentence_chunks(docs_dir: Path) -> list[str]:
    """Sentence-level chunks (a common RAG granularity) across all docs."""
    chunks: list[str] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sent in re.split(r"(?<=[.!?])\s+", line):
                if sent.strip():
                    chunks.append(sent.strip())
    return chunks


# ---------------------------------------------------------------------------
# Extension exercise 2 — LLM-based triple extractor (§13)
# ---------------------------------------------------------------------------

def extract_triples_llm(
    text: str,
    llm_fn: Callable[[str], str],
) -> list[tuple[str, str, str]]:
    """LLM-powered triple extractor — drop-in replacement for extract_triples().

    llm_fn(prompt: str) -> str  can be any callable:
        - a real LLM: lambda p: anthropic_client.messages.create(model=..., messages=[{"role":"user","content":p}]).content[0].text
        - a local model via Ollama: lambda p: requests.post("http://localhost:11434/api/generate", json={"model":"llama3","prompt":p}).json()["response"]
        - mock_llm() below for zero-key testing

    Entity resolution runs the same _canon() map as the rule-based extractor, so
    all downstream graph code (build_graph, traverse, returnable_products) is
    unchanged. That substitutability is the lesson of §13.
    """
    prompt = (
        "Extract factual (subject, relation, object) triples from the text below.\n"
        "Output exactly one triple per line as: subject | RELATION | object\n"
        "Use UPPERCASE_SNAKE_CASE for relations (e.g. RETURNABLE_WITHIN, HAS_WARRANTY).\n"
        "Only output triples, nothing else.\n\n"
        f"Text:\n{text}\n\nTriples:"
    )
    response = llm_fn(prompt)
    triples: list[tuple[str, str, str]] = []
    for line in response.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3:
            continue
        subj, rel, obj = parts
        subj, obj = _canon(subj), _canon(obj)
        rel = rel.upper().replace(" ", "_")
        if subj and rel and obj:
            triples.append((subj, rel, obj))
    return triples


def mock_llm(prompt: str) -> str:  # noqa: ARG001
    """Deterministic mock LLM for zero-key testing of the LLM-extractor path.

    In production replace this with a real API call — the pipeline is identical.
    Shows the five triples that exercise 2 is expected to produce from the seed
    docs so tests can verify the round-trip without network access.
    """
    return (
        "widget | RETURNABLE_WITHIN | 30 days\n"
        "gadget | HAS_WARRANTY | 90 days\n"
        "sprocket | NON_RETURNABLE | final sale\n"
        "widget | IS_A | accessory\n"
        "accessories | SHIPS_FROM | hanoi fulfillment center\n"
    )


def ingest_docs_to_graph_llm(docs_dir: Path, llm_fn: Callable[[str], str] = mock_llm) -> dict:
    """Same pipeline as ingest_docs_to_graph but using LLM extraction.

    Swap llm_fn for a real model and all downstream graph queries still work.
    """
    triples: list[tuple[str, str, str]] = []
    for path in sorted(Path(docs_dir).glob("*.md")):
        triples.extend(extract_triples_llm(path.read_text(encoding="utf-8"), llm_fn))
    return build_graph(triples)


def vector_foil(docs_dir: Path, subject: str, answer_hint: str) -> dict:
    """Show *why* flat chunk-retrieval RAG loses on a multi-hop question.

    A multi-hop fact is distributed across chunks: one chunk links the subject to
    a category, another links the category to the answer. No single chunk contains
    BOTH the subject and the answer, so top-1 retrieval can only ever return half
    the chain — regardless of embedding quality. The graph encodes the join, so a
    traversal returns the whole path. This contrast is the point of §13.
    """
    subject, answer_hint = subject.lower(), answer_hint.lower()
    chunks = _sentence_chunks(docs_dir)
    with_subject = [c for c in chunks if subject in c.lower()]
    with_answer = [c for c in chunks if answer_hint in c.lower()]
    complete = [c for c in chunks if subject in c.lower() and answer_hint in c.lower()]
    return {
        "n_chunks": len(chunks),
        "chunk_with_subject": with_subject[0] if with_subject else None,
        "chunk_with_answer": with_answer[0] if with_answer else None,
        "single_chunk_answers_it": bool(complete),   # False => flat RAG cannot bridge
    }
