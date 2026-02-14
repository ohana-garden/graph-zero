"""
Bahá'í Sacred Text Ingest Pipeline

Downloads docx from Bahá'í Reference Library → chunks → embeds via Voyage AI
→ scores virtues via Groq → stores in graph_zero FalkorDB.

Usage:
    result = await ingest_work("hidden-words", db, voyage_key, groq_key)
"""

import hashlib
import io
import json
import os
import re
import time
from typing import Optional

import httpx

# Lazy imports for docx parsing
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


# ============================================================
# Work catalog
# ============================================================

BASE = "https://www.bahai.org/library/authoritative-texts"

WORKS = {
    # Bahá'u'lláh
    "hidden-words": {
        "title": "The Hidden Words",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/hidden-words/hidden-words.docx?1feaeaf7",
        "layer": "bedrock",
        "chunk_style": "verse",  # each verse is a chunk
    },
    "kitab-i-iqan": {
        "title": "The Kitáb-i-Íqán",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/kitab-i-iqan/kitab-i-iqan.docx?9de83dd2",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "kitab-i-aqdas": {
        "title": "The Kitáb-i-Aqdas",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/kitab-i-aqdas/kitab-i-aqdas.docx?791e5c95",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "gleanings": {
        "title": "Gleanings from the Writings of Bahá'u'lláh",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/gleanings-writings-bahaullah/gleanings-writings-bahaullah.docx?51e17b1c",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "seven-valleys": {
        "title": "The Seven Valleys and The Four Valleys",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/seven-valleys-four-valleys/seven-valleys-four-valleys.docx",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "prayers-meditations": {
        "title": "Prayers and Meditations",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/prayers-meditations/prayers-meditations.docx?e2b54cc5",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "epistle-son-wolf": {
        "title": "Epistle to the Son of the Wolf",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/epistle-son-wolf/epistle-son-wolf.docx?d4a11421",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "tablets-bahaullah": {
        "title": "Tablets of Bahá'u'lláh",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/tablets-bahaullah/tablets-bahaullah.docx?63cbbeee",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "summons-lord-hosts": {
        "title": "The Summons of the Lord of Hosts",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/summons-lord-hosts/summons-lord-hosts.docx?a8db37aa",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "call-divine-beloved": {
        "title": "The Call of the Divine Beloved",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/call-divine-beloved/call-divine-beloved.docx?fe5641dd",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "gems-divine-mysteries": {
        "title": "Gems of Divine Mysteries",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/gems-divine-mysteries/gems-divine-mysteries.docx?27a6fd50",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "tabernacle-unity": {
        "title": "The Tabernacle of Unity",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/tabernacle-unity/tabernacle-unity.docx?c868154c",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "days-remembrance": {
        "title": "Days of Remembrance",
        "author": "Bahá'u'lláh",
        "url": f"{BASE}/bahaullah/days-remembrance/days-remembrance.docx?5f4c4b3b",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    # The Báb
    "selections-bab": {
        "title": "Selections from the Writings of the Báb",
        "author": "The Báb",
        "url": f"{BASE}/the-bab/selections-writings-bab/selections-writings-bab.docx?5d307111",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    # 'Abdu'l-Bahá
    "some-answered-questions": {
        "title": "Some Answered Questions",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/some-answered-questions/some-answered-questions.docx?6d585466",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "selections-abdul-baha": {
        "title": "Selections from the Writings of 'Abdu'l-Bahá",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/selections-writings-abdul-baha/selections-writings-abdul-baha.docx?81738aff",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "paris-talks": {
        "title": "Paris Talks",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/paris-talks/paris-talks.docx?0117ef45",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "secret-divine-civilization": {
        "title": "The Secret of Divine Civilization",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/secret-divine-civilization/secret-divine-civilization.docx?21b11b89",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "promulgation-universal-peace": {
        "title": "The Promulgation of Universal Peace",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/promulgation-universal-peace/promulgation-universal-peace.docx?e2cc1a44",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "tablets-divine-plan": {
        "title": "Tablets of the Divine Plan",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/tablets-divine-plan/tablets-divine-plan.docx?0bdfa310",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "will-testament": {
        "title": "Will and Testament of 'Abdu'l-Bahá",
        "author": "'Abdu'l-Bahá",
        "url": f"{BASE}/abdul-baha/will-testament-abdul-baha/will-testament-abdul-baha.docx?9c12164e",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    # Shoghi Effendi
    "advent-divine-justice": {
        "title": "The Advent of Divine Justice",
        "author": "Shoghi Effendi",
        "url": f"{BASE}/shoghi-effendi/advent-divine-justice/advent-divine-justice.docx?21cdec30",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "world-order": {
        "title": "The World Order of Bahá'u'lláh",
        "author": "Shoghi Effendi",
        "url": f"{BASE}/shoghi-effendi/world-order-bahaullah/world-order-bahaullah.docx?1eadb7ad",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "god-passes-by": {
        "title": "God Passes By",
        "author": "Shoghi Effendi",
        "url": f"{BASE}/shoghi-effendi/god-passes-by/god-passes-by.docx?c911e494",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
    "promised-day-come": {
        "title": "The Promised Day Is Come",
        "author": "Shoghi Effendi",
        "url": f"{BASE}/shoghi-effendi/promised-day-come/promised-day-come.docx?f4949c42",
        "layer": "bedrock",
        "chunk_style": "paragraph",
    },
}

# Virtue names in Graph Zero order
VIRTUES = [
    "unity", "justice", "truthfulness", "love",
    "detachment", "humility", "compassion", "wisdom", "service"
]


# ============================================================
# Chunking
# ============================================================

def _is_boilerplate(text: str) -> bool:
    """Filter out non-content paragraphs."""
    lower = text.lower()
    skip_patterns = [
        "this document has been downloaded",
        "bahá'í reference library",
        "terms of use",
        "copyright",
        "last modified",
        "www.bahai.org",
        "legal information",
    ]
    return any(p in lower for p in skip_patterns) or len(text) < 15


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def chunk_docx(docx_bytes: bytes, work_id: str, chunk_style: str = "paragraph",
               min_chunk_tokens: int = 50, max_chunk_tokens: int = 500) -> list[dict]:
    """Parse docx bytes into text chunks with metadata."""
    doc = DocxDocument(io.BytesIO(docx_bytes))
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # Filter boilerplate
    paras = [p for p in paras if not _is_boilerplate(p)]

    chunks = []

    if chunk_style == "verse":
        # Hidden Words style: number + address + verse body
        # Pattern: "1\tO SON OF SPIRIT!" followed by verse text
        current_address = ""
        current_text = ""
        verse_num = 0

        for p in paras:
            # Detect verse header (tab-separated number + address)
            m = re.match(r'^(\d+)\t(.+)$', p)
            if m:
                # Save previous verse
                if current_text:
                    chunk_id = f"{work_id}_v{verse_num}"
                    full_text = f"{current_address}\n{current_text}" if current_address else current_text
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": full_text.strip(),
                        "section": current_address,
                        "verse_num": verse_num,
                    })
                verse_num = int(m.group(1))
                current_address = m.group(2)
                current_text = ""
            elif verse_num > 0:
                # This is verse body
                current_text += (" " if current_text else "") + p
            else:
                # Preamble - still useful content
                if _estimate_tokens(p) >= min_chunk_tokens:
                    chunks.append({
                        "chunk_id": f"{work_id}_preamble_{len(chunks)}",
                        "text": p,
                        "section": "Preamble",
                    })

        # Don't forget last verse
        if current_text:
            chunk_id = f"{work_id}_v{verse_num}"
            full_text = f"{current_address}\n{current_text}" if current_address else current_text
            chunks.append({
                "chunk_id": chunk_id,
                "text": full_text.strip(),
                "section": current_address,
                "verse_num": verse_num,
            })

    else:
        # Paragraph style: combine short paragraphs, split long ones
        buffer = ""
        section = "Main"

        for p in paras:
            # Detect section headers (short, often ALL CAPS or Roman numerals)
            if (len(p) < 80 and (p.isupper() or re.match(r'^[IVXLCDM]+\.?\s', p)
                                 or re.match(r'^(Part|Chapter|Section)\s', p, re.I))):
                # Flush buffer
                if buffer and _estimate_tokens(buffer) >= min_chunk_tokens:
                    chunks.append({
                        "chunk_id": f"{work_id}_c{len(chunks)}",
                        "text": buffer.strip(),
                        "section": section,
                    })
                    buffer = ""
                section = p
                continue

            # Add to buffer
            if buffer:
                combined = buffer + " " + p
                if _estimate_tokens(combined) > max_chunk_tokens:
                    # Flush current buffer
                    if _estimate_tokens(buffer) >= min_chunk_tokens:
                        chunks.append({
                            "chunk_id": f"{work_id}_c{len(chunks)}",
                            "text": buffer.strip(),
                            "section": section,
                        })
                    buffer = p
                else:
                    buffer = combined
            else:
                buffer = p

        # Flush remaining
        if buffer and _estimate_tokens(buffer) >= min_chunk_tokens:
            chunks.append({
                "chunk_id": f"{work_id}_c{len(chunks)}",
                "text": buffer.strip(),
                "section": section,
            })

    return chunks


# ============================================================
# Voyage AI Embedding
# ============================================================

async def embed_chunks(chunks: list[dict], api_key: str,
                       model: str = "voyage-3.5",
                       batch_size: int = 128) -> list[dict]:
    """Embed chunk texts via Voyage AI. Returns chunks with 'embedding' added.
    
    Free tier: 3 RPM, 10K TPM. We batch aggressively and add delays.
    """
    import asyncio
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Estimate tokens and split into batches under 10K tokens each
    batches = []
    current_batch = []
    current_tokens = 0
    max_tokens_per_req = 9000  # leave headroom

    for c in chunks:
        est_tokens = len(c["text"]) // 4
        if current_tokens + est_tokens > max_tokens_per_req and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(c)
        current_tokens += est_tokens

    if current_batch:
        batches.append(current_batch)

    async with httpx.AsyncClient(timeout=60.0) as client:
        chunk_idx = 0
        for batch_num, batch in enumerate(batches):
            if batch_num > 0:
                # Rate limit: wait 25s between requests (3 RPM = 1 per 20s)
                await asyncio.sleep(25)

            texts = [c["text"] for c in batch]

            resp = await client.post(url, headers=headers, json={
                "input": texts,
                "model": model,
                "input_type": "document",
            })

            if resp.status_code == 429:
                # Rate limited - wait a full minute and retry
                await asyncio.sleep(65)
                resp = await client.post(url, headers=headers, json={
                    "input": texts,
                    "model": model,
                    "input_type": "document",
                })

                if resp.status_code == 429:
                    # Still limited - wait another minute
                    await asyncio.sleep(65)
                    resp = await client.post(url, headers=headers, json={
                        "input": texts,
                        "model": model,
                        "input_type": "document",
                    })

            if resp.status_code != 200:
                raise RuntimeError(f"Voyage API error {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            for j, emb_data in enumerate(data["data"]):
                # Map back to original chunks list
                idx = sum(len(b) for b in batches[:batch_num]) + j
                chunks[idx]["embedding"] = emb_data["embedding"]

    return chunks


# ============================================================
# Groq Virtue Scoring
# ============================================================

VIRTUE_PROMPT = """Score this sacred text passage on each of these 9 virtues.
Return ONLY a JSON object with float scores from 0.0 to 1.0.
0.0 = not at all related, 1.0 = deeply embodies this virtue.

Virtues: unity, justice, truthfulness, love, detachment, humility, compassion, wisdom, service

Text: "{text}"

JSON response (no other text):"""


async def score_virtues_batch(chunks: list[dict], api_key: str,
                              model: str = "llama-3.1-8b-instant",
                              batch_size: int = 5) -> list[dict]:
    """Score virtue relevance for each chunk via Groq.
    Returns chunks with 'virtue_scores' list added."""
    import asyncio
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    default_scores = [0.5] * 9  # fallback

    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, chunk in enumerate(chunks):
            text_preview = chunk["text"][:800]
            try:
                resp = await client.post(url, headers=headers, json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": VIRTUE_PROMPT.format(text=text_preview)}
                    ],
                    "max_tokens": 200,
                    "temperature": 0.1,
                })

                if resp.status_code == 429:
                    await asyncio.sleep(5)
                    resp = await client.post(url, headers=headers, json={
                        "model": model,
                        "messages": [
                            {"role": "user", "content": VIRTUE_PROMPT.format(text=text_preview)}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.1,
                    })

                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"].strip()
                    content = re.sub(r'```json\s*', '', content)
                    content = re.sub(r'```\s*', '', content)
                    scores_dict = json.loads(content)
                    scores = [float(scores_dict.get(v, 0.5)) for v in VIRTUES]
                    scores = [max(0.0, min(1.0, s)) for s in scores]
                    chunks[i]["virtue_scores"] = scores
                else:
                    chunks[i]["virtue_scores"] = default_scores

            except Exception:
                chunks[i]["virtue_scores"] = default_scores

            # Rate limit: ~20 RPM on free tier = 1 per 3s
            if i < len(chunks) - 1:
                await asyncio.sleep(2)

    return chunks


# ============================================================
# Store in FalkorDB
# ============================================================

def store_chunks(chunks: list[dict], work_meta: dict, graph) -> int:
    """Store processed chunks as TerrainNodes in graph_zero."""
    stored = 0
    work_id = work_meta.get("work_id", "unknown")
    title = work_meta.get("title", "")
    author = work_meta.get("author", "")
    layer = work_meta.get("layer", "bedrock")

    for chunk in chunks:
        node_id = f"t_{chunk['chunk_id']}"
        text = chunk["text"]
        embedding = chunk.get("embedding", [])
        virtue_scores = chunk.get("virtue_scores", [0.5] * 9)
        section = chunk.get("section", "")

        # Build properties
        props = {
            "source_text": text,
            "text": text,  # keep both for compatibility
            "source_work": title,
            "author": author,
            "layer": layer,
            "terrain_role": "bedrock",
            "section": section,
            "work_id": work_id,
            "provenance_type": "BEDROCK",
            "authority_weight": 1.0,
            "virtue_scores": virtue_scores,
            "momentum": [0.0] * 9,
            "kala_accumulated": 0.0,
            "traversal_count": 0,
            "created_at": int(time.time() * 1000),
        }

        # Store embedding separately if present
        if embedding:
            props["embedding"] = embedding
            props["embedding_dims"] = len(embedding)
            props["embedding_model"] = "voyage-3.5"

        try:
            graph.add_node(node_id, "TerrainNode", props)
            stored += 1
        except Exception as e:
            print(f"Error storing {node_id}: {e}")

    return stored


# ============================================================
# Main Pipeline
# ============================================================

async def ingest_work(work_id: str, graph, voyage_key: str, groq_key: str,
                      skip_virtues: bool = False) -> dict:
    """Full pipeline: download → chunk → embed → score → store."""
    if work_id not in WORKS:
        return {"error": f"Unknown work: {work_id}", "available": list(WORKS.keys())}

    if not HAS_DOCX:
        return {"error": "python-docx not installed"}

    work = WORKS[work_id]
    log = []
    t0 = time.time()

    # Step 1: Download
    log.append(f"Downloading {work['title']}...")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(work["url"])
        if resp.status_code != 200:
            return {"error": f"Download failed: {resp.status_code}", "url": work["url"]}
    docx_bytes = resp.content
    log.append(f"  Downloaded {len(docx_bytes)} bytes")

    # Step 2: Chunk
    chunks = chunk_docx(docx_bytes, work_id, work.get("chunk_style", "paragraph"))
    total_chars = sum(len(c["text"]) for c in chunks)
    log.append(f"  Chunked into {len(chunks)} pieces ({total_chars} chars, ~{total_chars//4} tokens)")

    # Step 3: Embed via Voyage AI
    log.append("Embedding via Voyage AI (voyage-3.5)...")
    t1 = time.time()
    chunks = await embed_chunks(chunks, voyage_key)
    embed_dims = len(chunks[0]["embedding"]) if chunks and "embedding" in chunks[0] else 0
    log.append(f"  Embedded {len(chunks)} chunks ({embed_dims} dims) in {time.time()-t1:.1f}s")

    # Step 4: Score virtues via Groq
    if not skip_virtues and groq_key:
        log.append("Scoring virtues via Groq (llama-3.1-8b-instant)...")
        t2 = time.time()
        chunks = await score_virtues_batch(chunks, groq_key)
        log.append(f"  Scored {len(chunks)} chunks in {time.time()-t2:.1f}s")
    else:
        log.append("Skipping virtue scoring")
        for c in chunks:
            c["virtue_scores"] = [0.5] * 9

    # Step 5: Store in graph_zero
    log.append("Storing in graph_zero...")
    work_meta = {"work_id": work_id, **work}
    stored = store_chunks(chunks, work_meta, graph)
    log.append(f"  Stored {stored} TerrainNodes")

    elapsed = time.time() - t0
    log.append(f"Done in {elapsed:.1f}s")

    return {
        "status": "complete",
        "work_id": work_id,
        "title": work["title"],
        "author": work["author"],
        "chunks": len(chunks),
        "stored": stored,
        "embedding_dims": embed_dims,
        "elapsed_seconds": round(elapsed, 1),
        "log": log,
    }


async def ingest_all(graph, voyage_key: str, groq_key: str,
                     skip_virtues: bool = False) -> dict:
    """Ingest all works sequentially."""
    results = {}
    total_stored = 0

    for work_id in WORKS:
        result = await ingest_work(work_id, graph, voyage_key, groq_key, skip_virtues)
        results[work_id] = result
        if result.get("stored"):
            total_stored += result["stored"]

    return {
        "status": "complete",
        "works_processed": len(results),
        "total_stored": total_stored,
        "results": results,
    }
