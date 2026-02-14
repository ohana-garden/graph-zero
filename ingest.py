"""
Graph Zero Terrain Ingest Pipeline
===================================
Scrapes Bahá'í sacred texts, chunks them intelligently,
embeds with Voyage AI, virtue-scores with Groq, 
and stores as TerrainNodes in graph_zero.

Cost estimate for full corpus (~5000 chunks):
  - Voyage AI: FREE (200M token free tier)
  - Groq Llama 3.1 8B: ~$0.06
  - Total: ~$0.06
"""

import os, re, hashlib, json, time, asyncio, logging
from typing import List, Dict, Optional, Tuple

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger("ingest")

# ── Source catalog ──────────────────────────────────────────────
# Each entry: (source_work, author, terrain_role, url)
# terrain_role: "bedrock" = Central Figures' Writings
#               "guidance" = Guardian / UHJ  
#               "community" = compilations / other

SOURCES = [
    # ── Bahá'u'lláh (12 works) ──
    ("The Hidden Words", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/hidden-words/hidden-words.xhtml"),
    ("Kitáb-i-Íqán", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/kitab-i-iqan/kitab-i-iqan.xhtml"),
    ("Kitáb-i-Aqdas", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/kitab-i-aqdas/kitab-i-aqdas.xhtml"),
    ("Gleanings from the Writings of Bahá'u'lláh", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/gleanings-writings-bahaullah/gleanings-writings-bahaullah.xhtml"),
    ("Prayers and Meditations", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/prayers-meditations/prayers-meditations.xhtml"),
    ("Tablets of Bahá'u'lláh", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/tablets-bahaullah/tablets-bahaullah.xhtml"),
    ("Epistle to the Son of the Wolf", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/epistle-son-wolf/epistle-son-wolf.xhtml"),
    ("The Call of the Divine Beloved", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/call-divine-beloved/call-divine-beloved.xhtml"),
    ("The Summons of the Lord of Hosts", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/summons-lord-hosts/summons-lord-hosts.xhtml"),
    ("Gems of Divine Mysteries", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/gems-divine-mysteries/gems-divine-mysteries.xhtml"),
    ("The Tabernacle of Unity", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/tabernacle-unity/tabernacle-unity.xhtml"),
    ("Days of Remembrance", "Bahá'u'lláh", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/bahaullah/days-remembrance/days-remembrance.xhtml"),

    # ── The Báb (1 work) ──
    ("Selections from the Writings of the Báb", "The Báb", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/the-bab/selections-writings-bab/selections-writings-bab.xhtml"),

    # ── 'Abdu'l-Bahá (12 works) ──
    ("Some Answered Questions", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/some-answered-questions/some-answered-questions.xhtml"),
    ("Selections from the Writings of 'Abdu'l-Bahá", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/selections-writings-abdul-baha/selections-writings-abdul-baha.xhtml"),
    ("The Secret of Divine Civilization", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/secret-divine-civilization/secret-divine-civilization.xhtml"),
    ("Paris Talks", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/paris-talks/paris-talks.xhtml"),
    ("Tablets of the Divine Plan", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/tablets-divine-plan/tablets-divine-plan.xhtml"),
    ("The Will and Testament of 'Abdu'l-Bahá", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/will-testament-abdul-baha/will-testament-abdul-baha.xhtml"),
    ("Memorials of the Faithful", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/memorials-faithful/memorials-faithful.xhtml"),
    ("Light of the World", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/light-of-the-world/light-of-the-world.xhtml"),
    ("The Promulgation of Universal Peace", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/promulgation-universal-peace/promulgation-universal-peace.xhtml"),
    ("Tablet to Dr. Auguste Forel", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/tablet-auguste-forel/tablet-auguste-forel.xhtml"),
    ("Tablets to The Hague", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/tablets-hague-abdul-baha/tablets-hague-abdul-baha.xhtml"),
    ("A Traveler's Narrative", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/travelers-narrative/travelers-narrative.xhtml"),
    ("Twelve Table Talks", "'Abdu'l-Bahá", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/abdul-baha/twelve-table-talks-abdul-baha/twelve-table-talks-abdul-baha.xhtml"),

    # ── Prayers (2 collections) ──
    ("Bahá'í Prayers", "Various", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/prayers/bahai-prayers/bahai-prayers.xhtml"),
    ("Bahá'í Prayers and Tablets for Children", "Various", "bedrock",
     "https://www.bahai.org/library/authoritative-texts/prayers/bahai-prayers-tablets-children/bahai-prayers-tablets-children.xhtml"),

    # ── Shoghi Effendi (6 works) ──
    ("The Advent of Divine Justice", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/advent-divine-justice/advent-divine-justice.xhtml"),
    ("The World Order of Bahá'u'lláh", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/world-order-bahaullah/world-order-bahaullah.xhtml"),
    ("The Promised Day Is Come", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/promised-day-come/promised-day-come.xhtml"),
    ("God Passes By", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/god-passes-by/god-passes-by.xhtml"),
    ("Bahá'í Administration", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/bahai-administration/bahai-administration.xhtml"),
    ("Citadel of Faith", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/citadel-faith/citadel-faith.xhtml"),
    ("This Decisive Hour", "Shoghi Effendi", "guidance",
     "https://www.bahai.org/library/authoritative-texts/shoghi-effendi/decisive-hour/decisive-hour.xhtml"),

    # ── Universal House of Justice (5 works) ──
    ("The Promise of World Peace", "Universal House of Justice", "guidance",
     "https://www.bahai.org/library/authoritative-texts/the-universal-house-of-justice/messages/19851001_001/19851001_001.xhtml"),
    ("The Institution of the Counsellors", "Universal House of Justice", "guidance",
     "https://www.bahai.org/library/authoritative-texts/the-universal-house-of-justice/the-institution-of-the-counsellors/the-institution-of-the-counsellors.xhtml"),
    ("Letter to the World's Religious Leaders", "Universal House of Justice", "guidance",
     "https://www.bahai.org/library/authoritative-texts/the-universal-house-of-justice/messages/20020401_001/20020401_001.xhtml"),
    ("Reflections on the First Century of the Formative Age", "Universal House of Justice", "guidance",
     "https://www.bahai.org/library/authoritative-texts/the-universal-house-of-justice/messages/20231128_001/20231128_001.xhtml"),
    ("Turning Point: Messages 1963-1986", "Universal House of Justice", "guidance",
     "https://www.bahai.org/library/authoritative-texts/the-universal-house-of-justice/turning-point/turning-point.xhtml"),

    # ── Compilations (25 works) ──
    ("A Chaste and Holy Life", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/chaste-holy-life/chaste-holy-life.xhtml"),
    ("Codification of the Law of Huqúqu'lláh", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/codification-law-huququllah/codification-law-huququllah.xhtml"),
    ("Bahá'í Funds and Contributions", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/bahai-funds-contributions/bahai-funds-contributions.xhtml"),
    ("Bahá'í Meetings", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/bahai-meetings/bahai-meetings.xhtml"),
    ("Consultation", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/consultation/consultation.xhtml"),
    ("The Covenant", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/covenant/covenant.xhtml"),
    ("Crisis and Victory", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/crisis-victory/crisis-victory.xhtml"),
    ("Excellence in All Things", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/excellence-all-things/excellence-all-things.xhtml"),
    ("Family Life", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/family-life/family-life.xhtml"),
    ("Fire and Light", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/fire-and-light/fire-and-light.xhtml"),
    ("Give Me Thy Grace to Serve Thy Loved Ones", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/give-me-thy-grace-serve-thy-loved-ones/give-me-thy-grace-serve-thy-loved-ones.xhtml"),
    ("Huqúqu'lláh—The Right of God", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/huququllah-right-god/huququllah-right-god.xhtml"),
    ("The Importance of the Arts", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/importance-art/importance-art.xhtml"),
    ("The Importance of Obligatory Prayer and Fasting", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/importance-obligatory-prayer-fasting/importance-obligatory-prayer-fasting.xhtml"),
    ("The Importance of Prayer, Meditation and the Devotional Attitude", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/importance-prayer-meditation-devotional-attitude/importance-prayer-meditation-devotional-attitude.xhtml"),
    ("The Institution of the Mashriqu'l-Adhkár", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/institution-mashriqul-adhkar/institution-mashriqul-adhkar.xhtml"),
    ("Issues Related to the Study of the Bahá'í Faith", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/issues-related-study-bahai-faith/issues-related-study-bahai-faith.xhtml"),
    ("The Local Spiritual Assembly", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/local-spiritual-assembly/local-spiritual-assembly.xhtml"),
    ("Music", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/music/music.xhtml"),
    ("The National Spiritual Assembly", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/national-spiritual-assembly/national-spiritual-assembly.xhtml"),
    ("Peace", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/peace/peace.xhtml"),
    ("The Power of Divine Assistance", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/power-divine-assistance/power-divine-assistance.xhtml"),
    ("Prayer and Devotional Life", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/prayer-devotional-life/prayer-devotional-life.xhtml"),
    ("The Sanctity and Nature of Bahá'í Elections", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/sanctity-nature-bahai-elections/sanctity-nature-bahai-elections.xhtml"),
    ("Scholarship", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/scholarship/scholarship.xhtml"),
    ("To Set the World in Order: Building and Preserving Strong Marriages", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/set-world-order/set-world-order.xhtml"),
    ("The Significance of the Formative Age", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/significance-formative-age-our-faith/significance-formative-age-our-faith.xhtml"),
    ("Social Action", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/social-action/social-action.xhtml"),
    ("Trustworthiness", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/trustworthiness/trustworthiness.xhtml"),
    ("The Universal House of Justice (Compilation)", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/universal-house-of-justice-compilation/universal-house-of-justice-compilation.xhtml"),
    ("Women", "Compilation", "community",
     "https://www.bahai.org/library/authoritative-texts/compilations/women/women.xhtml"),
]

# Fallback sources from bahai-library.com (if bahai.org blocks)
FALLBACK_SOURCES = [
    ("The Hidden Words", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_hidden_words"),
    ("Kitáb-i-Íqán", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_kitab-i-iqan"),
    ("Gleanings from the Writings of Bahá'u'lláh", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_gleanings"),
    ("Kitáb-i-Aqdas", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_kitab-i-aqdas"),
    ("Tablets of Bahá'u'lláh", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_tablets"),
    ("Epistle to the Son of the Wolf", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_epistle_son_wolf"),
    ("Prayers and Meditations", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_prayers_meditations"),
    ("The Call of the Divine Beloved", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_call_divine_beloved"),
    ("The Summons of the Lord of Hosts", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_summons_lord_hosts"),
    ("Gems of Divine Mysteries", "Bahá'u'lláh", "bedrock",
     "https://bahai-library.com/bahaullah_gems_divine_mysteries"),
    ("Selections from the Writings of the Báb", "The Báb", "bedrock",
     "https://bahai-library.com/bab_selections_writings_bab"),
    ("Some Answered Questions", "'Abdu'l-Bahá", "bedrock",
     "https://bahai-library.com/abdul-baha_some_answered_questions"),
    ("Selections from the Writings of 'Abdu'l-Bahá", "'Abdu'l-Bahá", "bedrock",
     "https://bahai-library.com/abdul-baha_selections_writings"),
    ("The Secret of Divine Civilization", "'Abdu'l-Bahá", "bedrock",
     "https://bahai-library.com/abdul-baha_secret_divine_civilization"),
    ("Paris Talks", "'Abdu'l-Bahá", "bedrock",
     "https://bahai-library.com/abdul-baha_paris_talks"),
    ("The Advent of Divine Justice", "Shoghi Effendi", "guidance",
     "https://bahai-library.com/shoghi-effendi_advent_divine_justice"),
    ("The World Order of Bahá'u'lláh", "Shoghi Effendi", "guidance",
     "https://bahai-library.com/shoghi-effendi_world_order_bahaullah"),
    ("God Passes By", "Shoghi Effendi", "guidance",
     "https://bahai-library.com/shoghi-effendi_god_passes_by"),
]


# ── Text extraction ─────────────────────────────────────────────

def extract_text_from_html(html: str, source_url: str) -> str:
    """Extract clean text from Bahá'í library HTML."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove scripts, styles, nav, headers, footers
    for tag in soup.find_all(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    
    # Try to find main content area
    content = None
    # bahai.org uses specific content divs
    for sel in [
        soup.find("div", class_="library-document"),
        soup.find("div", class_="content"),
        soup.find("article"),
        soup.find("main"),
        soup.find("div", id="content"),
        soup.find("div", class_="entry-content"),
        soup.find("body"),
    ]:
        if sel:
            content = sel
            break
    
    if not content:
        content = soup
    
    # Get text
    text = content.get_text(separator="\n", strip=True)
    
    # Clean up
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove common footer/nav text
    for remove in [
        "Bahá'í Reference Library",
        "Search Bahai.org",
        "Menu",
        "Previous Section",
        "Next Section",
    ]:
        text = text.replace(remove, "")
    
    return text.strip()


# ── Chunking ─────────────────────────────────────────────────────

def chunk_text(text: str, source_work: str, 
               min_chunk: int = 100, max_chunk: int = 800,
               overlap: int = 50) -> List[Dict]:
    """
    Chunk text into terrain-sized passages.
    
    Strategy:
    - Split on double newlines (paragraph boundaries)
    - Merge small paragraphs together until they reach min_chunk
    - Split long paragraphs at sentence boundaries if > max_chunk
    - Each chunk gets a stable ID from content hash
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        # Skip very short lines (headers, page numbers, etc)
        if len(para) < 20 and not any(c.isalpha() for c in para):
            continue
        
        if len(current) + len(para) < max_chunk:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            # Flush current if it's big enough
            if len(current) >= min_chunk:
                chunks.append(current)
                current = para
            elif current:
                # Current is too small, force merge
                current = (current + "\n\n" + para).strip()
            else:
                current = para
            
            # If current is now too big, split at sentences
            while len(current) > max_chunk:
                split_point = _find_sentence_break(current, max_chunk)
                chunks.append(current[:split_point].strip())
                current = current[split_point:].strip()
    
    # Don't forget the last chunk
    if current and len(current) >= min_chunk // 2:
        chunks.append(current)
    
    # Build chunk records with stable IDs
    records = []
    for i, text in enumerate(chunks):
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
        records.append({
            "chunk_id": f"{_slug(source_work)}_{i:04d}_{content_hash}",
            "text": text,
            "chunk_index": i,
        })
    
    return records


def _find_sentence_break(text: str, max_pos: int) -> int:
    """Find the best sentence break before max_pos."""
    # Look for sentence endings: . ! ? followed by space or newline
    best = max_pos
    for m in re.finditer(r'[.!?]\s', text[:max_pos]):
        best = m.end()
    if best == max_pos:
        # Fall back to any space
        space = text.rfind(' ', 0, max_pos)
        if space > max_pos // 2:
            best = space
    return best


def _slug(name: str) -> str:
    """Convert source name to URL-safe slug."""
    s = name.lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s.strip('_')[:40]


# ── Voyage AI Embeddings ────────────────────────────────────────

async def embed_batch(texts: List[str], api_key: str,
                      model: str = "voyage-3.5",
                      dims: int = 1024,
                      batch_size: int = 64) -> List[List[float]]:
    """
    Embed texts with Voyage AI.
    voyage-3.5: $0.06/M tokens, 200M free, 1024 dims default
    """
    all_embeddings = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": batch,
                    "input_type": "document",
                    "output_dimension": dims,
                },
            )
            
            if resp.status_code != 200:
                logger.error(f"Voyage API error {resp.status_code}: {resp.text[:200]}")
                # Return zero vectors for failed batch
                all_embeddings.extend([[0.0] * dims] * len(batch))
                continue
            
            data = resp.json()
            for item in data.get("data", []):
                all_embeddings.append(item["embedding"])
            
            usage = data.get("usage", {})
            logger.info(f"Embedded batch {i//batch_size + 1}: {usage.get('total_tokens', '?')} tokens")
            
            # Be nice to the API
            if i + batch_size < len(texts):
                await asyncio.sleep(0.5)
    
    return all_embeddings


# ── Groq Virtue Scoring ─────────────────────────────────────────

VIRTUE_SCORING_PROMPT = """Score this sacred/spiritual text passage on 9 virtues.
Return ONLY a JSON array of 9 floats between 0.0 and 1.0.
Order: [unity, justice, truthfulness, love, detachment, humility, compassion, wisdom, service]

Score based on how strongly the passage teaches, exemplifies, or calls for each virtue.
0.0 = not relevant to this virtue
0.5 = moderately relevant
1.0 = the passage is centrally about this virtue

Text: {text}

JSON array:"""


async def score_virtues_batch(chunks: List[Dict], api_key: str,
                               model: str = "llama-3.1-8b-instant",
                               batch_size: int = 10) -> List[List[float]]:
    """
    Score virtue relevance for text chunks using Groq.
    llama-3.1-8b: $0.05/$0.08 per 1M tokens — cheapest option.
    """
    default_scores = [0.5] * 9
    all_scores = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            tasks = []
            
            for chunk in batch:
                prompt = VIRTUE_SCORING_PROMPT.format(
                    text=chunk["text"][:500]  # Truncate to save tokens
                )
                tasks.append(
                    _score_single(client, api_key, model, prompt, default_scores)
                )
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(f"Virtue scoring error: {r}")
                    all_scores.append(default_scores)
                else:
                    all_scores.append(r)
            
            logger.info(f"Scored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            # Rate limit respect
            if i + batch_size < len(chunks):
                await asyncio.sleep(1.0)
    
    return all_scores


async def _score_single(client: httpx.AsyncClient, api_key: str,
                        model: str, prompt: str,
                        default: List[float]) -> List[float]:
    """Score a single chunk."""
    try:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 100,
            },
        )
        
        if resp.status_code != 200:
            return default
        
        text = resp.json()["choices"][0]["message"]["content"].strip()
        # Extract JSON array from response
        match = re.search(r'\[[\d.,\s]+\]', text)
        if match:
            scores = json.loads(match.group())
            if len(scores) == 9:
                return [max(0.0, min(1.0, float(s))) for s in scores]
        
        return default
    except Exception:
        return default


# ── FalkorDB Storage ────────────────────────────────────────────

def store_terrain_batch(graph, chunks: List[Dict], 
                        embeddings: List[List[float]],
                        scores: List[List[float]],
                        source_work: str, author: str,
                        terrain_role: str) -> Tuple[int, int]:
    """Store chunks as TerrainNodes in graph_zero."""
    stored = 0
    errors = 0
    
    # Map terrain_role to layer
    layer_map = {
        "bedrock": "bedrock",
        "guidance": "earned", 
        "community": "community",
    }
    layer = layer_map.get(terrain_role, "community")
    
    # Provenance
    prov_map = {
        "bedrock": "BEDROCK",
        "guidance": "CROSS_VERIFIED",
        "community": "WITNESS",
    }
    provenance = prov_map.get(terrain_role, "WITNESS")
    
    for i, chunk in enumerate(chunks):
        try:
            node_id = chunk["chunk_id"]
            text = chunk["text"]
            embedding = embeddings[i] if i < len(embeddings) else []
            virtue_scores = scores[i] if i < len(scores) else [0.5] * 9
            
            # Escape for Cypher
            safe_text = _cypher_escape(text)
            safe_source = _cypher_escape(source_work)
            safe_author = _cypher_escape(author)
            
            vs_str = "[" + ",".join(f"{v:.4f}" for v in virtue_scores) + "]"
            mom_str = "[" + ",".join("0.0" for _ in range(9)) + "]"
            
            q = (
                f"MERGE (n:TerrainNode {{_id: '{node_id}'}}) "
                f"SET n.source_text = '{safe_text}', "
                f"n.text = '{safe_text}', "
                f"n.source_work = '{safe_source}', "
                f"n.author = '{safe_author}', "
                f"n.layer = '{layer}', "
                f"n.terrain_role = '{terrain_role}', "
                f"n.provenance_type = '{provenance}', "
                f"n.virtue_scores = {vs_str}, "
                f"n.momentum = {mom_str}, "
                f"n.kala_accumulated = 0.0, "
                f"n.traversal_count = 0, "
                f"n.authority_weight = {'0.9' if terrain_role == 'bedrock' else '0.7' if terrain_role == 'guidance' else '0.5'}, "
                f"n.chunk_index = {chunk.get('chunk_index', 0)}, "
                f"n.created_at = {int(time.time() * 1000)}"
            )
            graph.query(q)
            
            # Store embedding separately (large data)
            if embedding and len(embedding) > 0:
                emb_str = "[" + ",".join(f"{e:.6f}" for e in embedding) + "]"
                graph.query(
                    f"MATCH (n:TerrainNode {{_id: '{node_id}'}}) "
                    f"SET n.embedding = vecf32({emb_str}), "
                    f"n.embedding_dims = {len(embedding)}"
                )
            
            stored += 1
            
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.error(f"Store error: {str(e)[:100]}")
    
    return stored, errors


def _cypher_escape(text: str) -> str:
    """Escape text for Cypher string literals."""
    return (text
            .replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", " ")
            .replace("\r", "")
            .replace("\t", " "))


# ── Main Pipeline ────────────────────────────────────────────────

async def ingest_source(source_work: str, author: str, terrain_role: str,
                        url: str, graph, voyage_key: str, groq_key: str,
                        embed_model: str = "voyage-3.5",
                        embed_dims: int = 1024,
                        score_model: str = "llama-3.1-8b-instant") -> Dict:
    """Full ingest pipeline for a single source."""
    log = []
    
    # 1. Fetch
    log.append(f"Fetching: {source_work}")
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "GraphZero/0.6 (terrain-ingest; bahai-community-tool)"
            })
            if resp.status_code != 200:
                return {"error": f"HTTP {resp.status_code} fetching {url}", "log": log}
            html = resp.text
    except Exception as e:
        return {"error": f"Fetch failed: {str(e)[:100]}", "log": log}
    
    # 2. Extract text
    text = extract_text_from_html(html, url)
    if len(text) < 100:
        return {"error": f"Extracted text too short ({len(text)} chars)", "log": log}
    log.append(f"Extracted {len(text)} chars")
    
    # 3. Chunk
    chunks = chunk_text(text, source_work)
    log.append(f"Chunked into {len(chunks)} passages")
    
    if not chunks:
        return {"error": "No chunks produced", "log": log}
    
    # 4. Embed
    log.append(f"Embedding with {embed_model} ({embed_dims}d)...")
    texts = [c["text"] for c in chunks]
    embeddings = await embed_batch(texts, voyage_key, 
                                    model=embed_model, dims=embed_dims)
    log.append(f"Embedded {len(embeddings)} chunks")
    
    # 5. Virtue score
    log.append(f"Scoring virtues with {score_model}...")
    scores = await score_virtues_batch(chunks, groq_key, model=score_model)
    log.append(f"Scored {len(scores)} chunks")
    
    # 6. Store
    log.append("Storing in graph_zero...")
    stored, errors = store_terrain_batch(
        graph, chunks, embeddings, scores,
        source_work, author, terrain_role
    )
    log.append(f"Stored {stored}, errors {errors}")
    
    return {
        "source_work": source_work,
        "author": author,
        "terrain_role": terrain_role,
        "text_chars": len(text),
        "chunks": len(chunks),
        "embedded": len(embeddings),
        "scored": len(scores),
        "stored": stored,
        "errors": errors,
        "log": log,
    }


async def ingest_all(graph, voyage_key: str, groq_key: str,
                     sources: Optional[List] = None,
                     embed_model: str = "voyage-3.5",
                     embed_dims: int = 1024,
                     score_model: str = "llama-3.1-8b-instant") -> Dict:
    """Ingest all sources sequentially."""
    if sources is None:
        sources = SOURCES
    
    results = []
    total_chunks = 0
    total_stored = 0
    total_errors = 0
    
    for source_work, author, terrain_role, url in sources:
        result = await ingest_source(
            source_work, author, terrain_role, url,
            graph, voyage_key, groq_key,
            embed_model, embed_dims, score_model
        )
        results.append(result)
        total_chunks += result.get("chunks", 0)
        total_stored += result.get("stored", 0)
        total_errors += result.get("errors", 0)
        
        # Brief pause between sources
        await asyncio.sleep(1.0)
    
    return {
        "sources_processed": len(results),
        "total_chunks": total_chunks,
        "total_stored": total_stored,
        "total_errors": total_errors,
        "results": results,
    }
