import json, re, requests, logging
from core.utils import hard_filter_local
from core.db import get_variable
from config import CONFIG

logger = logging.getLogger(__name__)

LLM_URL = CONFIG["llm"]["base_url"]
LLM_HEADERS = {
    "Authorization": f"Bearer {CONFIG['llm']['api_key']}",
    "Content-Type": "application/json"
}
LLM_MODEL = CONFIG["llm"]["model"]


def _extract_json(text: str):
    """
    Ambil objek JSON pertama dari string (model kadang menambah teks lain).
    """
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ==========================================================
# 🔹 AI FILTER (PRE) — pakai prompt dari DB (variables.name=prompt_pre_filter)
# ==========================================================
def ai_pre_filter(question: str):
    # Hard filter lokal dulu
    hard = hard_filter_local(question)
    if not hard["valid"]:
        logger.info(f"[HARD FILTER] ❌ {hard['reason']}")
        return hard

    # Ambil prompt dari DB, fallback ke default kalau kosong
    prompt_db = get_variable("prompt_pre_filter_rag")
    system_prompt = prompt_db or """
Anda adalah AI filter untuk pertanyaan terkait Pemerintah Kota Medan.

Petunjuk:
1. Balas HANYA dalam format JSON berikut:
   {"valid": true/false, "reason": "<penjelasan>", "clean_question": "<pertanyaan yang sudah dibersihkan>"}

2. Mark valid jika dan hanya jika pertanyaan membahas:
   - Dinas/instansi di bawah Pemko Medan
   - Layanan publik di Medan (KTP, SIM, pajak daerah, fasilitas kesehatan, pendidikan, dll)
   - Izin usaha/lingkungan/keramaian yang dikeluarkan Pemko Medan
   - Fasilitas umum milik Pemko Medan (taman, jalan, RSUD, dll)
   - Kebijakan atau program Pemerintah Kota Medan

3. Mark tidak valid jika:
   - Membahas daerah di luar Medan
   - Membahas figur publik non-pemerintah (selebriti, influencer, dll)
   - Membahas topik pribadi, gosip, atau tidak relevan
   - Pertanyaan tidak jelas/terlalu pendek

4. Bersihkan ejaan & tanda baca, jangan ubah maksud pertanyaan.
JANGAN BERIKAN PENJELASAN DI LUAR JSON.
"""

    try:
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": question.strip()}
            ],
            "temperature": 0.0,
            "top_p": 0.6
        }
        resp = requests.post(LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"])
        content = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _extract_json(content) or {
            "valid": True,
            "reason": "AI tidak mengembalikan JSON",
            "clean_question": question
        }
        logger.info(f"[AI FILTER] ✅ Valid: {parsed.get('valid')} | Reason: {parsed.get('reason')}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-Filter] {e}")
        return {"valid": True, "reason": "Fallback error AI Filter", "clean_question": question}


# ==========================================================
# 🔹 AI RELEVANCE CHECK (POST) — pakai prompt dari DB (variables.name=prompt_relevance)
# ==========================================================
def ai_check_relevance(user_q: str, rag_q: str):
    prompt_db = get_variable("prompt_relevance_rag")  # optional
    system_prompt = prompt_db or """
Tugas Anda mengevaluasi apakah hasil pencarian RAG sesuai dengan maksud
pertanyaan pengguna.
Balas hanya JSON:
{"relevant": true/false, "reason": "...", "reformulated_question": "..."}

Kriteria:
✅ Relevan jika topik sama (layanan publik, fasilitas, dokumen, kebijakan).
❌ Tidak relevan jika membahas jabatan/instansi berbeda,
   kota lain, atau konteks umum vs spesifik.
Jika tidak relevan, ubah pertanyaan jadi versi singkat berbentuk tanya
maks. 12 kata.
"""

    try:
        user_prompt = f"User: {user_q}\nRAG Result: {rag_q}"
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()}
            ],
            "temperature": 0.1,
            "top_p": 0.5
        }
        resp = requests.post(LLM_URL, headers=LLM_HEADERS, json=payload, timeout=CONFIG["llm"]["timeout_sec"])
        content = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _extract_json(content) or {"relevant": True, "reason": "-", "reformulated_question": ""}

        reform = parsed.get("reformulated_question", "").strip()
        if len(reform.split()) > 12:
            parsed["reformulated_question"] = " ".join(reform.split()[:12]) + "..."

        logger.info(f"[AI RELEVANCE] ✅ Relevant: {parsed.get('relevant')} | Reason: {parsed.get('reason')}")
        return parsed

    except Exception as e:
        logger.error(f"[AI-Post] {e}")
        return {"relevant": True, "reason": "AI relevance check failed", "reformulated_question": ""}
