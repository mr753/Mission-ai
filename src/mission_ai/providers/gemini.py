import os
import json
import time
import random
from PIL import Image
from mission_ai.models import ImageAnalysis, MissionContext
from mission_ai.providers.base import AIProvider

class GeminiProvider(AIProvider):
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError:
                raise ImportError("Gemini provider requires google-genai, which is not installed.")
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _is_transient_error(self, e: Exception) -> bool:
        err_str = str(e).lower()
        non_transient = [
            "401", "403", "unauthenticated", "permission denied",
            "invalid api key", "invalid_argument", "not_found", "400"
        ]
        if any(nt in err_str for nt in non_transient):
            return False

        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if code in {401, 403, 400}:
            return False
        if code in {503, 429, 500, 502, 504}:
            return True

        transient_keywords = [
            "503", "unavailable", "resource_exhausted", "rate limit",
            "429", "timed out", "timeout", "deadline exceeded",
            "internal", "server error", "high demand", "temporarily unavailable"
        ]
        return any(kw in err_str for kw in transient_keywords)

    def _is_quota_error(self, e: Exception) -> bool:
        """Identify exhausted API quota so the next model can be tried immediately."""
        err_str = str(e).lower()
        quota_markers = (
            "quota exceeded",
            "generate_content_free_tier_requests",
            "generaterequestsperdayperprojectpermodelfreetier",
            "exceeded your current quota",
        )
        return any(marker in err_str for marker in quota_markers)
    def _call_with_retry(self, func, *args, **kwargs):
        """Call Gemini with quota-aware model failover and bounded retries."""
        primary_model = kwargs.get("model", self.model_name)
        configured = os.getenv(
            "GEMINI_FALLBACK_MODELS",
            "gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.5-flash,gemini-2.5-flash-lite",
        )
        fallback_models = [m.strip() for m in configured.split(",") if m.strip()]
        models = []
        for model in [primary_model, *fallback_models]:
            if model and model not in models:
                models.append(model)

        max_attempts = max(1, int(os.getenv("GEMINI_RETRY_ATTEMPTS", "2")))
        base_delay = float(os.getenv("GEMINI_RETRY_BASE_DELAY", "2"))
        last_error = None
        for model in models:
            for attempt in range(max_attempts):
                try:
                    call_kwargs = dict(kwargs)
                    call_kwargs["model"] = model
                    response = func(*args, **call_kwargs)
                    if model != primary_model:
                        print(f"Gemini failover: using {model}")
                    return response
                except Exception as e:
                    last_error = e
                    if self._is_quota_error(e):
                        break
                    if not self._is_transient_error(e):
                        raise
                    if attempt == max_attempts - 1:
                        break
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
                    time.sleep(delay)
        raise RuntimeError(
            "Gemini request failed across models: "
            + ", ".join(models)
            + f". Last error: {last_error}"
        ) from last_error
    def analyze_image(self, image_path: str, mission_context: MissionContext) -> ImageAnalysis:
        client = self._get_client()
        image = Image.open(image_path)
        prompt = (f"Analyze this image based on the mission: {mission_context.instructions}. "
                  "Describe only visible facts. Do not invent people, locations, organizations, numbers, events, or claims. "
                  "Return valid JSON with keys: summary, visible_subjects, visual_context, relevant_details.")
        
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=[prompt, image]
        )
        text = response.text if response and hasattr(response, "text") else ""
        if not text or not text.strip():
            raise ValueError("Gemini response is empty or contains no text.")

        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        if not cleaned:
            raise ValueError("Gemini response is empty after removing markdown fences.")
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # Gemini can occasionally add prose around the JSON-only response.
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end <= start:
                raise exc
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                raise
        # Gemini may return extra fields. Keep only the ImageAnalysis schema.
        allowed = {"summary", "visible_subjects", "visual_context", "relevant_details", "confidence"}
        data = {key: value for key, value in data.items() if key in allowed}
        required = {"summary", "visible_subjects", "visual_context", "relevant_details"}
        missing = required - data.keys()
        if missing:
            raise ValueError(
                f"Gemini image analysis missing required fields: {sorted(missing)}"
            )
        return ImageAnalysis(**data)

    def select_best_image(self, analyses, mission_context: MissionContext) -> int:
        """Select the best candidate index for the mission."""
        client = self._get_client()
        candidates = []
        for index, (path, analysis) in enumerate(analyses):
            candidates.append({"index": index, "file": os.path.basename(str(path)), "summary": analysis.summary, "visible_subjects": analysis.visible_subjects, "visual_context": analysis.visual_context, "relevant_details": analysis.relevant_details})
        prompt = (
            "Pilih SATU gambar terbaik untuk mission berdasarkan hanya fakta visual. "
            "Nilai relevansi terhadap tema, kejelasan subjek, dan kecocokan konten. "
            "Kembalikan JSON valid dengan key selected_index (integer).\n"
            f"Mission: {mission_context.main_message}\n"
            f"Instruksi: {mission_context.instructions}\n"
            f"Kandidat: {json.dumps(candidates, ensure_ascii=False)}"
        )
        response = self._call_with_retry(client.models.generate_content, model=self.model_name, contents=prompt)
        text = (response.text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(
                    f"Gemini image selection did not return valid JSON: {text[:300]!r}"
                ) from exc
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                raise ValueError(
                    f"Gemini image selection returned malformed JSON: {text[:300]!r}"
                ) from exc
        index = int(data["selected_index"])
        if index < 0 or index >= len(analyses): raise ValueError(f"Gemini selected invalid image index: {index}")
        return index

    def generate_caption(self, image_analysis: ImageAnalysis, mission_context: MissionContext, platform: str) -> str:
        """Generate an engaging caption without adding facts absent from the image analysis."""
        import re

        client = self._get_client()
        summary = str(image_analysis.summary or "").strip()
        visible_subjects = [str(item).strip() for item in (image_analysis.visible_subjects or []) if str(item).strip()]
        relevant_details = [str(item).strip() for item in (image_analysis.relevant_details or []) if str(item).strip()]

        # Prefer an actual headline read from the image, not the model's visual description.
        source_title = ""
        for subject in visible_subjects:
            if not re.search(r"text overlay|teks|judul|headline|tertulis", subject, re.IGNORECASE):
                continue
            match = re.search(r"""['"“]([^'"”]{12,240})['"”]""", subject)
            if match:
                source_title = match.group(1).strip()
                break
        if not source_title:
            match = re.search(r"""['"“]([^'"”]{12,240})['"”]""", summary)
            source_title = match.group(1).strip() if match else summary
        if not source_title:
            source_title = str(mission_context.main_message or "").strip()

        facts = {
            "judul/topik": source_title,
            "ringkasan visual": summary,
            "subjek terlihat": visible_subjects,
            "detail terlihat": relevant_details,
        }
        prompt = (
            f"Buat caption {platform} dalam bahasa Indonesia yang menarik, singkat, natural, dan positif. "
            "Gunakan hanya fakta yang benar-benar didukung data di bawah. Judul boleh dijadikan topik, "
            "tetapi jangan menganggap klaim dalam judul sebagai bukti hasil atau dampak yang sudah terjadi. "
            "Jangan mengarang tanggal/waktu, lokasi, nama atau keahlian pembicara, kutipan, angka, penyebab, "
            "keberhasilan program, manfaat, dampak, transparansi, efisiensi, atau ajakan yang mengandaikan "
            "fakta yang tidak tersedia. Jangan menyebut 'kemarin' atau menyebut seseorang sebagai pakar "
            "tanpa bukti. Jangan mendeskripsikan poster secara berlebihan. Boleh gunakan satu pertanyaan "
            "atau ajakan netral, serta emoji/hashtag relevan. Keluarkan caption saja.\n\n"
            f"DATA SUMBER:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
            f"ARAH MISI:\n{mission_context.main_message}"
        )

        invalid_markers = (
            "kemarin", "para pakar", "pakar inspiratif", "sudah terbukti",
            "terbukti meningkatkan", "memastikan", "menjamin", "makin transparan",
            "lebih transparan", "rantai pasok yang efisien", "pasokan yang efisien",
            "dampak positif yang berkelanjutan", "masa depan bangsa yang lebih sehat",
            "masa depan yang lebih sehat", "berhasil meningkatkan",
        )
        for attempt in range(2):
            retry_prompt = prompt
            if attempt:
                retry_prompt += (
                    "\n\nTulis ulang. Hapus setiap klaim yang tidak tertulis secara eksplisit "
                    "dalam data sumber; jangan mengarang waktu, kualitas, hasil, atau dampak."
                )
            response = self._call_with_retry(
                client.models.generate_content,
                model=self.model_name,
                contents=retry_prompt,
            )
            candidate = (response.text or "").strip()
            # Preserve the provider contract: an empty model response remains empty.
            # The downstream pipeline can decide how to handle missing caption text.
            if not candidate:
                return ""
            if not any(marker in candidate.lower() for marker in invalid_markers):
                return candidate

        # Safe fallback: make no claim beyond the identified topic.
        topic = source_title.rstrip(".!?")
        return f"{topic}. Kenali lebih dekat topik ini."

    def _search_web_context(self, topic: str, max_results: int = 4) -> str:
        """Fetch Google News RSS results as optional research context."""
        from html import unescape
        from html.parser import HTMLParser
        from urllib.parse import quote_plus
        from urllib.request import Request, urlopen
        import xml.etree.ElementTree as ET

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []

            def handle_data(self, data):
                self.parts.append(data)

        try:
            words = topic.split()
            queries = [topic[:240]]
            if len(words) > 6:
                queries.append(" ".join(words[:6]))
            if len(words) > 3:
                queries.append(" ".join(words[:3]))
            # Keep order while removing duplicate queries.
            queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

            for search_topic in queries:
                query = quote_plus(search_topic)
                url = (
                    "https://news.google.com/rss/search?q="
                    f"{query}&hl=id&gl=ID&ceid=ID:id"
                )
                request = Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; Mission-AI/1.0)"},
                )
                with urlopen(request, timeout=12) as response:
                    xml_data = response.read(1_000_000)
                root = ET.fromstring(xml_data)
                lines = []
                for item in root.findall(".//item"):
                    title = (item.findtext("title") or "").strip()
                    if not title:
                        continue
                    source_node = item.find("source")
                    source = (
                        (source_node.text or "").strip()
                        if source_node is not None else ""
                    )
                    published = (item.findtext("pubDate") or "").strip()
                    link = (item.findtext("link") or "").strip()
                    description = unescape(item.findtext("description") or "")
                    extractor = TextExtractor()
                    extractor.feed(description)
                    snippet = " ".join(" ".join(extractor.parts).split())

                    details = [title]
                    if source:
                        details.append(f"Sumber: {source}")
                    if published:
                        details.append(f"Tanggal: {published}")
                    if snippet:
                        details.append(f"Cuplikan: {snippet[:500]}")
                    if link:
                        details.append(f"Tautan: {link}")
                    lines.append("- " + ". ".join(details))
                    if len(lines) >= max_results:
                        break

                if lines:
                    return "\n".join(lines)

            print(
                "Peringatan: Google News RSS tidak mengembalikan hasil yang dapat digunakan. "
                f"Topik pencarian: {topic[:120]!r}"
            )
            return ""
        except Exception as exc:
            # Research is optional; never disable TLS verification or break the mission pipeline.
            print(
                f"Peringatan: riset web tidak tersedia ({type(exc).__name__}). "
                "Voice-over memakai konteks gambar saja."
            )
            return ""

    def generate_voiceover_script(self, image_analysis: ImageAnalysis, mission_context: MissionContext) -> str:
        """Create natural Indonesian narration using the image topic plus optional web research."""
        import re

        client = self._get_client()
        summary = str(image_analysis.summary or "").strip()
        # Prefer text actually read from the image over an AI-generated visual description.
        source_title = ""
        for subject in (image_analysis.visible_subjects or []):
            subject_text = str(subject)
            if not re.search(r"text overlay|teks|judul|headline|tertulis", subject_text, re.IGNORECASE):
                continue
            title_match = re.search(r"""['"“]([^'"”]{12,240})['"”]""", subject_text)
            if title_match:
                source_title = title_match.group(1).strip()
                break
        if not source_title:
            title_match = re.search(r'["“]([^"”]{12,240})["”]', summary)
            source_title = title_match.group(1).strip() if title_match else summary
        if not source_title:
            source_title = str(mission_context.main_message or "").strip()
        if not source_title:
            raise ValueError("No source title or mission topic is available for voice-over.")

        research = self._search_web_context(source_title)
        research_section = research if research else (
            "Tidak ada hasil riset web yang berhasil diambil. Jangan berpura-pura telah melakukan riset "
            "dan jangan menambahkan fakta di luar judul sumber."
        )
        prompt = (
            "Buat naskah voice-over media sosial dalam bahasa Indonesia yang natural, jelas, dan enak didengar. "
            "Tulis 2–4 kalimat pendek bila sumber cukup; jika bukti terbatas, cukup 1–2 kalimat. "
            "Gunakan judul untuk mengenali topik dan hasil pencarian sebagai konteks faktual. "
            "Utamakan fakta yang didukung cuplikan sumber; jangan mengubah kemungkinan menjadi kepastian. "
            "Jangan menyatakan sebab-akibat, hasil, manfaat, angka, tanggal, atau peran yang tidak didukung sumber. "
            "Jangan mendeskripsikan tampilan gambar/poster atau memakai frasa seperti 'gambar ini menunjukkan'. "
            "Jangan mengarang sumber atau menyebut telah memverifikasi hal yang tidak tersedia. "
            "Jika sumber saling bertentangan atau cuplikan tidak cukup, tetap pada fakta judul dan nyatakan secara netral. "
            "Hindari pembuka klise, gaya artikel kaku, dan bahasa Inggris. Keluarkan hanya naskah voice-over.\n\n"
            f"JUDUL/TOPIK DARI GAMBAR:\n{source_title}\n\n"
            f"HASIL PENCARIAN WEB (cuplikan, bukan bukti lengkap):\n{research_section}\n\n"
            f"ARAH MISI:\n{mission_context.main_message}"
        )

        for attempt in range(2):
            response = self._call_with_retry(
                client.models.generate_content,
                model=self.model_name,
                contents=prompt if attempt == 0 else (
                    prompt + "\n\nPerbaiki naskah sebelumnya: hanya gunakan klaim yang didukung judul atau "
                    "cuplikan sumber, hapus klaim yang terlalu pasti, dan gunakan bahasa Indonesia lisan."
                ),
            )
            candidate = (response.text or "").strip()
            candidate = re.split(
                r"\n\s*(?:\*{1,2}|#{1,6})?\s*(?:analisis fakta|fakta vs\.? interpretasi|interpretasi yang dihindari|catatan:|analisis:)\s*",
                candidate, maxsplit=1, flags=re.IGNORECASE,
            )[0].strip()
            candidate = re.sub(
                r"^\s*(?:\*{1,2}|#{1,6})?\s*(?:naskah voice[- ]?over|voice[- ]?over)\s*(?:\*{1,2}|#{1,6})?\s*:?\s*",
                "", candidate, flags=re.IGNORECASE,
            )
            candidate = re.sub(r"^\s*[-*]\s+", "", candidate, flags=re.MULTILINE).strip()
            lower = candidate.lower()
            invalid_markers = (
                "an informational poster", "this image", "the image", "the picture",
                "the photograph", "numbered points", "visual layout", "a photograph of",
                "this poster", "the poster displays", "the poster shows",
                "gambar ini", "foto ini", "poster ini", "infografik ini",
                "poster informatif", "menyajikan teks dan infografik",
                "terdapat pula acara", "menampilkan sejumlah pembicara",
                "audiens yang menghadap", "terlihat pada gambar",
                "di bagian bawah gambar", "di atas panggung serta audiens",
            )
            unsupported_certainty = (
                "memastikan", "pasti akan", "terbukti meningkatkan", "menjamin",
                "secara otomatis membuat", "sudah berhasil", "dipastikan akan",
            )
            english_markers = ("the headline", "with three", "at the bottom", "this informational", "displaying the headline")
            if (
                candidate
                and not any(marker in lower for marker in invalid_markers)
                and not any(marker in lower for marker in english_markers)
                and not any(marker in lower for marker in unsupported_certainty)
            ):
                return candidate

        # Deterministic, topic-grounded fallback keeps the pipeline working if generation is unusable.
        return f"Topik yang dibahas adalah {source_title.rstrip('.!?')}."

