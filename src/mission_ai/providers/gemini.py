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
        client = self._get_client()
        prompt = (f"Generate a {platform} caption grounded in this analysis: {image_analysis.summary}. "
                  f"Mission context: {mission_context.main_message}. Return text only.")
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        return (response.text or "").strip()

    def generate_voiceover_script(self, image_analysis: ImageAnalysis, mission_context: MissionContext) -> str:
        client = self._get_client()
        prompt = (
            "Buat script voice-over sosial media dalam Bahasa Indonesia yang terdengar seperti "
            "seseorang benar-benar sedang berbicara kepada penonton.\n\n"
            f"Pesan utama mission: {mission_context.main_message}\n"
            f"Instruksi mission: {mission_context.instructions}\n"
            f"Analisis visual gambar: {image_analysis.summary}\n"
            f"Detail visual: {', '.join(image_analysis.relevant_details)}\n\n"
            "ATURAN UTAMA:\n"
            "1. Tulis untuk DIDENGARKAN, bukan untuk dibaca seperti artikel atau laporan.\n"
            "2. Gunakan bahasa percakapan Indonesia yang sederhana, luwes, dan natural.\n"
            "3. Hindari gaya berita, bahasa birokrasi, istilah laporan, dan kalimat yang terlalu resmi.\n"
            "4. Utamakan kalimat pendek dan sedang. Campurkan panjang kalimat agar ritmenya hidup.\n"
            "5. Gunakan tanda baca untuk membantu intonasi: koma untuk jeda pendek, titik untuk jeda lebih jelas. "
            "Boleh gunakan tanda tanya atau tanda seru jika memang sesuai konteks.\n"
            "6. Buat pembuka yang langsung menarik perhatian tanpa clickbait atau klaim yang tidak didukung.\n"
            "7. Setelah pembuka, jelaskan inti informasi dengan bahasa yang mudah dipahami.\n"
            "8. Tutup dengan satu kalimat yang terasa alami dan relevan, bukan slogan generik.\n"
            "9. Jangan mengulang ide yang sama hanya untuk memperpanjang durasi.\n"
            "10. Jangan membuat daftar poin dalam bentuk kalimat.\n"
            "11. Jangan mengarang fakta, angka, lokasi, nama, kejadian, atau klaim yang tidak didukung sumber.\n"
            "12. Jangan menyalin instruksi mission menjadi kalimat voice-over.\n"
            "13. Jangan gunakan markdown, hashtag, emoji, judul, atau catatan produksi.\n"
            "14. Panjang sekitar 55-80 kata. Tidak perlu mengejar durasi tertentu; kualitas dan kelancaran bicara lebih penting.\n\n"
            "Sebelum menulis, bayangkan kamu sedang menjelaskan informasi ini kepada penonton "
            "dengan suara santai dan percaya diri. Gunakan hanya informasi yang benar-benar didukung "
            "oleh mission dan analisis gambar.\n\n"
            "Keluarkan hanya script voice-over final."
        )
        response = self._call_with_retry(
            client.models.generate_content,
            model=self.model_name,
            contents=prompt
        )
        return (response.text or "").strip()
