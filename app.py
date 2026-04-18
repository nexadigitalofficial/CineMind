import os
import json
from functools import wraps
from flask import Flask, request, jsonify, render_template
from google import genai
from duckduckgo_search import DDGS
import firebase_admin
from firebase_admin import credentials, auth, firestore

# ==============================================
# Firebase Admin + Firestore
# ==============================================
firebase_key_json = os.environ.get("FIREBASE_KEY")
if not firebase_key_json:
    raise ValueError("FIREBASE_KEY environment variable eksik!")

cred = credentials.Certificate(json.loads(firebase_key_json))
firebase_admin.initialize_app(cred)
db = firestore.client()

# ==============================================
# Gemini Konfigürasyonu (YENİ SDK: google-genai)
# pip install google-genai
# ==============================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable eksik!")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"   # 2.0 serisi Haziran 2026'da kapatılıyor


def llm_generate(prompt: str) -> str:
    """Gemini ile metin üret."""
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip()


# ==============================================
# Flask
# ==============================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())


# ==============================================
# Auth Decorator
# ==============================================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization", "")
        if not token:
            return jsonify({"error": "Token eksik"}), 401
        try:
            if token.startswith("Bearer "):
                token = token[7:]
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({"error": f"Geçersiz token: {str(e)}"}), 401
        return f(*args, **kwargs)
    return decorated


# ==============================================
# CineMind Engine (Firestore tabanlı)
# ==============================================
class CineMindEngine:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.col = (
            db.collection("users")
            .document(user_id)
            .collection("favorites")
        )

    def _search_deep_context(self, title: str) -> str:
        queries = [
            f"{title} film analysis cinematography themes",
            f"{title} philosophical meaning explained",
            f"{title} visual style color palette atmosphere",
        ]
        parts = []
        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, max_results=2):
                        parts.append(r["body"])
                except Exception:
                    continue
        full = " ".join(parts)
        return full[:3000] if full else f"{title} is a cinematic work."

    def _extract_cinematic_dna(self, title: str, context: str) -> str:
        prompt = f"""
[SİNEMATİK DNA ANALİZİ]
Eser: {title}
İnternetten Toplanan Analiz Verileri: {context}

Görevin: Bu eseri izleyen bir insanın bilinçaltında neleri tetiklendiğini çözümle.
Aşağıdaki başlıkları kullanarak TEK BİR PARAGRAF yaz:
- ATMOSFER VE RENK PALETİ: Görsel doku nasıl? (Noir, Pastel Distopya, Karanlık Akademi vb.)
- PSİKOLOJİK TEMEL: Hangi varoluşsal korkulara/arzuslara dokunuyor?
- ANLATI RİTMİ: Hızlı tüketim mi yoksa yavaş yanan bir gerilim mi?
- AHLAKİ DÜZLEM: Karakterler gri alanda mı yoksa net iyi/kötü mü?

Cevabını sadece analiz metni olarak ver. Giriş cümlesi kullanma.
        """.strip()
        try:
            return llm_generate(prompt)
        except Exception as e:
            print(f"Gemini Hatası: {e}")
            return "Görsel olarak çarpıcı, psikolojik derinliği olan bir eser."

    def add_to_memory(self, title: str) -> str:
        print(f"[{self.user_id}] '{title}' araştırılıyor...")
        context = self._search_deep_context(title)
        print(f"[{self.user_id}] DNA çıkarılıyor...")
        dna = self._extract_cinematic_dna(title, context)

        doc_id = (
            title.lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("/", "")
        )
        self.col.document(doc_id).set({"title": title, "dna": dna})
        print(f"[{self.user_id}] '{title}' Firestore'a kaydedildi.")
        return dna

    def get_library(self):
        return [
            {"title": doc.get("title"), "dna": doc.get("dna")}
            for doc in self.col.stream()
        ]

    def delete_from_library(self, title: str) -> bool:
        doc_id = (
            title.lower()
            .replace(" ", "_")
            .replace(":", "")
            .replace("/", "")
        )
        ref = self.col.document(doc_id)
        if ref.get().exists:
            ref.delete()
            return True
        return False

    def generate_recommendations(self) -> str:
        library = self.get_library()
        if not library:
            return "Henüz favori eser eklemediniz."

        favorites_text = "\n".join(
            f"- {item['title']}: {item['dna']}" for item in library
        )
        prompt = f"""
[KULLANICI ZEVK PROFİLİ ÇIKARIMI]
Kullanıcının favori eserleri ve analizleri:
{favorites_text}

Görevler:
1. Kullanıcının sinematik zevkini TEK PARAGRAFTA özetle.
2. Bu profile uyan, listede OLMAYAN 3 eser öner.
3. Her öneri için "Neden Seveceksin?" açıklaması yaz.

Format:
[PROFİL ANALİZİ]
(paragraf)

[ÖNERİLER]
1. **Eser Adı (Yıl)** - Tür
   *Neden Seveceksin?*: ...
2. ...
3. ...
        """.strip()
        return llm_generate(prompt)


# ==============================================
# Routes
# ==============================================
@app.route("/")
def index():
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/health")
def health():
    """Render health check — uyku modunu önler."""
    return jsonify({"status": "ok"})

@app.route("/api/library", methods=["GET"])
@token_required
def api_library():
    engine = CineMindEngine(request.user["uid"])
    return jsonify(engine.get_library())

@app.route("/api/add", methods=["POST"])
@token_required
def api_add():
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Eser adı gerekli"}), 400
    engine = CineMindEngine(request.user["uid"])
    try:
        dna = engine.add_to_memory(title)
        return jsonify({"success": True, "title": title, "dna": dna})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/delete", methods=["DELETE"])
@token_required
def api_delete():
    data = request.json or {}
    title = data.get("title", "").strip()
    if not title:
        return jsonify({"error": "Eser adı gerekli"}), 400
    engine = CineMindEngine(request.user["uid"])
    deleted = engine.delete_from_library(title)
    return jsonify({"success": deleted})

@app.route("/api/recommend", methods=["GET"])
@token_required
def api_recommend():
    engine = CineMindEngine(request.user["uid"])
    try:
        result = engine.generate_recommendations()
        return jsonify({"success": True, "recommendations": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
