import os
import json
from functools import wraps
from flask import Flask, request, jsonify, render_template, session
import chromadb
import ollama
from duckduckgo_search import DDGS
import firebase_admin
from firebase_admin import credentials, auth

# ==============================================
# Firebase Admin Başlat
# ==============================================
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)

# ==============================================
# Flask Uygulaması
# ==============================================
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Session için

# ==============================================
# ChromaDB Persistent Client
# ==============================================
chroma_client = chromadb.PersistentClient(path="./cinemind_memory")

# ==============================================
# Kimlik Doğrulama Decorator'ı
# ==============================================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"error": "Token eksik"}), 401
        try:
            # Bearer token formatı: "Bearer <token>"
            if token.startswith('Bearer '):
                token = token.split(' ')[1]
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
        except Exception as e:
            return jsonify({"error": f"Geçersiz token: {str(e)}"}), 401
        return f(*args, **kwargs)
    return decorated

# ==============================================
# CineMind Motor Sınıfı (Kullanıcı bazlı)
# ==============================================
class CineMindEngine:
    def __init__(self, user_id):
        self.user_id = user_id
        self.model = "qwen2.5:7b"
        # Her kullanıcı için ayrı koleksiyon: user_{uid}_favorites
        collection_name = f"user_{user_id}_favorites"
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _search_deep_context(self, title: str) -> str:
        queries = [
            f"{title} film analysis cinematography themes",
            f"{title} philosophical meaning explained",
            f"{title} visual style color palette atmosphere"
        ]
        context_parts = []
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = ddgs.text(q, max_results=2)
                    for r in results:
                        context_parts.append(r['body'])
                except Exception as e:
                    print(f"Arama hatası ({q}): {e}")
                    continue
        full_context = " ".join(context_parts)
        if not full_context:
            full_context = f"{title} is a cinematic work."
        return full_context[:3000]

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
        """
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            print(f"LLM Hatası: {e}")
            return f"Görsel olarak çarpıcı, psikolojik derinliği olan bir eser."

    def add_to_memory(self, title: str):
        print(f"[{self.user_id}] '{title}' araştırılıyor...")
        context = self._search_deep_context(title)
        print(f"[{self.user_id}] Qwen2.5 DNA çıkarıyor...")
        dna_text = self._extract_cinematic_dna(title, context)

        embedding = ollama.embeddings(model=self.model, prompt=dna_text)["embedding"]

        doc_id = title.lower().replace(" ", "_").replace(":", "")
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[dna_text],
            metadatas=[{"title": title, "dna": dna_text}]
        )
        print(f"[{self.user_id}] '{title}' hafızaya kaydedildi.")
        return dna_text

    def get_library(self):
        items = self.collection.get()
        library = []
        if items['ids']:
            for meta, doc in zip(items['metadatas'], items['documents']):
                library.append({"title": meta['title'], "dna": doc})
        return library

    def generate_recommendations(self):
        all_items = self.collection.get()
        if not all_items['ids']:
            return "Henüz favori eser eklemediniz."

        favorites_text = "\n".join([
            f"- {m['title']}: {d}"
            for m, d in zip(all_items['metadatas'], all_items['documents'])
        ])

        synthesis_prompt = f"""
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
        """
        response = ollama.generate(model=self.model, prompt=synthesis_prompt)
        return response['response']

# ==============================================
# Rotalar
# ==============================================
@app.route('/')
def index():
    # Ana sayfa login sayfasına yönlendirir
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    # Giriş yapmış kullanıcılar için dashboard
    return render_template('dashboard.html')

@app.route('/api/library', methods=['GET'])
@token_required
def api_library():
    user_id = request.user['uid']
    engine = CineMindEngine(user_id)
    library = engine.get_library()
    return jsonify(library)

@app.route('/api/add', methods=['POST'])
@token_required
def api_add():
    user_id = request.user['uid']
    data = request.json
    title = data.get('title', '').strip()
    if not title:
        return jsonify({"error": "Eser adı gerekli"}), 400

    engine = CineMindEngine(user_id)
    try:
        dna = engine.add_to_memory(title)
        return jsonify({"success": True, "title": title, "dna": dna})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
@token_required
def api_recommend():
    user_id = request.user['uid']
    engine = CineMindEngine(user_id)
    try:
        result = engine.generate_recommendations()
        return jsonify({"success": True, "recommendations": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)