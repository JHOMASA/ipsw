import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from app.database import InventoryDB
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    """Load the embedding model with Streamlit caching"""
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        logger.info("Model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

class SemanticSearch:
    def __init__(self, db_path: str = "data/inventory.db"):
        """Initialize semantic search system"""
        try:
            self.tokenizer, self.model = load_model()
            self.db = sqlite3.connect(db_path)
            self.db.row_factory = sqlite3.Row  # Enable dict-style access
            self._init_db()
            logger.info("SemanticSearch initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    def _init_db(self):
        """Initialize database tables"""
        cursor = self.db.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS producto_embeddings (
            producto_id INTEGER PRIMARY KEY,
            nombre_embedding BLOB,
            descripcion_embedding BLOB,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        )
        """)
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_producto_id 
        ON producto_embeddings(producto_id)
        """)
        self.db.commit()

    def encode(self, text: str) -> np.ndarray:
        """Generate embeddings for text"""
        try:
            inputs = self.tokenizer(text, 
                                 return_tensors='pt', 
                                 padding=True, 
                                 truncation=True,
                                 max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.error(f"Encoding failed for text: {str(e)}")
            return np.zeros(self.model.config.hidden_size)

    @lru_cache(maxsize=1000)
    def _get_product_text(self, producto_id: int) -> Optional[Dict]:
        """Cache product text queries"""
        try:
            cursor = self.db.cursor()
            cursor.execute(
                "SELECT nombre, categoria, notas FROM productos WHERE id = ?", 
                (producto_id,)
            )
            if row := cursor.fetchone():
                return dict(row)
            return None
        except sqlite3.Error as e:
            logger.warning(f"Database query failed: {str(e)}")
            return None

    def generar_embeddings_producto(self, producto_id: int):
        """Generate and store embeddings for a product"""
        product_data = self._get_product_text(producto_id)
        if not product_data:
            raise ValueError(f"Product {producto_id} not found")
        
        try:
            nombre = product_data['nombre']
            descripcion = f"{nombre} {product_data['categoria'] or ''} {product_data['notas'] or ''}".strip()
            
            with ThreadPoolExecutor() as executor:
                nombre_embedding, desc_embedding = list(executor.map(
                    self.encode,
                    [nombre, descripcion]
                ))
            
            with self.db:  # Transaction
                self.db.execute("""
                INSERT OR REPLACE INTO producto_embeddings 
                (producto_id, nombre_embedding, descripcion_embedding)
                VALUES (?, ?, ?)
                """, (
                    producto_id, 
                    np.array(nombre_embedding).tobytes(), 
                    np.array(desc_embedding).tobytes()
                ))
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def buscar_semanticamente(self, consulta: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Perform semantic search"""
        try:
            consulta_embedding = self.encode(consulta)
            cursor = self.db.cursor()
            
            cursor.execute("""
            SELECT p.id, p.nombre, p.codigo, p.categoria,
                   e.nombre_embedding, e.descripcion_embedding
            FROM productos p
            LEFT JOIN producto_embeddings e ON p.id = e.producto_id
            WHERE p.activo = TRUE
            ORDER BY e.last_updated DESC
            """)
            
            results = []
            needs_embedding = []
            
            for row in cursor.fetchall():
                row = dict(row)
                if not row['nombre_embedding']:
                    needs_embedding.append(row['id'])
                    continue
                    
                similarities = self._calculate_similarities(
                    consulta_embedding,
                    np.frombuffer(row['nombre_embedding'], dtype=np.float32),
                    np.frombuffer(row['descripcion_embedding'], dtype=np.float32)
                )
                
                if similarities['total'] >= threshold:
                    results.append({
                        'id': row['id'],
                        'nombre': row['nombre'],
                        'codigo': row['codigo'],
                        'categoria': row['categoria'],
                        **similarities
                    })
            
            if needs_embedding:
                logger.info(f"Generating embeddings for {len(needs_embedding)} products")
                with ThreadPoolExecutor() as executor:
                    executor.map(self.generar_embeddings_producto, needs_embedding)
                return self.buscar_semanticamente(consulta, top_k, threshold)
            
            return sorted(results, key=lambda x: x['total'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _calculate_similarities(self, 
                              query_embedding: np.ndarray, 
                              name_embedding: np.ndarray, 
                              desc_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate weighted similarity scores"""
        def safe_sim(a, b):
            sim = self._cosine_similarity(a, b)
            return sim if not np.isnan(sim) else 0.0
        
        return {
            'name_sim': safe_sim(query_embedding, name_embedding),
            'desc_sim': safe_sim(query_embedding, desc_embedding),
            'total': 0.6 * safe_sim(query_embedding, name_embedding) + 
                    0.4 * safe_sim(query_embedding, desc_embedding)
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b) / (a_norm * b_norm) if a_norm * b_norm > 0 else 0.0

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'db'):
            self.db.close()
        logger.info("Resources cleaned up")

    def __del__(self):
        """Destructor"""
        self.close()

# Streamlit UI
def main():
    st.set_page_config(page_title="Semantic Product Search", layout="wide")
    
    st.title("🔍 Semantic Product Search")
    st.markdown("Search products using natural language understanding")
    
    # Initialize search system
    if 'search_system' not in st.session_state:
        with st.spinner("Loading search system..."):
            try:
                st.session_state.search_system = SemanticSearch()
            except Exception as e:
                st.error(f"Failed to initialize search system: {str(e)}")
                st.stop()
    
    # Search interface
    with st.form("search_form"):
        query = st.text_input("Search query", placeholder="Enter product name or description")
        col1, col2 = st.columns(2)
        top_k = col1.slider("Number of results", 1, 20, 5)
        threshold = col2.slider("Similarity threshold", 0.0, 1.0, 0.3, 0.05)
        submitted = st.form_submit_button("Search")
    
    if submitted and query:
        with st.spinner(f"Searching for '{query}'..."):
            results = st.session_state.search_system.buscar_semanticamente(
                query, top_k=top_k, threshold=threshold
            )
        
        if not results:
            st.warning("No matching products found. Try a different query or lower the threshold.")
        else:
            st.success(f"Found {len(results)} matching products")
            
            for i, result in enumerate(results, 1):
                with st.expander(f"#{i}: {result['nombre']} (Score: {result['total']:.2f})"):
                    st.markdown(f"""
                    **Code:** {result['codigo']}  
                    **Category:** {result['categoria']}  
                    **Name Similarity:** {result['name_sim']:.3f}  
                    **Description Similarity:** {result['desc_sim']:.3f}
                    """)

if __name__ == "__main__":
    main()
