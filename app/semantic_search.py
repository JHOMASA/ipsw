import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from app.database import InventoryDB
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, db_path: str = "data/inventory.db", model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize semantic search system with:
        - DB connection
        - Embedding model
        - Thread pool for parallel processing
        """
        self.model = SentenceTransformer(model_name)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row  # Enable dict-style row access
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._init_db()
        logger.info(f"Initialized SemanticSearch with model {model_name}")

    def _init_db(self):
        """Initialize database tables with proper error handling"""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS producto_embeddings (
                producto_id INTEGER PRIMARY KEY,
                nombre_embedding BLOB,
                descripcion_embedding BLOB,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (producto_id) REFERENCES productos(id) ON DELETE CASCADE
            )""")
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_producto_id 
            ON producto_embeddings(producto_id)""")
            self.db.commit()
        except sqlite3.Error as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    @lru_cache(maxsize=1000)
    def _get_product_text(self, producto_id: int) -> Optional[Dict]:
        """Cache product text queries with error handling"""
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
            logger.warning(f"Failed to fetch product {producto_id}: {str(e)}")
            return None

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Enhanced embedding generation with:
        - Batch support
        - Empty text handling
        - Automatic device detection
        """
        if isinstance(text, str):
            if not text.strip():
                return np.zeros(self.model.get_sentence_embedding_dimension())
            text = [text]
        
        return self.model.encode(
            text,
            convert_to_numpy=True,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            normalize_embeddings=True
        )

    def generar_embeddings_producto(self, producto_id: int):
        """
        Generate and store embeddings for a product with:
        - Automatic text concatenation
        - Parallel encoding
        - Transaction safety
        """
        product_data = self._get_product_text(producto_id)
        if not product_data:
            raise ValueError(f"Product {producto_id} not found")
        
        nombre = product_data['nombre']
        descripcion = f"{nombre} {product_data['categoria'] or ''} {product_data['notas'] or ''}".strip()
        
        try:
            # Process in parallel
            nombre_embedding, desc_embedding = self.encode([nombre, descripcion])
            
            with self.db:  # Transaction
                self.db.execute("""
                INSERT OR REPLACE INTO producto_embeddings 
                (producto_id, nombre_embedding, descripcion_embedding)
                VALUES (?, ?, ?)
                """, (
                    producto_id, 
                    nombre_embedding.tobytes(), 
                    desc_embedding.tobytes()
                ))
        except Exception as e:
            logger.error(f"Failed generating embeddings for {producto_id}: {str(e)}")
            raise

    def buscar_semanticamente(self, consulta: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """
        Enhanced semantic search with:
        - Hybrid scoring (name + description)
        - Automatic embedding generation for missing products
        - Proper result formatting
        """
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
            
            # Generate missing embeddings if needed
            if needs_embedding:
                logger.info(f"Generating embeddings for {len(needs_embedding)} products")
                list(self.thread_pool.map(
                    self.generar_embeddings_producto, 
                    needs_embedding
                ))
                return self.buscar_semanticamente(consulta, top_k, threshold)
            
            return sorted(results, key=lambda x: x['total'], reverse=True)[:top_k]
        
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def _calculate_similarities(self, 
                              query_embedding: np.ndarray, 
                              name_embedding: np.ndarray, 
                              desc_embedding: np.ndarray) -> Dict[str, float]:
        """
        Calculate weighted similarity scores with:
        - Name priority (60%)
        - Description support (40%)
        - NaN protection
        """
        def safe_similarity(a, b):
            with np.errstate(invalid='ignore'):
                sim = self._cosine_similarity(a, b)
            return sim if not np.isnan(sim) else 0.0
        
        return {
            'name_sim': safe_similarity(query_embedding, name_embedding),
            'desc_sim': safe_similarity(query_embedding, desc_embedding),
            'total': 0.6 * safe_similarity(query_embedding, name_embedding) + 
                    0.4 * safe_similarity(query_embedding, desc_embedding)
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Optimized cosine similarity with norm checks"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b) / (a_norm * b_norm) if a_norm * b_norm > 0 else 0.0

    def close(self):
        """Proper resource cleanup"""
        self.thread_pool.shutdown()
        self.db.close()
        logger.info("Resources cleaned up")

    def __del__(self):
        """Destructor for safety"""
        self.close()
