import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from app.database import InventoryDB
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

class SemanticSearch:
    def __init__(self, db_path: str = "data/inventory.db"):
        # Initialize model
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        
        # Initialize database
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row  # Enable dictionary-style access
        
        # Create table if not exists
        self._init_db()

    def _init_db(self):
        """Initialize database tables if they don't exist"""
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
        self.db.commit()

    def encode(self, text: str) -> np.ndarray:
        """Generate embeddings for text"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    @lru_cache(maxsize=1000)
    def _get_product_text(self, producto_id: int) -> Optional[Dict]:
        """Cache product text queries"""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT nombre, categoria, notas FROM productos WHERE id = ?", 
            (producto_id,)
        )
        if row := cursor.fetchone():
            return dict(row)
        return None

    def generar_embeddings_producto(self, producto_id: int, batch_size: int = 10):
        """Generate embeddings with batch processing support"""
        product_data = self._get_product_text(producto_id)
        if not product_data:
            raise ValueError(f"Product {producto_id} not found")
        
        nombre = product_data['nombre']
        descripcion = f"{nombre} {product_data['categoria'] or ''} {product_data['notas'] or ''}".strip()
        
        # Batch processing for efficiency
        with ThreadPoolExecutor() as executor:
            nombre_embedding, desc_embedding = list(executor.map(
                self.encode,  # Using our encode method instead of SentenceTransformer
                [nombre, descripcion]
            ))
        
        cursor = self.db.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO producto_embeddings 
        (producto_id, nombre_embedding, descripcion_embedding)
        VALUES (?, ?, ?)
        """, (
            producto_id, 
            np.array(nombre_embedding).tobytes(), 
            np.array(desc_embedding).tobytes()
        ))
        self.db.commit()
    def buscar_semanticamente(self, consulta: str, top_k: int = 5, threshold: float = 0.3) -> List[Dict]:
        """Enhanced semantic search with better performance"""
        consulta_embedding = self.model.encode(consulta)
        
        cursor = self.db.cursor()
        cursor.execute("""
        SELECT p.id, p.nombre, p.codigo, p.categoria,
               e.nombre_embedding, e.descripcion_embedding
        FROM productos p
        LEFT JOIN producto_embeddings e ON p.id = e.producto_id
        WHERE p.activo = TRUE
        ORDER BY e.last_updated DESC  # Prioritize recently updated
        """)
        
        results = []
        products_needing_embeddings = []
        
        for row in cursor.fetchall():
            row = dict(row)
            if not row['nombre_embedding']:
                products_needing_embeddings.append(row['id'])
                continue
                
            similarities = self._calculate_similarities(
                consulta_embedding,
                np.frombuffer(row['nombre_embedding'], dtype=np.float32),
                np.frombuffer(row['descripcion_embedding'], dtype=np.float32)
            )
            
            if similarities['total'] >= threshold:
                results.append({
                    **row,
                    **similarities,
                    'nombre_embedding': None,  # Remove blob data from results
                    'descripcion_embedding': None
                })
        
        # Generate missing embeddings in parallel
        if products_needing_embeddings:
            with ThreadPoolExecutor() as executor:
                executor.map(self.generar_embeddings_producto, products_needing_embeddings)
            return self.buscar_semanticamente(consulta, top_k, threshold)
        
        return sorted(results, key=lambda x: x['total'], reverse=True)[:top_k]

    def _calculate_similarities(self, query_embedding: np.ndarray, 
                              name_embedding: np.ndarray, 
                              desc_embedding: np.ndarray) -> Dict[str, float]:
        """Calculate all similarity metrics"""
        return {
            'name_sim': self._cosine_similarity(query_embedding, name_embedding),
            'desc_sim': self._cosine_similarity(query_embedding, desc_embedding),
            'total': 0.6 * self._cosine_similarity(query_embedding, name_embedding) + 
                    0.4 * self._cosine_similarity(query_embedding, desc_embedding)
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Optimized cosine similarity calculation"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return np.dot(a, b) / (a_norm * b_norm)

    def __del__(self):
        """Clean up resources"""
        self.db.close()
