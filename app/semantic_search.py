import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class SemanticSearch:
    def __init__(self, db_path: str = "data/inventory.db"):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.db = sqlite3.connect(db_path)
    
    def generar_embeddings_producto(self, producto_id: int):
        cursor = self.db.cursor()
        cursor.execute("SELECT nombre, categoria, notas FROM productos WHERE id = ?", (producto_id,))
        nombre, categoria, notas = cursor.fetchone()
        descripcion = f"{nombre} {categoria or ''} {notas or ''}".strip()
        
        nombre_embedding = self.model.encode(nombre)
        desc_embedding = self.model.encode(descripcion)
        
        cursor.execute("""
        INSERT OR REPLACE INTO producto_embeddings 
        (producto_id, nombre_embedding, descripcion_embedding)
        VALUES (?, ?, ?)
        """, (producto_id, nombre_embedding.tobytes(), desc_embedding.tobytes()))
        self.db.commit()
    
    def buscar_semanticamente(self, consulta: str, top_k: int = 5) -> List[Dict]:
        consulta_embedding = self.model.encode(consulta)
        cursor = self.db.cursor()
        cursor.execute("""
        SELECT p.id, p.nombre, p.codigo, p.categoria, 
               e.nombre_embedding, e.descripcion_embedding
        FROM productos p
        LEFT JOIN producto_embeddings e ON p.id = e.producto_id
        WHERE p.activo = TRUE
        """)
        
        resultados = []
        for row in cursor.fetchall():
            producto_id, nombre, codigo, categoria, nombre_emb, desc_emb = row
            if not nombre_emb:
                self.generar_embeddings_producto(producto_id)
                continue
                
            nombre_embedding = np.frombuffer(nombre_emb, dtype=np.float32)
            desc_embedding = np.frombuffer(desc_emb, dtype=np.float32)
            
            sim_nombre = self._cosine_similarity(consulta_embedding, nombre_embedding)
            sim_desc = self._cosine_similarity(consulta_embedding, desc_embedding)
            sim_total = 0.6 * sim_nombre + 0.4 * sim_desc
            
            if sim_total >= 0.3:
                resultados.append({
                    'id': producto_id,
                    'codigo': codigo,
                    'nombre': nombre,
                    'categoria': categoria,
                    'similitud': sim_total
                })
        
        return sorted(resultados, key=lambda x: x['similitud'], reverse=True)[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
