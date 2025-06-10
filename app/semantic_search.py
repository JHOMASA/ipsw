import os
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
import typing
from pathlib import Path
from database import InventoryDB
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import logging
import torch
from datetime import datetime

# Workaround for PyTorch + Streamlit bug
if hasattr(torch, '__path__'):
    torch.__path__ = [p for p in torch.__path__ if "__path__._path" not in p]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, db_path: str = None):
        self.model_name = 'paraphrase-MiniLM-L6-v2'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        if db_path is None:
            db_path = Path(__file__).parent / "data" / "inventory.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path = str(db_path)

        try:
            self.db = sqlite3.connect(db_path)
            self.db.execute("PRAGMA foreign_keys = ON")
            self._init_db()
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {str(e)}")

    def _init_db(self):
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
        try:
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            logger.error(f"Encoding failed for text: {str(e)}")
            return np.zeros(self.model.config.hidden_size)

    @lru_cache(maxsize=1000)
    def _get_product_text(self, producto_id: int) -> typing.Optional[typing.Dict]:
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT nombre, categoria, notas FROM productos WHERE id = ?", (producto_id,))
            row = cursor.fetchone()
            if row:
                return {'nombre': row[0], 'categoria': row[1], 'notas': row[2]}
            return None
        except sqlite3.Error as e:
            logger.warning(f"Database query failed: {str(e)}")
            return None

    def generar_embeddings_producto(self, producto_id: int):
        product_data = self._get_product_text(producto_id)
        if not product_data:
            raise ValueError(f"Product {producto_id} not found")

        try:
            nombre = product_data['nombre']
            descripcion = f"{nombre} {product_data['categoria'] or ''} {product_data['notas'] or ''}".strip()
            with ThreadPoolExecutor() as executor:
                nombre_embedding, desc_embedding = list(executor.map(self.encode, [nombre, descripcion]))

            with self.db:
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

    def buscar_semanticamente(self, consulta: str, top_k: int = 5, threshold: float = 0.3) -> typing.List[typing.Dict]:
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
                row = dict(zip(["id", "nombre", "codigo", "categoria", "nombre_embedding", "descripcion_embedding"], row))
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

    def _calculate_similarities(self, query_embedding: np.ndarray, name_embedding: np.ndarray, desc_embedding: np.ndarray) -> typing.Dict[str, float]:
        def safe_sim(a, b):
            sim = self._cosine_similarity(a, b)
            return sim if not np.isnan(sim) else 0.0

        return {
            'name_sim': safe_sim(query_embedding, name_embedding),
            'desc_sim': safe_sim(query_embedding, desc_embedding),
            'total': 0.6 * safe_sim(query_embedding, name_embedding) + 0.4 * safe_sim(query_embedding, desc_embedding)
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        return np.dot(a, b) / (a_norm * b_norm) if a_norm * b_norm > 0 else 0.0

    def close(self):
        if hasattr(self, 'db'):
            self.db.close()
        logger.info("Resources cleaned up")

    def __del__(self):
        self.close()




