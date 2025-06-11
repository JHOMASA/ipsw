from sqlite3 import dbapi2 as sqlite3
from typing import Dict, List, Optional
import os
from pathlib import Path
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Retrieve token and optional local password
hf_token = os.getenv("HF_TOKEN")
hf_password = os.getenv("HF_PASSWORD")  # Optional password for local control

# Optional local access check
if hf_password != "my_secure_password":
    raise EnvironmentError("Invalid local HF password. Check your .env file.")

# Authenticate with Hugging Face using token
if not hf_token:
    raise EnvironmentError("Hugging Face token (HF_TOKEN) not set in environment.")
login(token=hf_token)

class InventoryDB:
    def __init__(self, db_path: str = None):
        """Initialize database with proper path handling"""
        try:
            if db_path is None:
                db_path = Path(__file__).parent / "data" / "inventory.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_path = str(db_path)

            self.conn = sqlite3.connect(db_path)
            self.conn.execute("PRAGMA foreign_keys = ON")
            self._init_db()
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {str(e)}")

    def _init_db(self):
        """Initialize database structure"""
        cursor = self.conn.cursor()

        # Products table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS productos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo TEXT UNIQUE NOT NULL,
            nombre TEXT NOT NULL,
            categoria TEXT,
            unidad_medida TEXT DEFAULT 'unidades',
            stock_minimo INTEGER DEFAULT 0,
            precio_unitario DECIMAL(10,2) DEFAULT 0,
            notas TEXT,
            activo BOOLEAN DEFAULT TRUE,
            empresa_id INTEGER DEFAULT 1
        )
        """)

        # Movements table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS movimientos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            producto_id INTEGER NOT NULL,
            tipo TEXT CHECK(tipo IN ('entrada', 'salida')),
            cantidad INTEGER NOT NULL,
            precio_unitario DECIMAL(10,2) NOT NULL,
            precio_total DECIMAL(10,2) GENERATED ALWAYS AS (cantidad * precio_unitario) STORED,
            fecha_hora DATETIME DEFAULT CURRENT_TIMESTAMP,
            documento TEXT,
            responsable TEXT,
            notas TEXT,
            empresa_id INTEGER DEFAULT 1,
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        )
        """)

        # Monthly inventory table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS existencias (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            producto_id INTEGER NOT NULL,
            mes INTEGER NOT NULL CHECK (mes BETWEEN 1 AND 12),
            anio INTEGER NOT NULL,
            stock_inicial INTEGER NOT NULL,
            entradas INTEGER NOT NULL DEFAULT 0,
            salidas INTEGER NOT NULL DEFAULT 0,
            stock_final INTEGER NOT NULL,
            valor_inicial DECIMAL(15,2) NOT NULL,
            valor_entradas DECIMAL(15,2) NOT NULL DEFAULT 0,
            valor_salidas DECIMAL(15,2) NOT NULL DEFAULT 0,
            valor_final DECIMAL(15,2) NOT NULL,
            empresa_id INTEGER DEFAULT 1,
            UNIQUE(producto_id, mes, anio, empresa_id),
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        )
        """)

        # Product embeddings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS producto_embeddings (
            producto_id INTEGER PRIMARY KEY,
            nombre_embedding BLOB,
            descripcion_embedding BLOB,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (producto_id) REFERENCES productos(id)
        )
        """)

        self.conn.commit()


