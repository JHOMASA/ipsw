import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from database import InventoryDB
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import logging
import torch
from datetime import datetime

class RegistrationSystem:
    def __init__(self, db):
        self.db = db
        self.calculator = InventoryCalculator(db)

    def registrar_movimiento(self, movimiento: Dict):
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id FROM productos WHERE id = ? AND activo = TRUE", (movimiento['producto_id'],))
        if not cursor.fetchone():
            raise ValueError("Product not found or inactive")

        cursor.execute("""
        INSERT INTO movimientos (producto_id, tipo, cantidad, precio_unitario, fecha_hora, empresa_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            movimiento['producto_id'],
            movimiento['tipo'],
            movimiento['cantidad'],
            movimiento['precio_unitario'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            1
        ))

        now = datetime.now()
        existencias = self.calculator.calcular_existencias_mes(
            movimiento['producto_id'], now.month, now.year, 1
        )
        self._actualizar_existencias(existencias)
        self.db.conn.commit()

    def _actualizar_existencias(self, existencias: Dict):
        cursor = self.db.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO existencias (
            producto_id, mes, anio, stock_inicial, entradas, salidas, stock_final,
            valor_inicial, valor_entradas, valor_salidas, valor_final, empresa_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            existencias['producto_id'],
            existencias['mes'],
            existencias['anio'],
            existencias['stock_inicial'],
            existencias['entradas'],
            existencias['salidas'],
            existencias['stock_final'],
            existencias['valor_inicial'],
            existencias['valor_entradas'],
            existencias['valor_salidas'],
            existencias['valor_final'],
            existencias['empresa_id']
        ))
