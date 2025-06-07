from typing import List, Dict
import pandas as pd
from datetime import datetime
from functools import lru_cache
import streamlit as st

class InventoryReports:
    def __init__(self, db):
        self.db = db
    
    def generar_reporte_movimientos(self, 
                                  producto_id: int = None, 
                                  fecha_inicio: str = None, 
                                  fecha_fin: str = None) -> pd.DataFrame:
        """Generate movements report with optional filters"""
        try:
            # Validate dates
            if fecha_inicio and fecha_fin:
                if pd.to_datetime(fecha_inicio) > pd.to_datetime(fecha_fin):
                    raise ValueError("Start date cannot be after end date")
            
            query = """
            SELECT m.id, p.codigo, p.nombre, m.tipo, m.cantidad, 
                   m.precio_unitario, m.precio_total, m.fecha_hora,
                   m.documento, m.responsable,
                   p.unidad_medida
            FROM movimientos m
            JOIN productos p ON m.producto_id = p.id
            WHERE p.activo = TRUE
            """
            params = []
            
            if producto_id:
                query += " AND m.producto_id = ?"
                params.append(producto_id)
            
            if fecha_inicio:
                query += " AND DATE(m.fecha_hora) >= ?"
                params.append(fecha_inicio)
            
            if fecha_fin:
                query += " AND DATE(m.fecha_hora) <= ?"
                params.append(fecha_fin)
            
            query += " ORDER BY m.fecha_hora DESC"
            
            cursor = self.db.conn.cursor()
            cursor.execute(query, params)
            
            # Convert to DataFrame with better column names
            columns = ['ID', 'Código', 'Producto', 'Tipo', 'Cantidad',
                      'Precio Unitario', 'Total', 'Fecha', 
                      'Documento', 'Responsable', 'Unidad']
            data = cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            return df
            
        except sqlite3.Error as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
            return pd.DataFrame()

    @lru_cache(maxsize=12)
    def generar_reporte_inventario(self, 
                                 mes: int = None, 
                                 anio: int = None) -> pd.DataFrame:
        """Generate inventory valuation report"""
        try:
            now = datetime.now()
            mes = mes or now.month
            anio = anio or now.year
            
            if not (1 <= mes <= 12):
                raise ValueError("Month must be 1-12")
            if not (2000 <= anio <= 2100):
                raise ValueError("Year must be between 2000-2100")
            
            query = """
            SELECT p.codigo, p.nombre, p.categoria, p.unidad_medida,
                   e.stock_inicial, e.entradas, e.salidas, e.stock_final,
                   e.valor_inicial, e.valor_entradas, e.valor_salidas, e.valor_final,
                   ROUND(e.valor_final / NULLIF(e.stock_final, 0), 2) as costo_promedio
            FROM existencias e
            JOIN productos p ON e.producto_id = p.id
            WHERE e.mes = ? AND e.anio = ? AND p.activo = TRUE
            ORDER BY p.nombre
            """
            
            cursor = self.db.conn.cursor()
            cursor.execute(query, (mes, anio))
            
            columns = ['Código', 'Producto', 'Categoría', 'Unidad',
                      'Stock Inicial', 'Entradas', 'Salidas', 'Stock Final',
                      'Valor Inicial', 'Valor Entradas', 'Valor Salidas', 
                      'Valor Final', 'Costo Promedio']
            data = cursor.fetchall()
            
            return pd.DataFrame(data, columns=columns)
            
        except Exception as e:
            st.error(f"Error generating inventory report: {str(e)}")
            return pd.DataFrame()

    def generar_reporte_stock_minimo(self) -> pd.DataFrame:
        """Generate low stock alert report"""
        try:
            query = """
            SELECT p.codigo, p.nombre, p.categoria, p.unidad_medida,
                   p.stock_minimo as "Stock Mínimo", 
                   COALESCE((
                       SELECT e.stock_final 
                       FROM existencias e 
                       WHERE e.producto_id = p.id 
                       ORDER BY e.anio DESC, e.mes DESC 
                       LIMIT 1
                   ), 0) as "Stock Actual",
                   (p.stock_minimo - COALESCE((
                       SELECT e.stock_final 
                       FROM existencias e 
                       WHERE e.producto_id = p.id 
                       ORDER BY e.anio DESC, e.mes DESC 
                       LIMIT 1
                   ), 0)) as "Diferencia"
            FROM productos p
            WHERE p.activo = TRUE
            HAVING "Stock Actual" < "Stock Mínimo"
            ORDER BY "Diferencia" DESC
            """
            
            cursor = self.db.conn.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            df['% del Mínimo'] = (df['Stock Actual'] / df['Stock Mínimo'] * 100).round(1)
            return df
            
        except Exception as e:
            st.error(f"Error generating low stock report: {str(e)}")
            return pd.DataFrame()

