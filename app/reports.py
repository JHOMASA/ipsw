from app.typing import List, Dict
import pandas as pd
from app.datetime import datetime

class InventoryReports:
    def __init__(self, db):
        self.db = db
    
    def generar_reporte_movimientos(self, producto_id: int = None, 
                                  fecha_inicio: str = None, 
                                  fecha_fin: str = None) -> pd.DataFrame:
        """Generate movements report with optional filters"""
        query = """
        SELECT m.id, p.codigo, p.nombre, m.tipo, m.cantidad, 
               m.precio_unitario, m.precio_total, m.fecha_hora,
               m.documento, m.responsable
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
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        return pd.DataFrame(data, columns=columns)
    
    def generar_reporte_inventario(self, mes: int = None, anio: int = None) -> pd.DataFrame:
        """Generate inventory report for a specific month/year"""
        if not mes or not anio:
            now = datetime.now()
            mes = now.month
            anio = now.year
        
        query = """
        SELECT p.codigo, p.nombre, p.categoria, p.unidad_medida,
               e.stock_inicial, e.entradas, e.salidas, e.stock_final,
               e.valor_inicial, e.valor_entradas, e.valor_salidas, e.valor_final
        FROM existencias e
        JOIN productos p ON e.producto_id = p.id
        WHERE e.mes = ? AND e.anio = ? AND p.activo = TRUE
        ORDER BY p.nombre
        """
        
        cursor = self.db.conn.cursor()
        cursor.execute(query, (mes, anio))
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        return pd.DataFrame(data, columns=columns)
    
    def generar_reporte_stock_minimo(self) -> pd.DataFrame:
        """Generate report of products below minimum stock"""
        query = """
        SELECT p.codigo, p.nombre, p.categoria, p.unidad_medida,
               p.stock_minimo, 
               COALESCE((
                   SELECT e.stock_final 
                   FROM existencias e 
                   WHERE e.producto_id = p.id 
                   ORDER BY e.anio DESC, e.mes DESC 
                   LIMIT 1
               ), 0) as stock_actual
        FROM productos p
        WHERE p.activo = TRUE
        AND stock_actual < p.stock_minimo
        ORDER BY (p.stock_minimo - stock_actual) DESC
        """
        
        cursor = self.db.conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()
        
        return pd.DataFrame(data, columns=columns)
