from datetime import datetime
from typing import List, Dict, Optional

class InventoryCalculator:
    def __init__(self, db):
        self.db = db
    
    def calcular_existencias_mes(self, producto_id: int, mes: int, anio: int, empresa_id: int = 1) -> Dict:
        """Calculate monthly inventory with monetary valuation"""
        prev_month, prev_year = self._get_previous_month(mes, anio)
        prev_data = self._obtener_datos_mes_anterior(producto_id, prev_month, prev_year, empresa_id)
        movimientos = self._obtener_movimientos_mes(producto_id, mes, anio, empresa_id)
        
        stock_inicial = prev_data['stock_final'] if prev_data else 0
        valor_inicial = prev_data['valor_final'] if prev_data else 0
        
        entradas = sum(m['cantidad'] for m in movimientos if m['tipo'] == 'entrada')
        salidas = sum(m['cantidad'] for m in movimientos if m['tipo'] == 'salida')
        
        valor_entradas = sum(m['precio_total'] for m in movimientos if m['tipo'] == 'entrada')
        valor_salidas = sum(m['precio_total'] for m in movimientos if m['tipo'] == 'salida')
        
        return {
            'producto_id': producto_id,
            'mes': mes,
            'anio': anio,
            'empresa_id': empresa_id,
            'stock_inicial': stock_inicial,
            'entradas': entradas,
            'salidas': salidas,
            'stock_final': stock_inicial + entradas - salidas,
            'valor_inicial': valor_inicial,
            'valor_entradas': valor_entradas,
            'valor_salidas': valor_salidas,
            'valor_final': valor_inicial + valor_entradas - valor_salidas
        }
    
    def _get_previous_month(self, mes: int, anio: int) -> tuple:
        if mes == 1: return 12, anio - 1
        return mes - 1, anio
    
    def _obtener_datos_mes_anterior(self, producto_id: int, mes: int, anio: int, empresa_id: int) -> Optional[Dict]:
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT stock_final, valor_final FROM existencias 
        WHERE producto_id = ? AND mes = ? AND anio = ? AND empresa_id = ?
        """, (producto_id, mes, anio, empresa_id))
        result = cursor.fetchone()
        return {'stock_final': result[0], 'valor_final': result[1]} if result else None
    
    def _obtener_movimientos_mes(self, producto_id: int, mes: int, anio: int, empresa_id: int) -> List[Dict]:
        cursor = self.db.conn.cursor()
        cursor.execute("""
        SELECT tipo, cantidad, precio_unitario, precio_total
        FROM movimientos
        WHERE producto_id = ? 
        AND strftime('%m', fecha_hora) = ?
        AND strftime('%Y', fecha_hora) = ?
        AND empresa_id = ?
        """, (producto_id, f"{mes:02d}", str(anio), empresa_id))
        
        return [{
            'tipo': row[0],
            'cantidad': row[1],
            'precio_unitario': row[2],
            'precio_total': row[3]
        } for row in cursor.fetchall()]
