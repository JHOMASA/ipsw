from typing import Dict
from datetime import datetime
from inventory_logic import InventoryCalculator
from database import InventoryDB

class RegistrationSystem:
    def __init__(self, db):
        self.db = db
        self.calculator = InventoryCalculator(db)
    
    def registrar_movimiento(self, movimiento: Dict):
        """Register a movement and update inventory"""
        cursor = self.db.conn.cursor()
        
        # Verify product exists and is active
        cursor.execute("SELECT id FROM productos WHERE id = ? AND activo = TRUE", 
                       (movimiento['producto_id'],))
        if not cursor.fetchone():
            raise ValueError("Product not found or inactive")
        
        # Register movement
        cursor.execute("""
        INSERT INTO movimientos (
            producto_id, tipo, cantidad, precio_unitario, fecha_hora, empresa_id
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            movimiento['producto_id'],
            movimiento['tipo'],
            movimiento['cantidad'],
            movimiento['precio_unitario'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            1  # empresa_id
        ))
        
        # Get current month/year for inventory calculation
        now = datetime.now()
        mes = now.month
        anio = now.year
        
        # Calculate and update inventory
        existencias = self.calculator.calcular_existencias_mes(
            movimiento['producto_id'], mes, anio, 1
        )
        
        self._actualizar_existencias(existencias)
        self.db.conn.commit()
    
    def _actualizar_existencias(self, existencias: Dict):
        """Update or insert monthly inventory data"""
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
    
    def get_product_list(self) -> Dict[int, str]:
        """Get active products as {id: name} dictionary"""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT id, nombre FROM productos WHERE activo = TRUE")
        return {row[0]: row[1] for row in cursor.fetchall()}
