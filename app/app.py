import streamlit as st
from database import InventoryDB
from registration import RegistrationSystem
from reports import InventoryReports
from semantic_search import SemanticSearch

def main():
    st.set_page_config(page_title="Inventory System", layout="wide")
    db = InventoryDB()
    
    # Initialize components
    registros = RegistrationSystem(db)
    reportes = InventoryReports(db)
    buscador = SemanticSearch()
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Register", "Reports", "Search"])
    
    if page == "Register":
        show_register_page(registros)
    elif page == "Reports":
        show_reports_page(db, reportes)
    elif page == "Search":
        show_search_page(db, reportes, buscador)

def show_register_page(registros):
    st.header("📝 Register Movement")
    with st.form("movement_form"):
        productos = get_product_list(registros.db)
        producto_id = st.selectbox("Product", options=productos.keys(), format_func=lambda x: productos[x])
        tipo = st.radio("Type", ["entrada", "salida"])
        cantidad = st.number_input("Quantity", min_value=1)
        precio = st.number_input("Unit Price", min_value=0.0, format="%.2f")
        
        if st.form_submit_button("Register"):
            movimiento = {
                'producto_id': producto_id,
                'tipo': tipo,
                'cantidad': cantidad,
                'precio_unitario': precio
            }
            try:
                registros.registrar_movimiento(movimiento)
                st.success("Movement registered!")
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
