import sys
import os
from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta  # Added missing imports
import torch
if "torch.classes" not in sys.modules:
    sys.modules["torch.classes"] = types.ModuleType("torch.classes")
    sys.modules["torch.classes"].__path__ = []
    
# Configure paths
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# Now use direct imports
from registration import RegistrationSystem
from reports import InventoryReports
from semantic_search import SemanticSearch
from database import InventoryDB

def main():
    # Configure page
    st.set_page_config(
        page_title="Inventory Management System",
        layout="wide",
        page_icon="📦",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components with error handling
    try:
        db = InventoryDB()
        registros = RegistrationSystem(db)
        reportes = InventoryReports(db)
        buscador = SemanticSearch()
    except Exception as e:
        st.error(f"Failed to initialize database: {str(e)}")
        st.stop()
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
        .main { padding-top: 1.5rem; }
        .sidebar .sidebar-content { padding-top: 1rem; }
        .stRadio > div { flex-direction: row; }
        .stRadio > label { font-weight: bold; }
        .stAlert { padding: 0.5rem; }
        .st-bb { border-bottom: 1px solid #eee; }
        .stForm { border: 1px solid #eee; border-radius: 0.5rem; padding: 1.5rem; }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation with icons and better organization
    st.sidebar.title("📦 Inventory System")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        options=["Register", "Reports", "Search"],
        format_func=lambda x: {
            "Register": "📝 Register Movement",
            "Reports": "📊 View Reports",
            "Search": "🔍 Search Products"
        }[x]
    )
    
    # Add system info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Information**")
    st.sidebar.markdown(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Page routing
    if page == "Register":
        show_register_page(registros)
    elif page == "Reports":
        show_reports_page(db, reportes)
    elif page == "Search":
        show_search_page(db, reportes, buscador)

def get_product_list(db):
    """Get active products with caching"""
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def _get_products(_db):
        cursor = _db.conn.cursor()
        cursor.execute("SELECT id, codigo, nombre FROM productos WHERE activo = TRUE ORDER BY nombre")
        return {row[0]: f"{row[1]} - {row[2]}" for row in cursor.fetchall()}
    return _get_products(db)

def show_register_page(registros):
    st.header("📝 Register Inventory Movement")
    
    with st.expander("ℹ️ Help", expanded=False):
        st.markdown("""
        - **Entrada**: Register new items entering inventory
        - **Salida**: Register items leaving inventory
        - All fields are required
        """)
    
    with st.form("movement_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced product selection with search capability
            productos = get_product_list(registros.db)
            producto_id = st.selectbox(
                "Select Product*",
                options=productos.keys(),
                format_func=lambda x: productos[x],
                help="Select the product for this movement"
            )
            
            # Show current stock for selected product
            if producto_id:
                cursor = registros.db.conn.cursor()
                cursor.execute("""
                SELECT stock_final FROM existencias 
                WHERE producto_id = ? 
                ORDER BY anio DESC, mes DESC 
                LIMIT 1
                """, (producto_id,))
                stock = cursor.fetchone()
                current_stock = stock[0] if stock else 0
                st.metric("Current Stock", current_stock)
        
        with col2:
            tipo = st.radio(
                "Movement Type*",
                options=["entrada", "salida"],
                format_func=lambda x: "Inbound (entrada)" if x == "entrada" else "Outbound (salida)",
                horizontal=True
            )
            
            cantidad = st.number_input(
                "Quantity*",
                min_value=1,
                max_value=10000 if tipo == "entrada" else current_stock if producto_id else 10000,
                step=1,
                help="Number of units moving in/out"
            )
            
            precio = st.number_input(
                "Unit Price (USD)*",
                min_value=0.0,
                max_value=100000.0,
                value=0.0,
                format="%.2f",
                help="Price per unit for valuation"
            )
        
        # Additional optional fields
        with st.expander("Additional Details (Optional)"):
            doc_col, resp_col = st.columns(2)
            with doc_col:
                documento = st.text_input("Document Reference")
            with resp_col:
                responsable = st.text_input("Responsible Person")
            notas = st.text_area("Notes")
        
        submitted = st.form_submit_button("Register Movement", type="primary")
        
        if submitted:
            movimiento = {
                'producto_id': producto_id,
                'tipo': tipo,
                'cantidad': cantidad,
                'precio_unitario': precio,
                'documento': documento or None,
                'responsable': responsable or None,
                'notas': notas or None
            }
            
            try:
                registros.registrar_movimiento(movimiento)
                st.success("✅ Movement registered successfully!")
                st.balloons()
                
                # Show updated stock
                cursor = registros.db.conn.cursor()
                cursor.execute("""
                SELECT stock_final FROM existencias 
                WHERE producto_id = ? 
                ORDER BY anio DESC, mes DESC 
                LIMIT 1
                """, (producto_id,))
                new_stock = cursor.fetchone()[0]
                
                st.metric("Updated Stock", new_stock, delta=new_stock-current_stock)
                
            except Exception as e:
                st.error(f"❌ Error registering movement: {str(e)}")

def show_reports_page(db, reportes):
    st.header("📊 Inventory Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Movements", "Inventory", "Minimum Stock", "Valuation"],
        index=0,
        help="Choose the type of report to generate"
    )
    
    if report_type == "Movements":
        st.subheader("Movement History Report")
        
        with st.expander("Filters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=30),
                    max_value=datetime.now()
                )
            with col2:
                fecha_fin = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    max_value=datetime.now(),
                    min_value=fecha_inicio
                )
            
            productos = get_product_list(db)
            producto_id = st.selectbox(
                "Filter by Product (Optional)",
                options=[None] + list(productos.keys()),
                format_func=lambda x: "All Products" if x is None else productos[x]
            )
        
        if st.button("Generate Report", key="movement_report"):
            with st.spinner("Generating report..."):
                try:
                    df = reportes.generar_reporte_movimientos(
                        producto_id=producto_id if producto_id else None,
                        fecha_inicio=fecha_inicio.strftime('%Y-%m-%d'),
                        fecha_fin=fecha_fin.strftime('%Y-%m-%d')
                    )
                    
                    if len(df) > 0:
                        st.success(f"Found {len(df)} movements")
                        
                        # Show summary metrics
                        entradas = df[df['tipo'] == 'entrada']['cantidad'].sum()
                        salidas = df[df['tipo'] == 'salida']['cantidad'].sum()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Entries", entradas)
                        col2.metric("Total Exits", salidas)
                        col3.metric("Net Change", entradas - salidas)
                        
                        # Show data with better formatting
                        st.dataframe(
                            df.style
                             .format({
                                 'precio_unitario': "${:.2f}",
                                 'precio_total': "${:.2f}"
                             }),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download as CSV",
                            data=csv,
                            file_name=f"movements_{fecha_inicio}_to_{fecha_fin}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No movements found for selected filters")
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
    
    elif report_type == "Inventory":
        st.subheader("Monthly Inventory Report")
        
        col1, col2 = st.columns(2)
        with col1:
            mes = st.selectbox(
                "Month",
                range(1, 13),
                datetime.now().month - 1,
                format_func=lambda x: datetime(1900, x, 1).strftime('%B')
            )
        with col2:
            anio = st.number_input(
                "Year",
                min_value=2000,
                max_value=2100,
                value=datetime.now().year
            )
        
        if st.button("Generate Report", key="inventory_report"):
            with st.spinner("Generating inventory report..."):
                try:
                    df = reportes.generar_reporte_inventario(mes, anio)
                    
                    if len(df) > 0:
                        st.success(f"Inventory status for {datetime(anio, mes, 1).strftime('%B %Y')}")
                        
                        # Calculate totals
                        total_valor_final = df['valor_final'].sum()
                        total_items = len(df)
                        low_stock = len(df[df['stock_final'] < df['stock_minimo']])
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Items", total_items)
                        col2.metric("Total Value", f"${total_valor_final:,.2f}")
                        col3.metric("Low Stock Items", low_stock, delta_color="inverse")
                        
                        # Display dataframe with conditional formatting
                        def highlight_low_stock(row):
                            return ['background-color: #ffcccc' if row['stock_final'] < row['stock_minimo'] else '' for _ in row]
                        
                        st.dataframe(
                            df.style
                             .apply(highlight_low_stock, axis=1)
                             .format({
                                 'valor_inicial': "${:,.2f}",
                                 'valor_entradas': "${:,.2f}",
                                 'valor_salidas': "${:,.2f}",
                                 'valor_final': "${:,.2f}"
                             }),
                            use_container_width=True,
                            height=600
                        )
                    else:
                        st.warning("No inventory data found for selected period")
                
                except Exception as e:
                    st.error(f"Error generating inventory report: {str(e)}")
    
    elif report_type == "Minimum Stock":
        st.subheader("Low Stock Alert Report")
        
        if st.button("Generate Report", key="min_stock_report"):
            with st.spinner("Checking inventory levels..."):
                try:
                    df = reportes.generar_reporte_stock_minimo()
                    
                    if len(df) > 0:
                        st.warning(f"⚠️ Found {len(df)} products below minimum stock")
                        
                        # Calculate severity
                        df['deficit'] = df['stock_minimo'] - df['stock_actual']
                        df['severity'] = df['deficit'] / df['stock_minimo']
                        
                        # Sort by most critical
                        df = df.sort_values('severity', ascending=False)
                        
                        # Display with tabs
                        tab1, tab2 = st.tabs(["Table View", "Chart View"])
                        
                        with tab1:
                            st.dataframe(
                                df.style
                                 .format({'severity': "{:.0%}"})
                                 .bar(subset=['deficit'], color='#ff6b6b'),
                                use_container_width=True
                            )
                        
                        with tab2:
                            st.bar_chart(
                                df.set_index('nombre')['deficit'],
                                color="#ff6b6b"
                            )
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Low Stock Report",
                            data=csv,
                            file_name="low_stock_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.success("🎉 All products have sufficient stock!")
                
                except Exception as e:
                    st.error(f"Error generating low stock report: {str(e)}")

def show_search_page(db, reportes, buscador):
    st.header("🔍 Product Search")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search products by name, code, or description",
            placeholder="e.g., 'blue widgets' or 'HW-2023'",
            help="Uses semantic search to find relevant products"
        )
    with col2:
        top_k = st.number_input(
            "Max results",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of results to show"
        )
    
    if query:
        with st.spinner("Searching products..."):
            try:
                results = buscador.buscar_semanticamente(query, top_k=top_k)
                
                if results:
                    st.success(f"Found {len(results)} matching products")
                    
                    for res in results:
                        with st.expander(
                            f"{res['nombre']} (Score: {res['similitud']:.2f})",
                            expanded=False
                        ):
                            cols = st.columns([1, 3])
                            with cols[0]:
                                st.metric("Code", res['codigo'])
                                st.metric("Category", res['categoria'] or "N/A")
                            
                            with cols[1]:
                                # Show current stock
                                cursor = db.conn.cursor()
                                cursor.execute("""
                                SELECT stock_final FROM existencias 
                                WHERE producto_id = ? 
                                ORDER BY anio DESC, mes DESC 
                                LIMIT 1
                                """, (res['id'],))
                                stock = cursor.fetchone()
                                current_stock = stock[0] if stock else 0
                                
                                cursor.execute("""
                                SELECT stock_minimo FROM productos 
                                WHERE id = ?
                                """, (res['id'],))
                                min_stock = cursor.fetchone()[0]
                                
                                st.metric(
                                    "Current Stock",
                                    current_stock,
                                    delta=f"Minimum: {min_stock}",
                                    delta_color="inverse" if current_stock < min_stock else "normal"
                                )
                                
                                # Show last 3 movements
                                st.write("**Recent Movements**")
                                movimientos = reportes.generar_reporte_movimientos(
                                    producto_id=res['id']
                                ).head(3)
                                
                                if len(movimientos) > 0:
                                    st.dataframe(
                                        movimientos.style.format({
                                            'precio_unitario': "${:.2f}",
                                            'precio_total': "${:.2f}"
                                        }),
                                        hide_index=True,
                                        use_container_width=True
                                    )
                                else:
                                    st.info("No movement history found")
                else:
                    st.warning("No products found matching your search")
            
            except Exception as e:
                st.error(f"Search error: {str(e)}")

if __name__ == "__main__":
    main()
