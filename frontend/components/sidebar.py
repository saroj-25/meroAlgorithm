import streamlit as st

def menu_item(icon, text, active=False):
    cls = "sidebar-item active" if active else "sidebar-item"

    st.markdown(
        f"""
        <div class="{cls}">
            <span class="sidebar-icon">{icon}</span>
            <span>{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar():

    with st.sidebar:

        st.markdown("""
        <div class="logo-container">
            <div class="logo-icon">&lt;/&gt;</div>
            <div>
                <div class="logo-title">
                    <span class="orange">mero</span>Algorithm
                </div>
                <div class="logo-subtitle">
                    Visualize • Learn • Understand
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        menu_item("🏠", "Dashboard")

        st.markdown(
            '<div class="section-title">LEARNING MODULES</div>',
            unsafe_allow_html=True
        )

        menu_item("🧬", "DSA Visualizer", True)
        menu_item("🕸️", "Graph Algorithms")
        menu_item("🌳", "Tree Algorithms")
        menu_item("↕️", "Sorting Algorithms")
        menu_item("🌐", "Minimax (Game AI)")
        menu_item("📈", "Dynamic Programming")
        menu_item("💬", "NLP Interpreter")
        menu_item("🔍", "RAG Search Engine")
        menu_item("✨", "Complexity Analyzer")
        menu_item("💻", "Code Playground")

        st.markdown(
            '<div class="section-title">OTHER</div>',
            unsafe_allow_html=True
        )

        menu_item("📝", "Notes")
        menu_item("🔖", "Bookmarks")
        menu_item("🕒", "History")
        menu_item("⚙️", "Settings")
        menu_item("❔", "Help & Support")

   