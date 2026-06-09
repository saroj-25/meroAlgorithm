import streamlit as st

def render_navbar():
    st.html("""
    <div class="navbar">
        <div class="search-wrapper">
            <input
                type="text"
                placeholder="English or Nepali Roman ma sodhnus..."
            />
        </div>
        <button class="lang-btn active">NP</button>
        <button class="lang-btn">EN</button>
        <button class="ask-ai-btn">Ask AI</button>
        <button class="icon-btn">🔔</button>
        <div class="profile-card">
            <div class="avatar">👤</div>
            <div class="user-info">
                <span class="name">Puja</span>
            </div>
        </div>
    </div>
    """)