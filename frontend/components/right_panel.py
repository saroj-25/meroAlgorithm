import streamlit as st

def render_right_panel():

    st.html("""
    <div class="right-panel">

        <!-- RAG Pipeline Overview -->
        <div class="rp-card">
            <h4 class="rp-card-title">RAG Pipeline Overview</h4>
            <div class="pipeline-flow">
                <div class="pipeline-step">
                    <div class="pipeline-icon purple-icon">
                        👤
                    </div>
                    <span class="pipeline-step-label">User Query</span>
                </div>
                <div class="pipeline-arrow">🡺</div>
                <div class="pipeline-step">
                    <div class="pipeline-icon blue-icon">
                        🧠
                    </div>
                    <span class="pipeline-step-label">NLP Understanding</span>
                </div>
                <div class="pipeline-arrow">🡺</div>
                <div class="pipeline-step">
                    <div class="pipeline-icon green-icon">
                        🗄️
                    </div>
                    <span class="pipeline-step-label">Retrieve (Vector DB)</span>
                </div>
                <div class="pipeline-arrow">🡺</div>
                <div class="pipeline-step">
                    <div class="pipeline-icon orange-icon">
                        💡
                    </div>
                    <span class="pipeline-step-label">Generate Answer</span>
                </div>
            </div>
            <a class="pipeline-link" href="#">View Pipeline Details 🡺</a>
        </div>

        <!-- Retrieved Documents -->
        <div class="rp-card">
            <div class="rp-card-header">
                <span class="rp-section-icon">📄</span>
                <h4 class="rp-card-title">Retrieved Documents (Top 3)</h4>
            </div>
            <div class="doc-list">
                <div class="doc-item">
                    <div class="doc-info">
                        <span class="doc-index">1.</span>
                        <div class="doc-text">
                            <span class="doc-title">Minimax Algorithm Explained</span>
                            <span class="doc-source">AI Notes – Chapter 4</span>
                        </div>
                    </div>
                    <span class="doc-score high">0.92</span>
                </div>
                <div class="doc-item">
                    <div class="doc-info">
                        <span class="doc-index">2.</span>
                        <div class="doc-text">
                            <span class="doc-title">Game Playing AI – Minimax</span>
                            <span class="doc-source">Lecture Slides</span>
                        </div>
                    </div>
                    <span class="doc-score mid">0.89</span>
                </div>
                <div class="doc-item">
                    <div class="doc-info">
                        <span class="doc-index">3.</span>
                        <div class="doc-text">
                            <span class="doc-title">Tic Tac Toe and Minimax</span>
                            <span class="doc-source">DSA Guide Book</span>
                        </div>
                    </div>
                    <span class="doc-score low">0.86</span>
                </div>
            </div>
            <a class="pipeline-link" href="#">View All Sources 🡺</a>
        </div>

        <!-- Final Answer -->
        <div class="rp-card answer-card">
            <div class="rp-card-header">
                <h4 class="rp-card-title answer-title">Final Answer (AI Generated)</h4>
            </div>
            <p class="answer-text">
                Minimax ek decision-making algorithm ho jo duita player games (zero-sum) ma use huncha.
                AI le sabai possible moves analyze garera best move choose garcha, j bata usle maximum score paos.
                Yo algorithm Tic-Tac-Toe, Chess, Checkers jasta games ma use huncha.
            </p>
            <button class="copy-btn">
                📋 Copy Answer
            </button>
        </div>

        <!-- Ask Follow-up -->
        <div class="rp-card followup-card">
            <div class="rp-card-header">
                <span class="rp-section-icon">💬</span>
                <h4 class="rp-card-title">Ask Follow-up</h4>
            </div>
            <div class="suggestion-chips">
                <button class="suggestion-chip">minimax vs alpha beta pruning?</button>
                <button class="suggestion-chip">time complexity k ho?</button>
            </div>
            <div class="followup-input-row">
                <input class="followup-input" type="text" placeholder="Follow-up question sodhnus..." />
                <button class="followup-send">
                    ➤
                </button>
            </div>
        </div>

    </div>
    """)