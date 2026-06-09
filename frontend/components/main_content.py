import streamlit as st

def render_main_content():
    st.html("""
    <div class="main-content">

        <!-- Controls Bar -->
        <div class="controls-bar">
            <div class="control-group">
                <label class="control-label">Algorithm</label>
                <div class="custom-select">
                    <select>
                        <option>Minimax (Tic-Tac-Toe)</option>
                        <option>BFS</option>
                        <option>DFS</option>
                        <option>Dijkstra</option>
                        <option>A* Search</option>
                    </select>
                    <span class="select-arrow">▾</span>
                </div>
            </div>
            <div class="control-group">
                <label class="control-label">Visualization Mode</label>
                <div class="custom-select">
                    <select>
                        <option>Game Tree</option>
                        <option>Step by Step</option>
                        <option>Full Tree</option>
                    </select>
                    <span class="select-arrow">▾</span>
                </div>
            </div>
            <div class="control-group">
                <label class="control-label">Speed</label>
                <div class="speed-control">
                    <input type="range" min="1" max="5" value="3" class="speed-slider" />
                    <span class="speed-label">Medium</span>
                </div>
            </div>
            <button class="reset-btn">↺ Reset</button>
        </div>

        <!-- Visualizer Panel -->
        <div class="visualizer-panel">
            <div class="panel-header">
                <h3 class="panel-title">Minimax Game Tree Visualization <span class="info-icon">ⓘ</span></h3>
                <div class="legend">
                    <span class="legend-item"><span class="dot green"></span> MAX (AI - O)</span>
                    <span class="legend-item"><span class="dot red"></span> MIN (Player - X)</span>
                    <span class="legend-item"><span class="dot blue"></span> Terminal State</span>
                </div>
            </div>

            <!-- Tree Visualization -->
            <div class="tree-container">

                <!-- ROOT: MAX -->
                <div class="tree-level">
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border">
                            <div class="board-grid">
                                <span></span><span>O</span><span></span>
                                <span>X</span><span></span><span>X</span>
                                <span></span><span>X</span><span></span>
                            </div>
                        </div>
                        <div class="score-badge">Score: 0</div>
                    </div>
                </div>

                <!-- Connector lines -->
                <div class="connector-row">
                    <div class="h-line"></div>
                </div>

                <!-- LEVEL 1: MIN nodes -->
                <div class="tree-level">
                    <div class="tree-node-wrapper">
                        <div class="node-label red-label">MIN</div>
                        <div class="board-node red-border">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span></span><span>O</span><span>X</span>
                                <span>X</span><span></span><span></span>
                            </div>
                        </div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label red-label">MIN</div>
                        <div class="board-node red-border">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span>X</span><span>X</span><span>X</span>
                                <span></span><span></span><span>O</span>
                            </div>
                        </div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label red-label">MIN</div>
                        <div class="board-node red-border">
                            <div class="board-grid">
                                <span>H</span><span></span><span></span>
                                <span>X</span><span></span><span>O</span>
                                <span></span><span>X</span><span></span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Connector lines -->
                <div class="connector-row">
                    <div class="h-line"></div>
                </div>

                <!-- LEVEL 2: MAX terminal nodes -->
                <div class="tree-level terminal-level">
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span>X</span><span>O</span><span>X</span>
                                <span></span><span>X</span><span></span>
                            </div>
                        </div>
                        <div class="score-chip positive">+1</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>H</span><span></span><span></span>
                                <span>X</span><span>O</span><span>D</span>
                                <span></span><span>X</span><span>H</span>
                            </div>
                        </div>
                        <div class="score-chip neutral">0</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span></span><span>O</span><span>H</span>
                                <span></span><span>O</span><span>H</span>
                                <span>X</span><span>X</span><span></span>
                            </div>
                        </div>
                        <div class="score-chip negative">-1</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>X</span><span>O</span><span>X</span>
                                <span></span><span>X</span><span></span>
                                <span>O</span><span></span><span></span>
                            </div>
                        </div>
                        <div class="score-chip neutral">0</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span></span><span>O</span><span>X</span>
                                <span>X</span><span>O</span><span></span>
                            </div>
                        </div>
                        <div class="score-chip positive">+1</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span></span><span>O</span><span>X</span>
                                <span>X</span><span></span><span>O</span>
                            </div>
                        </div>
                        <div class="score-chip neutral">0</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>X</span><span></span><span>H</span>
                                <span>X</span><span></span><span>H</span>
                                <span>O</span><span>X</span><span></span>
                            </div>
                        </div>
                        <div class="score-chip negative">-1</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>X</span><span></span><span>O</span>
                                <span>X</span><span></span><span>O</span>
                                <span></span><span>X</span><span></span>
                            </div>
                        </div>
                        <div class="score-chip negative">-1</div>
                    </div>
                    <div class="tree-node-wrapper">
                        <div class="node-label green-label">MAX</div>
                        <div class="board-node green-border small-board">
                            <div class="board-grid">
                                <span>O</span><span></span><span></span>
                                <span>X</span><span>O</span><span>X</span>
                                <span></span><span>X</span><span>O</span>
                            </div>
                        </div>
                        <div class="score-chip positive">+1</div>
                    </div>
                </div>
            </div>

            <!-- Controls -->
            <div class="playback-controls">
                <div class="playback-buttons">
                    <button class="play-btn">▶ Play</button>
                    <button class="pause-btn">⏸ Pause</button>
                    <button class="next-btn">⏭ Next Step</button>
                    <button class="stop-btn">↺ Reset</button>
                </div>
                <div class="step-progress">
                    <span class="step-label">Step Progress</span>
                    <span class="step-count">Step: 4 / 18</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 22%"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom Tabs -->
        <div class="bottom-panel">
            <div class="tab-bar">
                <button class="tab active">🤖 AI Explanation</button>
                <button class="tab">💻 Code View</button>
            </div>
            <div class="tab-content">
                <div class="ai-explanation">
                    <div class="ai-header">
                        <div class="ai-icon">🤖</div>
                        <strong>AI Explanation:</strong>
                    </div>
                    <p class="ai-text">
                        Yo step ma AI (MAX) le sabai possible moves evaluate gardai cha. Player (MIN) le j jawab dinxa,
                        tesko hisab garera AI le best score choose gardai cha.
                    </p>
                    <ul class="score-legend">
                        <li><span class="chip positive">+1</span> → AI wins</li>
                        <li><span class="chip neutral">0</span> → Draw</li>
                        <li><span class="chip negative">-1</span> → Player wins</li>
                    </ul>
                    <p class="ai-text">AI le maximum score dine move choose garcha.</p>
                </div>
            </div>
        </div>

    </div>
    """)