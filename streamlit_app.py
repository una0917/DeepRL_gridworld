import streamlit as st
import numpy as np
import random
from typing import List, Dict, Tuple

# ==================== 配置 ====================
st.set_page_config(
    page_title="RL Grid World",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== 常數定義 ====================
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}

GAMMA = 0.9
REWARD_STEP = -1
REWARD_GOAL = 10
THETA = 1e-4

# ==================== 輔助函數 ====================
def is_valid_state(r: int, c: int, n: int, obstacles: List[List[int]]) -> bool:
    """檢查狀態是否在網格內並且不是障礙物"""
    return 0 <= r < n and 0 <= c < n and [r, c] not in obstacles

def compute_value_iteration(n: int, goal: List[int], obstacles: List[List[int]]) -> Tuple[Dict, Dict]:
    """執行值迭代算法"""
    V = {(r, c): 0.0 for r in range(n) for c in range(n)}
    policy = {(r, c): [] for r in range(n) for c in range(n)}
    
    while True:
        delta = 0
        new_V = V.copy()
        
        for r in range(n):
            for c in range(n):
                if [r, c] == goal or [r, c] in obstacles:
                    continue
                
                max_v = -float('inf')
                best_actions = []
                
                for a in ACTIONS:
                    dr, dc = ACTION_DELTAS[a]
                    nr, nc = r + dr, c + dc
                    
                    if is_valid_state(nr, nc, n, obstacles):
                        reward = REWARD_GOAL if [nr, nc] == goal else REWARD_STEP
                        v_val = reward + GAMMA * V[(nr, nc)]
                        
                        if v_val > max_v + 1e-6:
                            max_v = v_val
                            best_actions = [a]
                        elif abs(v_val - max_v) <= 1e-6:
                            best_actions.append(a)
                
                if max_v == -float('inf'):
                    max_v = 0
                    best_actions = []
                
                delta = max(delta, abs(max_v - V[(r, c)]))
                new_V[(r, c)] = max_v
                policy[(r, c)] = best_actions
        
        V = new_V
        if delta < THETA:
            break
    
    return V, policy

def find_optimal_path(start: List[int], goal: List[int], policy: Dict, n: int, obstacles: List[List[int]]) -> List[List[int]]:
    """找出最優路徑"""
    path = []
    if start and goal:
        curr = tuple(start)
        visited = set()
        
        while curr != tuple(goal):
            if curr in visited or list(curr) in obstacles:
                break
            visited.add(curr)
            path.append(list(curr))
            
            if not policy[curr]:
                break
            
            a = random.choice(policy[curr])
            dr, dc = ACTION_DELTAS[a]
            nr, nc = curr[0] + dr, curr[1] + dc
            
            if is_valid_state(nr, nc, n, obstacles):
                curr = (nr, nc)
            else:
                break
        
        if curr == tuple(goal):
            path.append(goal)
    
    return path

# ==================== Streamlit UI ====================
st.title("🤖 Grid World - 強化學習環境")
st.markdown("---")

# 左側：設定面板
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ 環境設定")
    
    grid_size = st.slider("網格大小", min_value=5, max_value=9, value=5, step=1)
    
    st.markdown("**設置環境：**")
    mode = st.radio("選擇模式", ["設置起點", "設置目標", "設置障礙物"], horizontal=True)
    
    # 網格選擇區域
    st.markdown("**點擊選擇位置：**")
    
    # 創建網格按鈕
    grid_cols = st.columns(grid_size)
    selected_pos = None
    
    for r in range(grid_size):
        for c in range(grid_size):
            with grid_cols[c]:
                if st.button(f"{r},{c}", key=f"grid_{r}_{c}", use_container_width=True):
                    selected_pos = [r, c]
    
    # 保存選擇到 session state
    if selected_pos:
        if mode == "設置起點":
            st.session_state.start = selected_pos
        elif mode == "設置目標":
            st.session_state.goal = selected_pos
        else:
            if 'obstacles' not in st.session_state:
                st.session_state.obstacles = []
            if selected_pos not in st.session_state.obstacles:
                st.session_state.obstacles.append(selected_pos)
            else:
                st.session_state.obstacles.remove(selected_pos)
    
    # 初始化 session state
    if 'start' not in st.session_state:
        st.session_state.start = None
    if 'goal' not in st.session_state:
        st.session_state.goal = None
    if 'obstacles' not in st.session_state:
        st.session_state.obstacles = []
    
    # 顯示當前設置
    st.markdown("**當前設置：**")
    if st.session_state.start:
        st.success(f"✓ 起點: {st.session_state.start}")
    else:
        st.warning("❌ 未設置起點")
    
    if st.session_state.goal:
        st.success(f"✓ 目標: {st.session_state.goal}")
    else:
        st.warning("❌ 未設置目標")
    
    if st.session_state.obstacles:
        st.info(f"✓ 障礙物數量: {len(st.session_state.obstacles)}")
    
    st.markdown("---")
    
    # 執行算法按鈕
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        run_policy_eval = st.button("📊 策略評估", use_container_width=True)
    
    with col_btn2:
        run_value_iter = st.button("🎯 值迭代", use_container_width=True)
    
    # 清除設置按鈕
    if st.button("🔄 重置所有", use_container_width=True):
        st.session_state.start = None
        st.session_state.goal = None
        st.session_state.obstacles = []
        st.session_state.V = None
        st.session_state.policy = None
        st.session_state.path = []
        st.rerun()

# 右側：結果展示
with col2:
    if run_policy_eval and st.session_state.goal:
        st.session_state.V, st.session_state.policy = compute_value_iteration(
            grid_size, st.session_state.goal, st.session_state.obstacles
        )
        st.session_state.path = []
    
    if run_value_iter and st.session_state.start and st.session_state.goal:
        st.session_state.V, st.session_state.policy = compute_value_iteration(
            grid_size, st.session_state.goal, st.session_state.obstacles
        )
        st.session_state.path = find_optimal_path(
            st.session_state.start, st.session_state.goal, 
            st.session_state.policy, grid_size, st.session_state.obstacles
        )
    
    # 結果展示
    if 'V' in st.session_state and st.session_state.V:
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.subheader("📈 價值矩陣 (Value Matrix)")
            
            # 生成價值矩陣展示
            value_display = []
            for r in range(grid_size):
                row = []
                for c in range(grid_size):
                    if [r, c] in st.session_state.obstacles:
                        row.append("🚫")
                    elif [r, c] == st.session_state.goal:
                        row.append("🎯")
                    elif [r, c] == st.session_state.start:
                        row.append("🟢")
                    else:
                        v = st.session_state.V.get((r, c), 0)
                        row.append(f"{v:.2f}")
                value_display.append(row)
            
            # 用表格展示
            st.write("| " + " | ".join(f"{c}" for c in range(grid_size)) + " |")
            st.write("|" + "|".join([":-:"] * grid_size) + "|")
            for r, row in enumerate(value_display):
                st.write("| " + " | ".join(str(v) for v in row) + " |")
        
        with result_col2:
            st.subheader("🧭 策略矩陣 (Policy Matrix)")
            
            # 生成策略矩陣展示
            policy_display = []
            for r in range(grid_size):
                row = []
                for c in range(grid_size):
                    if [r, c] in st.session_state.obstacles:
                        row.append("🚫")
                    elif [r, c] == st.session_state.goal:
                        row.append("🎯")
                    else:
                        actions = st.session_state.policy.get((r, c), [])
                        arrow_str = "".join([ARROWS[a] for a in actions]) if actions else "—"
                        row.append(arrow_str)
                policy_display.append(row)
            
            # 用表格展示
            st.write("| " + " | ".join(f"{c}" for c in range(grid_size)) + " |")
            st.write("|" + "|".join([":-:"] * grid_size) + "|")
            for r, row in enumerate(policy_display):
                st.write("| " + " | ".join(str(v) for v in row) + " |")
        
        # 最優路徑展示
        if 'path' in st.session_state and st.session_state.path:
            st.subheader("📍 最優路徑")
            st.success(f"路徑長度: {len(st.session_state.path)} 步")
            
            # 生成路徑矩陣
            path_display = []
            for r in range(grid_size):
                row = []
                for c in range(grid_size):
                    if [r, c] in st.session_state.obstacles:
                        row.append("🚫")
                    elif [r, c] == st.session_state.goal:
                        row.append("🎯 (終點)")
                    elif [r, c] == st.session_state.start:
                        row.append("🟢 (起點)")
                    elif [r, c] in st.session_state.path:
                        path_idx = st.session_state.path.index([r, c])
                        row.append(f"⭐ {path_idx}")
                    else:
                        row.append("·")
                path_display.append(row)
            
            # 用表格展示
            st.write("| " + " | ".join(f"{c}" for c in range(grid_size)) + " |")
            st.write("|" + "|".join([":-:"] * grid_size) + "|")
            for r, row in enumerate(path_display):
                st.write("| " + " | ".join(str(v) for v in row) + " |")
            
            # 顯示路徑詳情
            path_str = " → ".join([f"({p[0]},{p[1]})" for p in st.session_state.path])
            st.info(f"路徑: {path_str}")

st.markdown("---")
st.markdown("""
### 📖 使用說明
1. **調整網格大小** (5-9)
2. **選擇模式** 並點擊網格設置環境
3. **點擊按鈕** 執行算法並查看結果

### ⚙️ 算法參數
- **折扣因子 (γ)**: 0.9
- **移動獎勵**: -1
- **目標獎勵**: +10
- **收斂閾值 (θ)**: 1e-4
""")
