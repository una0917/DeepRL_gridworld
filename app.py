from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTAS = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
ARROWS = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→'}

GAMMA = 0.9
REWARD_STEP = -1
REWARD_GOAL = 10
THETA = 1e-4

def is_valid_state(r, c, n, obstacles):
    """檢查狀態是否在網格內並且不是障礙物。"""
    return 0 <= r < n and 0 <= c < n and [r, c] not in obstacles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hw1_2', methods=['POST'])
def hw1_2_policy_eval():
    """按鈕 2：產生最佳價值與政策矩陣（不包含路徑）"""
    data = request.json
    n = data['n']
    goal = data['goal']
    obstacles = data['obstacles']

    V = { (r, c): 0.0 for r in range(n) for c in range(n) }
    policy = { (r, c): [] for r in range(n) for c in range(n) }

    # 價值迭代算法
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

    res_V = [[round(V[(r, c)], 2) if [r, c] not in obstacles else "" for c in range(n)] for r in range(n)]
    res_P = [["".join([ARROWS[a] for a in policy[(r, c)]]) if [r, c] not in obstacles and [r, c] != goal else "" for c in range(n)] for r in range(n)]
    res_V[goal[0]][goal[1]] = 0.0 
    
    # 僅回傳矩陣，不回傳路徑
    return jsonify({'V': res_V, 'policy': res_P, 'path': []})

@app.route('/hw1_3', methods=['POST'])
def hw1_3_value_iteration():
    """按鈕 3：產生最佳價值與政策矩陣，並推導最佳路徑"""
    data = request.json
    n = data['n']
    start = data['start']
    goal = data['goal']
    obstacles = data['obstacles']

    V = { (r, c): 0.0 for r in range(n) for c in range(n) }
    policy = { (r, c): [] for r in range(n) for c in range(n) }

    # 1. 價值迭代算法
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

    # 2. 推導最佳路徑
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
            curr = (curr[0] + dr, curr[1] + dc)
        
        if curr == tuple(goal):
            path.append(goal)

    res_V = [[round(V[(r, c)], 2) if [r, c] not in obstacles else "" for c in range(n)] for r in range(n)]
    res_P = [["".join([ARROWS[a] for a in policy[(r, c)]]) if [r, c] not in obstacles and [r, c] != goal else "" for c in range(n)] for r in range(n)]
    res_V[goal[0]][goal[1]] = 0.0 

    return jsonify({'V': res_V, 'policy': res_P, 'path': path})

if __name__ == '__main__':
    app.run(debug=True)