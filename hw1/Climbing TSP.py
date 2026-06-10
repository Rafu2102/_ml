import math
import random

# ==============================================================================
# 0. 環境設定：產生 8 個地點的座標與距離矩陣
# ==============================================================================
num_places = 8
random.seed(42)  # 固定隨機種子，讓每次執行的結果相同

# 隨機產生 8 個地點的 (x, y) 座標 (範圍 0~100)
coordinates = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(1, num_places + 1)}

# 計算兩點間的歐幾里得距離
def get_distance(p1, p2):
    x1, y1 = coordinates[p1]
    x2, y2 = coordinates[p2]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ==============================================================================
# 1. 核心函數設計 (完全符合你的題目要求)
# ==============================================================================

# 【初始解】：1 => 2 => 3 => ... => n => 1
def get_initial_solution(n):
    # 陣列儲存：[1, 2, 3, 4, 5, 6, 7, 8]，最後算距離時再連回起點
    return list(range(1, n + 1))

# 【Height 函數】：總距離 * -1
def get_height(solution):
    total_distance = 0
    # 計算相鄰兩點的距離
    for i in range(len(solution)):
        p1 = solution[i]
        p2 = solution[(i + 1) % len(solution)]  # 當 i 是最後一個時，回到起點
        total_distance += get_distance(p1, p2)
    return total_distance * -1

# 【Neighbor 函數】：斷開 (a,b) 與 (c,d)，改連 (a,d) 與 (b,c)
def get_neighbors(solution):
    neighbors = []
    n = len(solution)
    
    # 雙層迴圈挑選兩個不相鄰的邊進行 2-opt 翻轉
    for i in range(n):
        for j in range(i + 2, n):
            # 避免挑到頭尾相連的邊（那樣翻轉沒有意義）
            if i == 0 and j == n - 1:
                continue
                
            # 建立鄰居：保持 i 之前不變，將 i+1 到 j 之間的路段反轉，j 之後不變
            # 這等同於將邊 (i, i+1) 和 (j, j+1) 斷開，重新交叉相連
            neighbor = solution[:i+1] + solution[i+1:j+1][::-1] + solution[j+1:]
            neighbors.append(neighbor)
            
    return neighbors

# ==============================================================================
# 2. 爬山演算法主程式
# ==============================================================================
def hill_climbing_tsp():
    # 取得初始解
    current_solution = get_initial_solution(num_places)
    current_height = get_height(current_solution)
    
    print(f"初始解: {current_solution} -> 1")
    print(f"初始高度 (距離 * -1): {current_height:.2f} (實際距離: {-current_height:.2f})\n")
    
    step = 1
    while True:
        # 找出目前解的所有鄰居
        neighbors = get_neighbors(current_solution)
        
        # 評估所有鄰居的高度，找出「最高」的那個鄰居
        best_neighbor = None
        best_neighbor_height = float('-inf')
        
        for neighbor in neighbors:
            h = get_height(neighbor)
            if h > best_neighbor_height:
                best_neighbor_height = h
                best_neighbor = neighbor
        
        # 爬山法核心邏輯：如果最棒的鄰居比現在的位置還要高，就往上爬
        if best_neighbor_height > current_height:
            current_solution = best_neighbor
            current_height = best_neighbor_height
            print(f"第 {step} 步移動到更佳解: {current_solution} -> 1 | 高度: {current_height:.2f} (實際距離: {-current_height:.2f})")
            step += 1
        else:
            # 四週的鄰居都比目前低（或一樣低），代表到山頂（區域最佳解）了，停止搜尋
            print("\n[到達山頂] 四週已無更高的地方。")
            break
            
    return current_solution, current_height

# ==============================================================================
# 執行與輸出結果
# ==============================================================================
print("--- 地點座標資訊 ---")
for k, v in coordinates.items():
    print(f"地點 {k}: {v}")
print("-------------------\n")

best_route, max_height = hill_climbing_tsp()

print("\n--- 最終搜尋結果 ---")
print(f"最佳路線: {' -> '.join(map(str, best_route))} -> 1")
print(f"最短總距離: {-max_height:.2f}")