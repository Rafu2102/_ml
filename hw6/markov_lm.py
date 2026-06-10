import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ==========================================
# 0. 載入並預處理資料
# ==========================================
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tw.txt")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"找不到測試資料檔案：{data_path}")

print("正在讀取並解析 tw.txt 語料庫...")
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 將每句拆分為字元列表，並加上前後綴標記
# 使用字元級別做為 Token，但特殊符號 <B> 與 <E> 保持完整
BOS = "<B>"
EOS = "<E>"
sentences = []
for line in lines:
    line_str = line.strip()
    if not line_str:
        continue
    tokens = [BOS, BOS] + list(line_str) + [EOS]
    sentences.append(tokens)

print(f"成功載入 {len(sentences)} 句語料。")

# ==========================================
# 1. 構建 Trigram 特徵與標籤 (X, y)
# ==========================================
# X: 前兩個字的狀態組合字串，例如 "小_貓"
# y: 當前預測的下一個字元
X = []
y = []

for tokens in sentences:
    for i in range(2, len(tokens)):
        c1 = tokens[i - 2]
        c2 = tokens[i - 1]
        c3 = tokens[i]
        X.append(f"{c1}_{c2}")
        y.append(c3)

# 轉換為 numpy 陣列以便進行 scikit-learn 處理
X_arr = np.array(X).reshape(-1, 1)
y_arr = np.array(y)

# 使用 One-Hot Encoder 對合併狀態進行編碼（支援稀疏矩陣提升效能）
encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_arr)

print(f"Trigram 特徵數（相異前綴狀態組合數）：{X_encoded.shape[1]}")
print(f"訓練樣本總數：{len(y)}")

# ==========================================
# 2. 訓練決策樹分類器（等價於極大似然馬可夫鏈）
# ==========================================
print("\n正在使用 DecisionTreeClassifier 訓練馬可夫語言模型...")
# 使用熵 (Entropy) 作為分裂標準，契合資訊理論與語言模型設計
classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
classifier.fit(X_encoded, y_arr)
print("模型訓練完成！")

# ==========================================
# 3. 評估模型效能
# ==========================================
# 計算訓練集上的準確度 (Accuracy)
y_pred = classifier.predict(X_encoded)
accuracy = np.mean(y_pred == y_arr)

# 計算訓練集上的交叉熵損失 (Cross-Entropy Loss)
probs = classifier.predict_proba(X_encoded)
class_to_idx = {c: idx for idx, c in enumerate(classifier.classes_)}
y_indices = np.array([class_to_idx[yi] for yi in y_arr])
true_probs = probs[np.arange(len(y_arr)), y_indices]

# 加入極小常數 (1e-15) 避免 log(0)
loss = -np.mean(np.log(true_probs + 1e-15))

print("\n--- 模型效能指標 ---")
print(f"訓練集準確度 (Accuracy): {accuracy * 100:.2f}%")
print(f"交叉熵損失 (Cross-Entropy Loss): {loss:.4f}")
print(f"語言模型困惑度 (Perplexity): {np.exp(loss):.4f}")
print("--------------------")

# ==========================================
# 4. 文字生成邏輯（支援溫度抽樣）
# ==========================================
def generate_sentence(classifier, encoder, max_len=30, temperature=0.5):
    # 從初始狀態開始
    tokens = [BOS, BOS]
    
    for _ in range(max_len):
        state = f"{tokens[-2]}_{tokens[-1]}"
        state_arr = np.array([[state]])
        state_encoded = encoder.transform(state_arr)
        
        # 預測下一個字的機率分佈
        probs = classifier.predict_proba(state_encoded)[0]
        
        if temperature <= 0.0:
            # 貪婪選擇 (Greedy)：選擇機率最高的字
            next_token = classifier.classes_[np.argmax(probs)]
        else:
            # 溫度加權抽樣
            logits = np.log(probs + 1e-15)
            scaled_logits = logits / temperature
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits)) # 減去最大值防溢位
            scaled_probs = exp_logits / np.sum(exp_logits)
            
            # 從詞彙表中進行多項式抽樣
            next_token = np.random.choice(classifier.classes_, p=scaled_probs)
            
        if next_token == EOS:
            break
        tokens.append(next_token)
        
    return "".join(tokens[2:])

# ==========================================
# 5. 生成測試與互動
# ==========================================
print("\n--- 自動生成句子範例 ---")
for temp in [0.2, 0.5, 0.8]:
    print(f"\n溫度 (Temperature) = {temp} 的生成結果：")
    for i in range(5):
        sentence = generate_sentence(classifier, encoder, temperature=temp)
        print(f"  句子 {i+1}: {sentence}")

print("\n=======================================")
print("互動模式：請輸入前兩個字（中文），馬可夫模型將為您預測下一個字與接續生成。")
print("（輸入 'q' 退出）")
print("=======================================")

while True:
    try:
        user_input = input("\n請輸入前綴兩個字 (例如 '小貓', '今天'): ").strip()
        if user_input == 'q':
            print("退出互動模式。")
            break
        if len(user_input) != 2:
            print("請輸入「剛好兩個字」作為預測前綴！")
            continue
            
        # 驗證輸入字元
        state = f"{user_input[0]}_{user_input[1]}"
        state_arr = np.array([[state]])
        state_encoded = encoder.transform(state_arr)
        
        # 取得下一個字預測
        probs = classifier.predict_proba(state_encoded)[0]
        # 排除 <B>、<E> 或機率太低的字，找出前三個最可能的候選字
        top_indices = np.argsort(probs)[::-1][:3]
        
        print("預測最可能的下一個字：")
        for rank, idx in enumerate(top_indices):
            char = classifier.classes_[idx]
            p = probs[idx]
            if p > 0.001:
                print(f"  候選 {rank+1}: '{char}' (機率: {p*100:.1f}%)")
                
        # 基於前綴自動生成一整句
        print("接續生成的整句話：")
        for i in range(3):
            # 前綴手動載入到 token 歷史中
            tokens = [BOS, user_input[0], user_input[1]]
            sentence = generate_sentence(classifier, encoder, temperature=0.4)
            # 將前綴與生成的後半段拼接
            full_sentence = user_input + sentence
            print(f"  生成 {i+1}: {full_sentence}")
            
    except (KeyboardInterrupt, EOFError):
        print("\n退出互動模式。")
        break
