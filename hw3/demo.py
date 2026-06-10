import math
import random
from nn0 import Value, Adam, linear

# ==========================================
# 1. 產生模擬數據：圓形邊界分類
# ==========================================
# 任務：判斷二維平面上的點 (x1, x2) 是否在半徑為 0.7 的圓內
# 如果 x1^2 + x2^2 < 0.49，標籤為 0；否則為 1
def generate_data(num_samples=100):
    dataset = []
    for _ in range(num_samples):
        # 隨機產生 [-1, 1] 之間的點
        x1 = random.uniform(-1, 1)
        x2 = random.uniform(-1, 1)
        # 目標標籤 (Target)
        target = 0 if (x1**2 + x2**2) < 0.49 else 1
        dataset.append(([x1, x2], target))
    return dataset

# ==========================================
# 2. 定義神經網路模型
# ==========================================
class SimpleNet:
    """一個簡單的單隱藏層神經網路"""
    def __init__(self):
        # 初始化權重與偏差，使用 Value 包裝以便自動微分
        # 輸入層 (2) -> 隱藏層 (4)
        self.W1 = [[Value(random.uniform(-0.5, 0.5)) for _ in range(2)] for _ in range(4)]
        self.b1 = [Value(0.0) for _ in range(4)]
        
        # 隱藏層 (4) -> 輸出層 (1)
        self.W2 = [[Value(random.uniform(-0.5, 0.5)) for _ in range(4)] for _ in range(1)]
        self.b2 = [Value(0.0) for _ in range(1)]

    def forward(self, x):
        # x 為 [x1, x2] 的純數字列表，先轉成 Value 節點
        x_val = [Value(xi) for xi in x]
        
        # 第一層：h = relu(W1 @ x + b1)
        h = linear(x_val, self.W1)
        h = [hi + bi for hi, bi in zip(h, self.b1)]
        h_act = [hi.relu() for hi in h]
        
        # 第二層：out = W2 @ h_act + b2
        out = linear(h_act, self.W2)
        out = [oi + bi for oi, bi in zip(out, self.b2)]
        
        # 因為只有一個輸出節點，直接返回該 Value
        return out[0]

    def get_parameters(self):
        # 攤平所有參數，以便傳給 Adam 優化器
        params = []
        for row in self.W1: params.extend(row)
        params.extend(self.b1)
        for row in self.W2: params.extend(row)
        params.extend(self.b2)
        return params

# ==========================================
# 3. 損失函數與主訓練流程
# ==========================================
def main():
    # 設定隨機種子以確保結果可重現
    random.seed(1)
    
    # 建立數據與模型
    train_data = generate_data(120)
    model = SimpleNet()
    
    # 獲取參數並配置 Adam 優化器
    params = model.get_parameters()
    optimizer = Adam(params, lr=0.05)
    
    epochs = 40
    batch_size = 10
    
    print("開始訓練神經網路...")
    print("---------------------------------------")
    
    for epoch in range(epochs):
        # 每個 epoch 打亂數據
        random.shuffle(train_data)
        
        epoch_loss = 0.0
        # 小批次 (Mini-batch) 訓練
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # 用來累積這個 batch 的總損失
            batch_loss = Value(0.0)
            
            for x, target in batch:
                # 前向傳播得到預測值 (Logit)
                pred = model.forward(x)
                
                # 使用 Sigmoid 激活函數的數學變形計算二元交叉熵損失 (Binary Cross Entropy)
                # 這裡為了避免你的 Value 沒寫 sigmoid，直接用數學式表達：
                # loss = -[target * log(p) + (1 - target) * log(1 - p)]
                # 數值穩定的 Softplus 寫法或直覺寫法如下：
                if target == 1:
                    # loss = -log(sigmoid(pred)) = log(1 + exp(-pred))
                    # 為了簡化，直接用你的 .exp() 與 .log()
                    # 這裡採用逼近安全值的方式：
                    loss_i = (Value(1.0) + (-pred).exp()).log()
                else:
                    # loss = -log(1 - sigmoid(pred)) = log(1 + exp(pred))
                    loss_i = (Value(1.0) + pred.exp()).log()
                
                batch_loss = batch_loss + loss_i
            
            # 計算平均損失
            batch_loss = batch_loss / len(batch)
            epoch_loss += batch_loss.data
            
            # 反向傳播與優化
            batch_loss.backward()
            optimizer.step()
            
        # 每個 Epoch 印出一次平均損失
        avg_loss = epoch_loss / (len(train_data) / batch_size)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Average Loss: {avg_loss:.4f}")

    print("---------------------------------------")
    print("訓練完成！開始測試模型預測能力：")
    
    # ==========================================
    # 4. 模型測試與驗證
    # ==========================================
    test_data = generate_data(10)
    correct = 0
    
    for x, target in test_data:
        pred_value = model.forward(x)
        # 通過 Sigmoid 判定機率是否大於 0.5，等同於 pred_value.data 是否 > 0
        prediction = 1 if pred_value.data > 0 else 0
        
        status = "✅" if prediction == target else "❌"
        if prediction == target: correct += 1
        
        # 計算點到原點的距離，方便視覺化理解
        dist = math.sqrt(x[0]**2 + x[1]**2)
        print(f"輸入座標: ({x[0]:.2f}, {x[1]:.2f}) | 離原點距離: {dist:.2f} | 目標: {target} | 預測: {prediction} {status}")
        
    print(f"\n測試準確度 (Accuracy): {correct / len(test_data) * 100:.1f}%")

if __name__ == "__main__":
    main()