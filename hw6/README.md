# 馬可夫語言模型 - scikit-learn 實作版

本專案使用 **scikit-learn** 的機器學習分類器，實作了一個經典的 **二階馬可夫鏈（2nd-order Markov Chain）** 語言模型（又稱 Trigram 語言模型）。

本專案實作於 [markov_lm.py](./markov_lm.py)，並使用 [tw.txt](./tw.txt) 作為訓練與測試語料。

---

## 數學與演算法原理

在二階馬可夫模型中，我們假設下一個字出現的機率只取決於**前兩個字**的狀態：

$$P(c_i \mid c_1, c_2, \dots, c_{i-1}) = P(c_i \mid c_{i-2}, c_{i-1})$$

為了解決這個最佳化問題，本實作將其轉化為一個**機器學習分類任務**：
1. **狀態特徵 ($X$)**：前兩個字組成的狀態字串，例如 `"小_貓"`。我們使用 `OneHotEncoder` 對其進行獨熱編碼，產生高度優化的稀疏矩陣。
2. **預測目標 ($y$)**：緊接在狀態後的第三個字，例如 `'坐'`。
3. **分類器等價性**：我們採用了 **決策樹分類器（`DecisionTreeClassifier`）**。當特徵為離散的 One-Hot 狀態時，決策樹的葉節點會精確分離每個相異的狀態。其輸出的預測機率分佈（`predict_proba`）在數學上與最大概似估計（MLE）的轉移機率完全一致：

$$\text{Probability}(c_3 \mid c_1, c_2) = \frac{\text{Count}(c_1, c_2, c_3)}{\text{Count}(c_1, c_2)}$$

---

## 核心功能與程式碼連結

您可以透過以下連結跳轉至 [markov_lm.py](./markov_lm.py) 的核心實作：

* **[資料預處理](./markov_lm.py#L7-L28)**：讀取語料庫，對每句話包裝 `BOS`（開頭）與 `EOS`（結尾）標記。
* **[特徵提取與 One-Hot 編碼](./markov_lm.py#L30-L50)**：將 Trigram 狀態轉換為機器學習特徵。
* **[分類器訓練](./markov_lm.py#L52-L60)**：配置決策樹進行模型擬合。
* **[模型指標評估](./markov_lm.py#L62-L80)**：計算訓練集上的交叉熵損失（Cross-Entropy Loss）、準確度（Accuracy）與困惑度（Perplexity）。
* **[隨機溫度抽樣生成](./markov_lm.py#L82-L113)**：支援透過設定溫度（Temperature）來平衡生成句子的多樣性與真實性。

---

## 如何執行

請確保您的電腦已安裝 scikit-learn 與 numpy。在終端機中切換至本目錄，並執行以下指令：

```powershell
python markov_lm.py
```

訓練完成後，程式會印出效能指標，並自動進入**互動模式**，您可以輸入任意兩個中文漢字，模型會為您預測下一個最可能的字，並自動接續生成完整的句子！
