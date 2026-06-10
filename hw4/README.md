# 高性能 GPT - PyTorch 實作版

本專案是針對 Andrej Karpathy 釋出的 [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) 進行研究，並使用 **PyTorch** 重新實作的高效能、高效率 GPT 模型。

本專案實作於 [gpt.py](./gpt.py)。它保留了 `microgpt.py` 的極簡架構設計，同時透過張量平行化運算與 GPU（CUDA）加速，實現了數千倍的訓練速度提升。

---

## 專案特點與優化

1. **GPU 加速與自動偵測**：自動偵測系統是否支援 CUDA，並啟用 GPU 進行加速運算。
2. **批次訓練（Batching）**：使用自訂的 [NameDataset](./gpt.py#L30-L49) 與 PyTorch 的 `DataLoader` 進行 Batch 訓練，並透過 `ignore_index` 排除 `PAD` 填充符號的損失計算，使訓練更加穩定與高效。
3. **保持架構純粹性**：精準復刻 `microgpt` 的經典設計：
   * 使用無權重的 [RMSNorm](./gpt.py#L52-L59)（取代 LayerNorm）。
   * 線性層與自注意力層皆無偏置（No Biases）。
   * 激活函數採用 ReLU。
4. **平行序列計算**：在 [CausalSelfAttention](./gpt.py#L61-L86) 中使用下三角遮罩（Causal Mask），使得訓練時能夠平行處理整個序列，而非像原版那樣逐字前向傳播。
5. **互動式前綴生成（Prompt Completion）**：新增 [generate_with_prompt](./gpt.py#L230-L255) 邏輯。使用者可輸入名字前綴字母，模型會以此提示詞（Prompt）為基礎接續生成，達到姓名自動完成的效果。

---

## 核心模組架構

* **[MicroGPT](./gpt.py#L102-L132)**：模型主體，包含 Token Embedding、Position Embedding、多個 Transformer 解碼層與最後的預設線性層。
* **[TransformerBlock](./gpt.py#L88-L100)**：包含自注意力機制與前饋網路（MLP），每層均有殘差連接（Residual Connections）。
* **[CausalSelfAttention](./gpt.py#L61-L86)**：多頭因果自注意力機制。

---

## 如何執行

請確保您的電腦已安裝 PyTorch。在終端機中執行以下指令即可啟動訓練：

```powershell
python gpt.py
```

訓練結束後，程式會自動進入**互動模式**，您可以輸入前綴英文字母進行測試。

---

## 執行結果預期

由於使用了 PyTorch 與批次化訓練，模型僅需數秒即可在 `names.txt` 數據集上完成 15 個 Epoch 的訓練，並生成極具統計真實感的英文名字，接著進入提示詞互動：

```text
num docs: 32033
vocab size: 28
開始訓練神經網路...
---------------------------------------
Epoch 01/15 | Average Loss: 2.5833
Epoch 05/15 | Average Loss: 2.1154
Epoch 10/15 | Average Loss: 2.0521
Epoch 15/15 | Average Loss: 2.0229
---------------------------------------
訓練完成！開始生成測試：

--- inference (new, hallucinated names) ---
sample  1: mariah
sample  2: kaylee
sample  3: jaxon
sample  4: aliah
...

=======================================
互動模式：請輸入名字的前幾個字母（英文），GPT 將為您續寫完成名字。
（輸入 'q' 退出）
=======================================

請輸入前綴字母 (例如 'an', 'ka', 'ma'): ka
候選名字生成中...
  選項 1: kamon
  選項 2: katviah
  選項 3: karai
  選項 4: kaina
  選項 5: karia
```
