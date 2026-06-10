import os
import urllib.request
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 0. 環境與數據設定
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下載名字數據集
if not os.path.exists('input.txt'):
    print("Downloading names dataset...")
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.seed(42)
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer 設定
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)       # 起始/結束 Token
PAD = len(uchars) + 1   # 填充 Token
vocab_size = len(uchars) + 2 # 總 Vocabulary 大小
print(f"vocab size: {vocab_size}")

# ==========================================
# 1. Dataset 與 DataLoader 實作（批次處理）
# ==========================================
class NameDataset(Dataset):
    def __init__(self, docs, uchars, block_size):
        self.docs = docs
        self.uchars = uchars
        self.block_size = block_size
        
    def __len__(self):
        return len(self.docs)
        
    def __getitem__(self, idx):
        doc = self.docs[idx]
        # 包裹 BOS
        tokens = [BOS] + [self.uchars.index(ch) for ch in doc] + [BOS]
        
        # 填充或截斷至長度為 block_size + 1
        if len(tokens) < self.block_size + 1:
            tokens = tokens + [PAD] * (self.block_size + 1 - len(tokens))
        else:
            tokens = tokens[:self.block_size + 1]
            
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y

# ==========================================
# 2. 定義 PyTorch 版 GPT 模組（比照 microgpt 架構）
# ==========================================
class RMSNorm(nn.Module):
    """自訂 RMSNorm 模組（無 learnable weights，與 microgpt 一致）"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

class CausalSelfAttention(nn.Module):
    """因果自注意力機制 (Causal Self-Attention)"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)
        
        # 因果遮罩 (Causal Mask)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class TransformerBlock(nn.Module):
    """單個 Transformer 解碼層"""
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attn_norm = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.mlp_norm = RMSNorm(n_embd)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp_fc2(F.relu(self.mlp_fc1(self.mlp_norm(x))))
        return x

class MicroGPT(nn.Module):
    """完整 GPT 模型"""
    def __init__(self, vocab_size, n_layer=1, n_embd=16, n_head=4, block_size=16):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.init_norm = RMSNorm(n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.init_norm(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # 忽略 PAD Token 的損失計算
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=PAD)
            
        return logits, loss

# ==========================================
# 3. 主訓練流程
# ==========================================
def main():
    # 模型超參數（與 microgpt.py 相同以利對照）
    n_layer = 1
    n_embd = 16
    block_size = 16
    n_head = 4
    
    # 訓練參數
    batch_size = 256
    epochs = 15
    lr = 0.005
    
    # 初始化 Dataset 與 DataLoader
    dataset = NameDataset(docs, uchars, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 初始化模型與優化器
    model = MicroGPT(vocab_size, n_layer, n_embd, n_head, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    print("開始訓練神經網路...")
    print("---------------------------------------")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Average Loss: {avg_loss:.4f}")
        
    print("---------------------------------------")
    print("訓練完成！開始生成測試：")
    
    # 進行推理測試
    generate(model, uchars, device, num_samples=10, temperature=0.7)
    
    # 互動式對話（提示詞續寫名字）
    print("\n=======================================")
    print("互動模式：請輸入名字的前幾個字母（英文），GPT 將為您續寫完成名字。")
    print("（輸入 'q' 退出）")
    print("=======================================")
    while True:
        try:
            user_input = input("\n請輸入前綴字母 (例如 'an', 'ka', 'ma'): ").strip().lower()
            if user_input == 'q':
                print("退出互動模式。")
                break
            if not user_input:
                continue
                
            print(f"候選名字生成中...")
            for i in range(5):
                name = generate_with_prompt(model, uchars, device, user_input, temperature=0.7)
                print(f"  選項 {i+1}: {name}")
        except (KeyboardInterrupt, EOFError):
            print("\n退出互動模式。")
            break

# ==========================================
# 4. 推理生成邏輯 (Inference)
# ==========================================
@torch.no_grad()
def generate(model, uchars, device, num_samples=20, temperature=0.7):
    model.eval()
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(num_samples):
        tokens = [BOS]
        for _ in range(model.block_size):
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _ = model(idx)
            
            # 取得最後一個時間步的預測機率並套用 Temperature
            logits = logits[0, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            # 取樣下一個 Token
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == BOS or next_token == PAD:
                break
            tokens.append(next_token)
            
        # 解碼印出生成的姓名
        name = ''.join([uchars[t] for t in tokens[1:]])
        print(f"sample {sample_idx+1:2d}: {name}")

@torch.no_grad()
def generate_with_prompt(model, uchars, device, prompt, temperature=0.7):
    """給定前綴字母（Prompt），由模型接續生成剩餘字元"""
    model.eval()
    tokens = [BOS]
    for ch in prompt:
        if ch in uchars:
            tokens.append(uchars.index(ch))
            
    # 開始從前綴接續生成
    for _ in range(model.block_size - len(tokens)):
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, _ = model(idx)
        
        logits = logits[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == BOS or next_token == PAD:
            break
        tokens.append(next_token)
        
    # 還原成文字
    name = ''.join([uchars[t] for t in tokens[1:]])
    return name

if __name__ == "__main__":
    main()
