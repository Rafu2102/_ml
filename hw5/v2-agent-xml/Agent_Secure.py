#!/usr/bin/env python3
# Agent_Secure.py - AI Agent with strict security control
# Run: python Agent_Secure.py

import subprocess
import os
import asyncio
import aiohttp
import re
import glob

# ─── Configuration & Security Setup ───

# 獲取程式所在的實體目錄（防護根目錄）
ALLOWED_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() or '__file__' in globals() else os.getcwd()
ALLOWED_DIR = os.path.realpath(ALLOWED_DIR)

# 將工作區限制在程式資料夾內部的 .agent0 目錄下，避免寫入外部檔案
WORKSPACE = os.path.join(ALLOWED_DIR, ".agent0")
MODEL = "minimax-m2.5:cloud"
MAX_TURNS = 5

# 限制的敏感工具/解譯器黑名單（防止代碼注入與靜態混淆）
SUSPICIOUS_KEYWORDS = {
    # 殼層與語言解譯器
    'python', 'python3', 'py', 'node', 'deno', 'ruby', 'perl', 'php', 'bash', 'sh', 'zsh', 'fish',
    'powershell', 'pwsh', 'cmd', 'wscript', 'cscript', 'mshta', 'rundll32', 'reg', 'regedit',
    'cmd.exe', 'powershell.exe', 'bash.exe', 'sh.exe',
    # 編碼與解碼工具（防止二進位/路徑混淆規避）
    'base64', 'certutil', 'openssl',
    # 指令執行與指令碼載入輔助工具
    'xargs', 'eval', 'exec', 'source', 'run', 'start',
    # 變數定義與環境設定工具（防止執行期動態變數宣告）
    'set', 'export', 'alias', 'env', 'setenv',
    # 套件管理器與遠端下載器（防止下載並執行惡意程式碼）
    'pip', 'pip3', 'npm', 'yarn', 'conda', 'curl', 'wget',
    # 系統設定與診斷工具
    'net', 'netsh', 'sc', 'taskkill', 'tasklist'
}

# ─── Security Helpers ───

def is_inside(path, parent_dir):
    """判斷路徑是否安全地被包在 parent_dir 內部（含親資料夾本身）"""
    path = os.path.realpath(path)
    parent_dir = os.path.realpath(parent_dir)
    
    # 在 Windows 系統上，路徑是不區分大小寫的
    if os.name == 'nt':
        path = path.lower()
        parent_dir = parent_dir.lower()
        
    # 精確比對路徑：必須等於父目錄，或是以父目錄路徑加上分隔符號為開頭
    # 這能有效防止 "v2-agent-xml-alternative" 對 "v2-agent-xml" 的路徑比對越界漏洞
    return path == parent_dir or path.startswith(parent_dir + os.sep)

def check_command_safety(cmd, allowed_dir):
    """
    對命令字串進行全面靜態分析與安全性檢查。
    若安全，回傳 (True, "")；若有風險，回傳 (False, 具體原因)。
    """
    allowed_dir = os.path.realpath(allowed_dir)
    
    # 1. 擷取並檢查指令中是否存在未解析的動態或環境變數（防範執行期變數繞過）
    # 擷取 %VAR% 與 $VAR / ${VAR}
    cmd_vars = re.findall(r'%([A-Za-z0-9_]+)%', cmd)
    bash_vars = re.findall(r'\$(?:\{([A-Za-z0-9_:]+)\}|([A-Za-z0-9_:]+))', cmd)
    var_names = cmd_vars + [v[0] or v[1] for v in bash_vars]
    
    for v in var_names:
        v_lookup = v[4:] if v.startswith('env:') else v
        if v_lookup not in os.environ:
            return False, f"偵測到未解析的動態或環境變數：'{v}'"
            
    # 2. 在完整的指令字串上進行變數展開
    expanded_cmd = os.path.expandvars(cmd)
    def replace_env(match):
        var_name = match.group(1) or match.group(2)
        if var_name.startswith('env:'):
            var_name = var_name[4:]
        return os.environ.get(var_name, '')
    expanded_cmd = re.sub(r'\$(?:\{([A-Za-z0-9_:]+)\}|([A-Za-z0-9_:]+))', replace_env, expanded_cmd)
    
    # 3. 將所有的 shell 運算子替換為空格以進行分詞
    # 運算子包括：&&, ||, ;, |, &, >, <, `, $, (, ), \n, \r
    normalized_cmd = re.sub(r'(&&|\|\||[;&|><`$()\n\r])', ' ', expanded_cmd)
    
    import shlex
    try:
        # 使用 posix=False 保留 Windows 反斜線，並正確處理帶空格的雙引號路徑
        tokens = shlex.split(normalized_cmd, posix=False)
    except ValueError:
        return False, "命令語法解析失敗（可能包含未閉合的引號）"
    
    for t in tokens:
        # 去除引號與括號包裹
        t_clean = t.strip('\'"`()')
        if not t_clean:
            continue
            
        # 4. 檢查是否呼叫了黑名單工具
        base_name = os.path.basename(t_clean).lower()
        if base_name.endswith('.exe'):
            base_name = base_name[:-4]
        if base_name in SUSPICIOUS_KEYWORDS:
            return False, f"嘗試執行限制的工具或指令解譯器：'{base_name}'"
            
        # 5. 檢查通配符（如 * 與 ?）展開後的實體檔案是否越界
        if '*' in t_clean or '?' in t_clean:
            try:
                matches = glob.glob(t_clean, recursive=True)
                for m in matches:
                    resolved = os.path.abspath(m)
                    if not is_inside(resolved, allowed_dir):
                        return False, f"通配符展開指向外部路徑：'{resolved}'"
            except Exception:
                pass
                
        # 6. 解析 Token 代表的絕對路徑，驗證是否指向專案目錄之外
        try:
            resolved = os.path.abspath(t_clean)
            if not is_inside(resolved, allowed_dir):
                return False, f"企圖存取外部路徑：'{resolved}'"
        except Exception as e:
            return False, f"路徑解析失敗：'{t_clean}' ({e})"
            
    return True, ""

# ─── Memory ───

conversation_history = []
key_info = []

# ─── Ollama API ───

async def call_ollama(prompt: str, system: str = "") -> str:
    """Call Ollama API"""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            result = await resp.json()
            return result.get("response", "").strip()

# ─── Memory Management ───

def build_context():
    context_parts = []
    if key_info:
        items_xml = "\n".join(f"  <item>{k}</item>" for k in key_info)
        context_parts.append(f"<memory>\n{items_xml}\n</memory>")
    if conversation_history:
        context_parts.append("<history>\n" + "\n".join(conversation_history[-MAX_TURNS*2:]) + "\n</history>")
    return "\n\n".join(context_parts)

def update_memory(user_input, assistant_response, tool_result=None):
    conversation_history.append(f"  <user>{user_input}</user>")
    conversation_history.append(f"  <assistant>{assistant_response}</assistant>")
    if tool_result:
        conversation_history.append(f"  <tool>{tool_result[:500]}</tool>")
    
    while len(conversation_history) > MAX_TURNS * 4:
        conversation_history.pop(0)

async def extract_key_info(user_input, assistant_response):
    extract_prompt = f"""根據這段對話，有沒有需要長期記憶的關鍵資訊？
如果有，用以下格式輸出（最多 2 項）。如果沒有，輸出 <memory></memory>。

<memory>
  <item>要記憶的資訊 1</item>
  <item>要記憶的資訊 2</item>
</memory>

對話：
<user>{user_input}</user>
<assistant>{assistant_response}</assistant>"""
    
    try:
        result = await call_ollama(extract_prompt, "")
        matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
        for item in matches:
            item = item.strip()
            if item and item not in key_info:
                key_info.append(item)
    except:
        pass

# ─── Agent ───

SYSTEM_PROMPT = """你是 Jarvis，一個有用的 AI 助理。

重要規則：
1. 當你需要執行 shell 命令時，必須用 <shell> 標籤包住命令
2. <shell> 標籤內可以是多行命令（用反斜槓 \\ 或 && 連接）
3. 當你完成所有操作後，用 <end/> 結束你的回覆

流程：
- 如果需要執行命令，輸出 <shell>...</shell>
- 執行完後我會顯示結果
- 如果還需要更多命令，繼續輸出 <shell>
- 當完成所有操作後，輸出 <end/> 表示結束"""

def main():
    os.makedirs(WORKSPACE, exist_ok=True)
    
    print(f"Agent0 - {MODEL}（安全防護版）")
    print(f"安全防護根目錄：{ALLOWED_DIR}")
    print(f"工作區：{WORKSPACE}")
    print("指令：/quit、/memory（顯示關鍵資訊）\n")
    
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再見！")
            break
        
        if not user_input:
            continue
        if user_input.lower() in ["/quit", "/exit", "/q"]:
            print("再見！")
            break
        if user_input.lower() == "/memory":
            print(f"關鍵資訊：{key_info}")
            continue
        
        context = build_context()
        full_prompt = f"{context}\n\n<user>{user_input}</user>" if context else f"<user>{user_input}</user>"
        
        response = asyncio.run(call_ollama(full_prompt, SYSTEM_PROMPT))
        
        tool_result = None
        current_response = response
        
        while True:
            if "<end/>" in current_response:
                response = current_response.split("<end/>")[0].strip()
                break
            
            shell_matches = re.findall(r'<shell>(.+?)</shell>', current_response, re.DOTALL)
            if not shell_matches:
                response = current_response
                break
            
            all_outputs = []
            for cmd in shell_matches:
                cmd = cmd.strip()
                
                # ─── 安全控管攔截器 (Security Filter Interceptor) ───
                is_safe, reason = check_command_safety(cmd, ALLOWED_DIR)
                
                approved = False
                if not is_safe:
                    print(f"\n⚠️  [安全警告] 偵測到指令企圖存取外部路徑或執行限制的敏感工具！")
                    print(f"原因：{reason}")
                    print(f"指令：{cmd}")
                    
                    # 提示人工進行核可 (Human-in-the-loop)
                    try:
                        user_choice = input("是否核可此操作？(y/n): ").strip().lower()
                        if user_choice in ['y', 'yes']:
                            approved = True
                            print("使用者已核可，繼續執行...\n")
                        else:
                            print("使用者已拒絕，指令已被攔截。\n")
                    except (EOFError, KeyboardInterrupt):
                        print("\n指令被自動攔截並拒絕。\n")
                else:
                    # 完全安全，自動核可執行
                    approved = True
                
                if approved:
                    try:
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=os.getcwd())
                        output = result.stdout + result.stderr
                        print(f"\n=== 執行命令 ===\n{cmd}\n\n結果：{output if output else '（無輸出）'}\n")
                        all_outputs.append(f"$ {cmd}\n{output if output else '（無輸出）'}")
                    except Exception as e:
                        print(f"錯誤：{e}")
                        all_outputs.append(f"$ {cmd}\n錯誤：{e}")
                else:
                    all_outputs.append(f"$ {cmd}\n安全警告：存取外部檔案/執行限制命令被攔截並拒絕。")
            
            tool_result = (tool_result or "") + "\n" + "\n".join(all_outputs)
            
            follow_up_prompt = f"""<context>{context}</context>

<user>{user_input}</user>
<assistant>{current_response}</assistant>
<output>
{chr(10).join(all_outputs)}
</output>

如果需要更多命令就輸出 <shell>。否則，輸出 <end/> 表示結束："""
            current_response = asyncio.run(call_ollama(follow_up_prompt, SYSTEM_PROMPT))
        
        print(f"\n🤖 {response}\n")
        
        update_memory(user_input, response, tool_result)
        if tool_result:
            asyncio.run(extract_key_info(user_input, response))

if __name__ == "__main__":
    main()
