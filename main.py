import os
import threading
import time
from ingest import Librarian
from rag_agent import run_rag_loop

def background_librarian_task(interval_seconds: int = 86400):
    """
    Background task that runs the Librarian sync periodically.
    Default interval is 24 hours (86400 seconds).
    """
    librarian = Librarian()
    while True:
        print("\n[Background] Librarian is checking for knowledge updates...")
        librarian.sync()
        print(f"[Background] Next check in {interval_seconds/3600:.1f} hours.")
        time.sleep(interval_seconds)

def main():
    print("\n" + "="*40)
    print("   Multi-Role AI Agent RAG System")
    print("="*40)
    
    # 1. Initial Sync
    print("\n[1/3] 正在初始化知识库...")
    print(">>> Librarian 正在扫描数据目录并生成向量索引，请稍等...")
    librarian = Librarian()
    librarian.sync()
    
    # 2. Start Background Librarian (Daily Sync)
    print("\n[2/3] 正在启动后台守护进程...")
    librarian_thread = threading.Thread(target=background_librarian_task, daemon=True)
    librarian_thread.start()
    print(">>> Librarian 已进入后台，将每 24 小时自动更新一次知识。")
    
    # 3. Start Interactive RAG Assistant
    print("\n[3/3] 正在启动知识助手...")
    print("\n" + "-"*40)
    print("✅ 系统已就绪！您可以开始提问了。")
    print("-"*40)
    
    try:
        run_rag_loop()
    except KeyboardInterrupt:
        print("\n\n[System] 正在关闭系统...")

if __name__ == "__main__":
    main()
