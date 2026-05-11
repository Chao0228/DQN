# 強化學習作業 3：DQN 及其變體 (DQN and its variants)
**測試硬體：** NVIDIA GeForce RTX 4090 / CUDA 12.1  

本專案實作了基礎的 Deep Q-Network (DQN) 以及進階變體（Double DQN、Dueling DQN），並將模型應用於《Deep Reinforcement Learning in Action》第三章的自定義 `Gridworld` 迷宮環境中。最終透過 `PyTorch Lightning` 框架對難度最高的隨機環境進行訓練優化。

---

## 📁 專案檔案結構

請確保以下所有檔案都位於同一個資料夾下才能順利執行：

- `DQN_Experience.py` : HW3-1 執行腳本 (Naive DQN 與 Experience Replay)
- `double_DQN.py` : HW3-2 執行腳本 (Double DQN + Dueling DQN)
- `light_DQN.py` : HW3-3 執行腳本 (PyTorch Lightning 與訓練優化技巧)
- `Gridworld.py` : 環境主程式 (教師提供)
- `GridBoard.py` : 環境依賴模組 (教師提供)

---

## 🛠️ 環境需求與套件安裝

本專案使用 Python 撰寫，建議在具備 GPU 加速 (CUDA) 的環境下執行。

**核心依賴套件：**
- `torch` (>= 2.4.0)
- `numpy` (>= 2.x 相容)
- `pytorch-lightning` (>= 2.4.0)

**安裝環境指令：**
```bash
pip install torch numpy
pip install "pytorch-lightning>=2.4.0"