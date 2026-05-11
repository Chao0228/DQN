import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import sys
sys.path.append("C:/Users/User/Desktop/reinforce/0506/DeepReinforcementLearningInAction-master/Chapter 3")

# 現在 Python 找得到這個資料夾了，可以順利載入環境
# 載入老師的環境 (請確保 Gridworld.py 和 GridBoard.py 在同一個資料夾)
from Gridworld import Gridworld

# 1. 經驗回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32), 
                np.array(next_state), np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# 2. 基礎 DQN 神經網路 (Homework 3-1 要求: Naive DQN)
class NaiveDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NaiveDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 3. 損失函數計算 
def compute_td_loss(batch_size, replay_buffer, optimizer, q_net, target_net, gamma=0.9):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)

    q_values = q_net(state)
    q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_net(next_state)
        next_q_value  = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = F.mse_loss(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 4. 主程式執行
if __name__ == "__main__":
    print("啟動老師的環境: Gridworld (Static Mode)...")
    
    # 初始化迷宮：HW3-1 規定要用 'static'
    env = Gridworld(size=4, mode='static')
    
    state_dim = 64 # 老師的 4x4 迷宮展開後固定是 64 維
    action_dim = 4 # 上、下、左、右

    q_net = NaiveDQN(state_dim, action_dim)
    target_net = NaiveDQN(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(1000)

    batch_size = 32
    episodes = 500
    gamma = 0.9
    epsilon = 1.0       
    epsilon_decay = 0.99
    epsilon_min = 0.01

    print("開始訓練 Naive DQN (Static Mode)...")

    for episode in range(episodes):
        # 每次重新開始遊戲
        env = Gridworld(size=4, mode='static')
        
        # 取得初始狀態，加上極小的隨機雜訊，防止神經網路過擬合
        state = env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
        
        done = False
        episode_reward = 0
        step_count = 0

        # 設定最大步數防止無窮迴圈
        while not done and step_count < 15:
            step_count += 1
            
            # Epsilon-Greedy
            if random.random() < epsilon:
                action = random.randint(0, 3) 
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = q_net(state_tensor).argmax().item()

            # 執行老師環境的動作
            env.makeMove(action)
            reward = env.reward()
            next_state = env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
            
            # 判斷老師環境是否結束
            done = True if reward != -1 else False

            # 存入 Experience Replay
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                compute_td_loss(batch_size, replay_buffer, optimizer, q_net, target_net, gamma)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        # 狀態輸出
        if episode % 50 == 0:
            print(f"Episode: {episode:3d} | Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f} | Steps: {step_count}")

    print("訓練結束！這份結果可以用於 HW3-1 的報告截圖。")