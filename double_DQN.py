import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import sys

# 將包含 Gridworld.py 的資料夾路徑加入系統路徑
sys.path.append("C:/Users/User/Desktop/reinforce/0506/DeepReinforcementLearningInAction-master/Chapter 3")
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

# 2. 🌟 Dueling DQN 神經網路架構
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        # 共用特徵提取層
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 150),
            nn.ReLU()
        )
        # Value 分支: 評估「這個狀態本身」有多好
        self.value_stream = nn.Sequential(
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, 1)
        )
        # Advantage 分支: 評估「每個動作比平均」好多少
        self.advantage_stream = nn.Sequential(
            nn.Linear(150, 150),
            nn.ReLU(),
            nn.Linear(150, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 組合公式: Q(s,a) = V(s) + (A(s,a) - mean(A))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

# 3. 🌟 Double DQN 損失計算邏輯
def compute_double_dqn_loss(batch_size, replay_buffer, optimizer, q_online, q_target, gamma=0.9):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action     = torch.LongTensor(action)
    reward     = torch.FloatTensor(reward)
    done       = torch.FloatTensor(done)

    # 取得當前 Q 值
    q_values = q_online(state)
    q_value  = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    # Double DQN 核心：用 Online 選動作，用 Target 算價值
    with torch.no_grad():
        next_actions = q_online(next_state).argmax(dim=1, keepdim=True)
        next_q_values = q_target(next_state).gather(1, next_actions).squeeze(1)
        expected_q_value = reward + gamma * next_q_values * (1 - done)

    # 使用 Huber Loss (Smooth L1) 取代 MSE，讓訓練更穩定
    loss = F.smooth_l1_loss(q_value, expected_q_value)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 4. 主程式執行
if __name__ == "__main__":
    print("啟動老師的環境: Gridworld (Player Mode)...")
    
    state_dim = 64 
    action_dim = 4 

    # 使用 DuelingDQN
    q_online = DuelingDQN(state_dim, action_dim)
    q_target = DuelingDQN(state_dim, action_dim)
    q_target.load_state_dict(q_online.state_dict())
    
    optimizer = optim.Adam(q_online.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(2000)

    batch_size = 64
    episodes = 1000 # Player 模式較難，稍微增加回合數
    gamma = 0.9
    epsilon = 1.0       
    epsilon_decay = 0.995 # 減緩探索衰減率
    epsilon_min = 0.05

    print("開始訓練 Dueling + Double DQN (Player Mode)...")

    for episode in range(episodes):
        # 🌟 HW3-2 規定要用 'player' 模式
        env = Gridworld(size=4, mode='player')
        
        state = env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
        
        done = False
        episode_reward = 0
        step_count = 0

        # Player 模式可能需要多走幾步，上限調至 25
        while not done and step_count < 25:
            step_count += 1
            
            if random.random() < epsilon:
                action = random.randint(0, 3) 
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = q_online(state_tensor).argmax().item()

            env.makeMove(action)
            reward = env.reward()
            next_state = env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
            
            done = True if reward != -1 else False

            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            if len(replay_buffer) > batch_size:
                compute_double_dqn_loss(batch_size, replay_buffer, optimizer, q_online, q_target, gamma)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10 == 0:
            q_target.load_state_dict(q_online.state_dict())
            
        if episode % 50 == 0:
            print(f"Episode: {episode:3d} | Reward: {episode_reward:5.1f} | Epsilon: {epsilon:.3f} | Steps: {step_count}")

    print("訓練結束！這份結果可以用於 HW3-2 的報告截圖。")