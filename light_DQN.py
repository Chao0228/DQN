import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import numpy as np
import random
from collections import deque
import sys

# 載入老師的環境
sys.path.append("C:/Users/User/Desktop/reinforce/0506/DeepReinforcementLearningInAction-master/Chapter 3")
from Gridworld import Gridworld

# 1. 經驗回放池
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(states)), torch.LongTensor(actions), 
                torch.FloatTensor(rewards), torch.FloatTensor(np.array(next_states)), torch.FloatTensor(dones))
    def __len__(self):
        return len(self.buffer)

# 2. 網路架構 (使用 Dueling DQN 應付最難的 Random 模式)
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(nn.Linear(input_dim, 150), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(150, 150), nn.ReLU(), nn.Linear(150, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(150, 150), nn.ReLU(), nn.Linear(150, output_dim))

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

# 3. 🌟 PyTorch Lightning 核心模組 (包含 Bonus Training Tips)
class LitRLAgent(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # HW3-3 要求使用 random 模式
        self.env = Gridworld(size=4, mode='random')
        self.q_net = DuelingDQN(64, 4)
        self.q_target = DuelingDQN(64, 4)
        self.q_target.load_state_dict(self.q_net.state_dict())
        
        self.buffer = ReplayBuffer(5000)
        self.batch_size = 64
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.999 # 放慢衰減讓它有多點時間在 random 模式中探索
        self.epsilon_min = 0.05
        self.tau = 0.005 # Soft Update 參數
        
        self.step_count = 0
        self.state = self.env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0

        # 先隨機亂走收集一些初始資料放入 Buffer
        for _ in range(100):
            self.play_one_step()

    # 負責與環境互動的邏輯
    def play_one_step(self):
        if random.random() < self.epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
                action = self.q_net(state_tensor).argmax().item()

        self.env.makeMove(action)
        reward = self.env.reward()
        next_state = self.env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
        
        done = True if reward != -1 else False
        self.step_count += 1
        if self.step_count >= 25:
            done = True

        self.buffer.push(self.state, action, reward, next_state, done)

        if done:
            self.env = Gridworld(size=4, mode='random')
            self.state = self.env.board.render_np().flatten() + np.random.rand(1, 64).flatten() / 10.0
            self.step_count = 0
        else:
            self.state = next_state

    # Lightning 訓練步驟
    def training_step(self, batch, batch_idx):
        # 1. 每次訓練前，先在環境裡走一步
        self.play_one_step()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 2. 從 Buffer 取出資料訓練
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.q_target(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 💡 Bonus Tip 1: 使用 Huber Loss (Smooth L1 Loss) 取代 MSE，對極端誤差更穩定
        loss = F.smooth_l1_loss(q_values, target_q_values)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('epsilon', self.epsilon, prog_bar=True)
        return loss

    # 💡 Bonus Tip 2: Soft Target Update (每步微調 Target 網路，而不是久久覆蓋一次)
    def on_train_batch_end(self, outputs, batch, batch_idx):
        for target_param, online_param in zip(self.q_target.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * online_param.data + (1.0 - self.tau) * target_param.data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        # 💡 Bonus Tip 3: Learning Rate Scheduler (隨著訓練降低學習率幫助收斂)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
        return [optimizer], [scheduler]

# 4. 主程式
if __name__ == "__main__":
    print("啟動 PyTorch Lightning 訓練 (Random Mode)...")
    
    # 建立假的 Dataset 來驅動 Lightning 的迴圈 (我們設定訓練 2000 步)
    dummy_dataset = TensorDataset(torch.zeros(2000))
    dataloader = DataLoader(dummy_dataset, batch_size=1)

    model = LitRLAgent()

    # 💡 Bonus Tip 4: Gradient Clipping (限制梯度爆炸)
    trainer = pl.Trainer(
        max_epochs=1, 
        gradient_clip_val=1.0, 
        enable_model_summary=False
    )
    
    trainer.fit(model, train_dataloaders=dataloader)
    print("HW3-3 訓練結束！")