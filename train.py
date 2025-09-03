# train.py (GPU 활용도 최적화 버전)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import time

from src.neural_network import GreatKingdomNet
from self_play import self_play
from evaluate import evaluate_models

# ================================================================
# 1. 하이퍼파라미터 설정 (캐글 GPU 최적화 버전)
# ================================================================
NUM_ITERATIONS = 100
NUM_GAMES_PER_ITERATION = 50
NUM_EPOCHS = 10
BATCH_SIZE = 256 # GPU 메모리를 최대한 활용하기 위해 배치 사이즈 증가
LEARNING_RATE = 0.001
NUM_SIMULATIONS = 150 # GPU 성능을 믿고 수읽기 깊이 증가

MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
CHALLENGER_PATH = os.path.join(MODEL_DIR, "challenger.pth")

# ================================================================
# 2. 데이터 증강을 위한 커스텀 데이터셋 클래스
# ================================================================
class AugmentedDataset(Dataset):
    def __init__(self, states, policies, outcomes):
        self.states = states
        self.policies = policies
        self.outcomes = outcomes

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        policy = self.policies[idx]
        outcome = self.outcomes[idx]

        # 8가지 대칭 변환 중 하나를 무작위로 선택
        choice = np.random.randint(8)
        
        # 좌우 반전
        if choice >= 4:
            state = np.flip(state, axis=2).copy()
        
        # 회전
        rotation = choice % 4
        if rotation > 0:
            state = np.rot90(state, k=rotation, axes=(1, 2)).copy()

        # 정책(policy)도 동일하게 변환
        policy_board = policy[:-1].reshape(9, 9)
        policy_pass = policy[-1]
        
        if choice >= 4:
            policy_board = np.fliplr(policy_board).copy()
        if rotation > 0:
            policy_board = np.rot90(policy_board, k=rotation).copy()
        
        policy_augmented = np.append(policy_board.flatten(), policy_pass)

        return torch.FloatTensor(state), torch.FloatTensor(policy_augmented), torch.FloatTensor([outcome])

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = GreatKingdomNet().to(device)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Champion model loaded from {BEST_MODEL_PATH}")
    else:
        print("No champion model found. Creating a new model.")
        torch.save(model.state_dict(), BEST_MODEL_PATH)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for i in range(1, NUM_ITERATIONS + 1):
        print(f"\n===== Iteration {i}/{NUM_ITERATIONS} =====")
        
        print(f"Generating {NUM_GAMES_PER_ITERATION} games of self-play data...")
        model.eval()
        all_game_data = []
        for g in range(NUM_GAMES_PER_ITERATION):
            print(f"  - Playing game {g+1}/{NUM_GAMES_PER_ITERATION}...", end='\r')
            game_data = self_play(model, NUM_SIMULATIONS, device)
            all_game_data.extend(game_data)
        print(f"\nData generation complete. Generated {len(all_game_data)} data points.")

        print("Preparing dataset with real-time augmentation and parallel loading...")
        states, policies, outcomes = zip(*all_game_data)
        dataset = AugmentedDataset(np.array(states), np.array(policies), np.array(outcomes))
        
        # --- [핵심 수정] 병렬 처리(num_workers) 옵션 추가 ---
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True,
            num_workers=4,  # CPU 코어 4개를 사용하여 데이터 로딩
            pin_memory=True # GPU로의 데이터 전송 가속화
        )
        # --------------------------------------------------

        print(f"Training the model for {NUM_EPOCHS} epochs...")
        model.train()
        
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            epoch_start_time = time.time()
            for batch_states, batch_policies, batch_outcomes in dataloader:
                batch_states = batch_states.to(device, non_blocking=True)
                batch_policies = batch_policies.to(device, non_blocking=True)
                batch_outcomes = batch_outcomes.to(device, non_blocking=True)
                
                log_pred_policies, pred_values = model(batch_states)
                
                # --- [핵심 수정] 안정적인 손실 함수 사용 ---
                policy_loss = F.kl_div(log_pred_policies, batch_policies, reduction='batchmean')
                value_loss = F.mse_loss(pred_values.squeeze(), batch_outcomes.squeeze())
                # ----------------------------------------
                
                loss = policy_loss + value_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            epoch_end_time = time.time()
            print(f"  Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {total_loss / len(dataloader):.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")

        torch.save(model.state_dict(), CHALLENGER_PATH)
        print(f"Challenger model saved to {CHALLENGER_PATH}")

        # 모델 평가
        is_new_champion = evaluate_models(
            challenger_path=CHALLENGER_PATH, 
            best_path=BEST_MODEL_PATH, 
            num_games=20, # 평가 게임 수를 늘려 더 신뢰도 높은 결과 확인
            device=device
        )

        if not is_new_champion:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
            print("Model reverted to the champion model for the next iteration.")

if __name__ == '__main__':
    train()

