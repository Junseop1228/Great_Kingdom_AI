# play_ai.py (완전 수정 최종본)

import torch
import time
import copy
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.game_engine import GreatKingdomGame
from src.mcts import MCTS
from src.neural_network import GreatKingdomNet

def main():
    """
    훈련된 AI와 사람이 대결하는 메인 게임 루프입니다. (최종 버전)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading AI model from 'models/best_model.pth'...")
    model = GreatKingdomNet().to(device)
    try:
        model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'best_model.pth' not found. AI will play with a random, untrained model.")
    
    model.eval()

    game = GreatKingdomGame()

    HUMAN_PLAYER = None
    while HUMAN_PLAYER is None:
        choice = input("Choose your side. Enter 'O' (First) or 'X' (Second): ").upper()
        if choice == 'O':
            HUMAN_PLAYER = 1
        elif choice == 'X':
            HUMAN_PLAYER = -1
        else:
            print("Invalid choice. Please enter 'O' or 'X'.")

    AI_PLAYER = -HUMAN_PLAYER
    print(f"You are Player {choice}. The AI is Player {'O' if AI_PLAYER == 1 else 'X'}.")
    time.sleep(1)

    while not game.game_over:
        game.display_board()
        
        if game.current_player == AI_PLAYER:
            # --- AI의 턴 ---
            print(f"AI (Player {'O' if AI_PLAYER == 1 else 'X'}) is thinking...")
            start_time = time.time()
            
            mcts = MCTS(copy.deepcopy(game), model, device=device)
            mcts.search(iterations=500)
            policy = mcts.get_policy(temperature=0)

            end_time = time.time()

            if np.sum(policy) > 0:
                best_move_idx = np.argmax(policy)
                best_move = ((best_move_idx // 9, best_move_idx % 9) if best_move_idx < 81 else 'pass')
            else:
                print("AI cannot find a good move and decided to PASS.")
                best_move = 'pass'
            
            if best_move == 'pass':
                print(f"AI chose to PASS in {end_time - start_time:.2f} seconds.")
            else:
                print(f"AI chose ({int(best_move[0])}, {int(best_move[1])}) in {end_time - start_time:.2f} seconds.")

            # --- [핵심 수정] AI가 결정한 수를 실제 게임에 적용합니다. ---
            game.make_move(best_move)
            # --------------------------------------------------------

        else:
            # --- 사람의 턴 ---
            player_symbol = 'O' if HUMAN_PLAYER == 1 else 'X'
            is_valid_input = False
            move = None
            while not is_valid_input:
                move_str = input(f"Your turn (Player {player_symbol}), enter move (e.g., '3,4' or 'pass'): ")
                try:
                    if move_str.lower() == 'pass':
                        move = 'pass'
                        is_valid_input = True
                    else:
                        row, col = map(int, move_str.split(','))
                        move = (row, col)
                        
                        temp_game = copy.deepcopy(game)
                        if temp_game.make_move(move, silent=True):
                            is_valid_input = True
                        else:
                            print("Invalid move. That spot might be occupied or a suicide move.")
                except ValueError:
                    print("Invalid input format or coordinate out of range. Please use 'row,col' (0-8) or 'pass'.")
            
            game.make_move(move)
    
    print("\n" + "="*20)
    print("--- FINAL BOARD ---")
    game.display_board()
    
    winner_val = game.winner
    if winner_val == 0:
        winner_msg = "Draw"
    elif winner_val == HUMAN_PLAYER:
        winner_msg = "Congratulations, you win!"
    else:
        winner_msg = "The AI wins!"
        
    print(f"\nGAME OVER. Result: {winner_msg}")
    print("="*20 + "\n")

if __name__ == '__main__':
    main()
