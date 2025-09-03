# evaluate.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import torch, copy, numpy as np
from src.game_engine import GreatKingdomGame
from src.mcts import MCTS
from src.neural_network import GreatKingdomNet

def play_game(model1, model2, device):
    game = GreatKingdomGame()
    models = {1: model1, -1: model2}
    while not game.game_over:
        current_model = models[game.current_player]
        mcts = MCTS(copy.deepcopy(game), current_model, device)
        mcts.search(100) # Evaluation simulations
        policy = mcts.get_policy(temperature=0)
        action = np.argmax(policy)
        move = ((action // 9, action % 9) if action < 81 else 'pass')
        game.make_move(move, silent=True)
    return game.get_winner()

def evaluate_models(challenger_path, best_path, num_games=10, device="cpu"):
    challenger_model = GreatKingdomNet().to(device)
    challenger_model.load_state_dict(torch.load(challenger_path, map_location=device))
    challenger_model.eval()

    best_model = GreatKingdomNet().to(device)
    best_model.load_state_dict(torch.load(best_path, map_location=device))
    best_model.eval()

    challenger_wins = 0
    print(f"\n--- Evaluating Challenger vs. Champion for {num_games} games ---")
    for i in range(num_games):
        winner = play_game(challenger_model, best_model, device) if i % 2 == 0 else play_game(best_model, challenger_model, device)
        if (i % 2 == 0 and winner == 1) or (i % 2 != 0 and winner == -1):
            challenger_wins += 1
        print(f"  Game {i+1}/{num_games} complete. Challenger wins: {challenger_wins}", end='\r')
    
    win_rate = challenger_wins / num_games
    print(f"\nEvaluation Complete. Challenger Win Rate: {win_rate:.2%}")
    return win_rate > 0.55
