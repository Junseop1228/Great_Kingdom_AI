# self_play.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, numpy as np, copy, random
from src.game_engine import GreatKingdomGame
from src.mcts import MCTS
from src.neural_network import GreatKingdomNet

def self_play(model, num_simulations, device):
    game = GreatKingdomGame()
    training_data = []
    model.to(device)
    model.eval()

    while not game.game_over:
        if len(game.history) < 4:
            valid_moves = game.get_valid_moves()
            if not valid_moves: move = 'pass'
            else: move = random.choice(valid_moves)
            game.make_move(move, silent=True)
            continue
        
        mcts = MCTS(copy.deepcopy(game), model, device)
        mcts.search(num_simulations)
        
        tau = 1.0 if len(game.history) < 15 else 0.1
        policy = mcts.get_policy(temperature=tau)
        
        if np.sum(policy) == 0: action_idx = 81 # Pass if no valid moves
        elif tau > 0: action_idx = np.random.choice(len(policy), p=policy)
        else: action_idx = np.argmax(policy)
        
        training_data.append([game.get_state_tensor(), policy, game.current_player])
        move = ((action_idx // 9, action_idx % 9) if action_idx < 81 else 'pass')
        game.make_move(move, silent=True)

    winner = game.get_winner()
    dataset = []
    for state, policy, player in training_data:
        outcome = winner * player if winner is not None and winner != 0 else 0
        dataset.append((state, policy, outcome))
    return dataset
