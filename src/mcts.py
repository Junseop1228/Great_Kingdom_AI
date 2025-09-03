# src/mcts.py

import numpy as np
import math
import copy
import torch

class Node:
    """
    MCTS 트리의 각 노드를 나타냅니다. 게임 상태와 통계 정보를 저장합니다.
    """
    def __init__(self, game, parent=None, move=None, prior_p=0):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.prior_p = prior_p

class MCTS:
    """
    '순수 강화학습' 원칙에 따라 설계된, 안정성이 강화된 MCTS 클래스
    """
    def __init__(self, game, model, device, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
        self.game = game
        self.model = model
        self.device = device
        self.root = Node(game=copy.deepcopy(self.game), prior_p=1.0)
        
        # 루트 노드를 즉시 확장하여 초기 정책과 가치를 얻습니다.
        # 이 과정에서 발생할 수 있는 value=None 문제를 방어합니다.
        initial_value = self._expand_and_evaluate(self.root)
        if initial_value is None:
            initial_value = 0.0 # 만약의 경우를 대비한 방어 코드
        
        # 루트 노드의 자식들에게 디리클레 노이즈를 추가하여 탐험을 장려합니다.
        num_children = len(self.root.children)
        if num_children > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * num_children)
            for i, child in enumerate(self.root.children):
                child.prior_p = (1 - dirichlet_epsilon) * child.prior_p + dirichlet_epsilon * noise[i]

    def search(self, iterations):
        """
        주어진 횟수만큼 MCTS 시뮬레이션을 실행합니다.
        """
        for _ in range(iterations):
            node = self.root
            
            # 1. 선택 (Selection): 리프 노드까지 UCT 점수가 가장 높은 자식을 따라 내려갑니다.
            while node.children:
                node = self._select_child(node)
            
            # 2. 확장 및 평가 (Expand & Evaluate) / 3. 역전파 (Backpropagation)
            value = 0.0 # value의 기본값을 명확히 0.0으로 설정
            
            if not node.game.game_over:
                # 게임이 끝나지 않았으면, 노드를 확장하고 신경망으로 가치를 평가합니다.
                value = self._expand_and_evaluate(node)
                if value is None: value = 0.0 # 방어 코드
            else:
                # 게임이 끝났으면, 실제 승패 결과를 가치로 사용합니다.
                winner = node.game.get_winner()
                if winner is not None:
                    # 현재 노드의 플레이어 관점에서의 승패 가치 (-1, 0, 1)
                    value = float(winner * node.game.current_player)
            
            # 계산된 가치를 루트 노드까지 역전파합니다.
            self._backpropagate(node, value)
            
    def get_policy(self, temperature=1.0):
        """
        탐색 완료 후, 루트 노드의 방문 횟수를 바탕으로 정책 벡터를 반환합니다.
        """
        if not self.root.children: return np.zeros(82)
        
        visits = np.array([child.visits for child in self.root.children])
        if np.sum(visits) == 0: return np.ones(82) / 82

        if temperature == 0: # 결정적 선택 (가장 많이 방문한 수 선택)
            probs = np.zeros_like(visits, dtype=float)
            if np.sum(visits) > 0: probs[np.argmax(visits)] = 1.0
        else: # 확률적 선택 (방문 횟수에 비례하여 선택)
            visits_temp = visits**(1.0 / temperature)
            probs = visits_temp / np.sum(visits_temp)
            
        policy = np.zeros(82)
        for child, prob in zip(self.root.children, probs):
            if child.move != 'pass':
                idx = child.move[0] * 9 + child.move[1]
                policy[idx] = prob
            else:
                policy[81] = prob
        return policy

    def _select_child(self, node, exploration_constant=1.4):
        """
        PUCT(UCT 변형) 점수가 가장 높은 자식 노드를 선택합니다.
        """
        best_score, best_child = -float('inf'), None
        for child in node.children:
            # Q(s,a): 평균 액션 가치 (승률)
            q_value = (child.wins / child.visits) if child.visits > 0 else 0.0
            # U(s,a): 탐험 보너스
            u_value = exploration_constant * child.prior_p * (math.sqrt(node.visits) / (1 + child.visits))
            uct_score = q_value + u_value # 이 노드의 최종 점수
            if uct_score > best_score:
                best_score, best_child = uct_score, child
        return best_child

    def _expand_and_evaluate(self, node):
        """
        리프 노드를 확장하고, 신경망을 통해 자식 노드의 정책과 현재 노드의 가치를 평가합니다.
        """
        state_tensor = torch.FloatTensor(node.game.get_state_tensor()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_policy, value_tensor = self.model(state_tensor)
        
        policy = torch.exp(log_policy).squeeze().cpu().numpy()
        
        valid_moves = node.game.get_valid_moves()
        if not valid_moves: valid_moves.append('pass') # 둘 곳이 없으면 패스만 가능
        
        for move in valid_moves:
            idx = (move[0] * 9 + move[1]) if move != 'pass' else 81
            prior_p = policy[idx]
            
            child_game = copy.deepcopy(node.game)
            if child_game.make_move(move, silent=True):
                child_node = Node(game=child_game, parent=node, move=move, prior_p=prior_p)
                node.children.append(child_node)
        
        return value_tensor.item()

    def _backpropagate(self, node, value):
        """
        시뮬레이션 결과를 루트 노드까지 역전파하여 통계를 업데이트합니다.
        """
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            # 현재 노드의 '부모' 플레이어 관점에서 승패를 기록합니다.
            # 즉, 내 턴의 결과가 부모(상대 턴)에게는 어떤 가치를 갖는지 업데이트합니다.
            current_node.wins += value if current_node.game.current_player != self.root.game.current_player else -value
            current_node = current_node.parent
