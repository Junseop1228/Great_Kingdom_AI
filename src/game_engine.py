# src/game_engine.py

import numpy as np

class GreatKingdomGame:
    """
    '그레이트 킹덤'의 공식 규칙(공성 승리 포함)을 관리하는 최종 엔진 클래스입니다.
    """
    def __init__(self):
        self.board = np.zeros((9, 9), dtype=int)
        self.board[4, 4] = 2 # 중립 성
        self.current_player = 1 # 1: Player O (선공), -1: Player X (후공)
        self.game_over = False
        self.winner = None
        self.last_move_was_pass = False
        self.history = []

    def make_move(self, move, silent=False):
        """
        플레이어의 수를 처리하고 게임 상태를 업데이트합니다.
        """
        if self.game_over:
            return False

        if move == 'pass':
            if self.last_move_was_pass:
                p1_score, p2_score = self._calculate_territory()
                p2_score += 3
                if not silent: print(f"Game Over by territory. O_Score: {p1_score}, X_Score: {p2_score-3}+3={p2_score}")
                self.winner = 1 if p1_score > p2_score else -1 if p2_score > p1_score else 0
                self.game_over = True
                return True
            else:
                self.last_move_was_pass = True
                self.current_player *= -1
                return True
        
        row, col = move
        if not (0 <= row < 9 and 0 <= col < 9 and self.board[row, col] == 0):
            return False
        
        # 1. 임시로 수를 둠
        original_board = np.copy(self.board)
        self.board[row, col] = self.current_player
        
        # 2. 상대방 돌 포획 처리
        captured_stones = self._check_and_remove_captured_stones(-self.current_player)
        num_captured = len(captured_stones)

        # 3. 자살수 확인
        _, my_liberties = self._get_group(row, col, set())
        if not my_liberties and not captured_stones:
            self.board = original_board # 자살수이므로 보드 원상복구
            if not silent: print("Error: Suicide move is not allowed.")
            return False

        # 4. 공성 승리 확인 (포획 발생 시)
        if num_captured > 0:
            if not silent: print(f"Player {('O' if self.current_player == 1 else 'X')} captured {num_captured} stones!")
            self.game_over = True
            self.winner = self.current_player
            return True

        # 5. 모든 검증 통과 후, 턴 전환
        self.last_move_was_pass = False
        self.history.append(move)
        self.current_player *= -1
        return True

    def _get_group(self, r, c, visited_local):
        player = self.board[r, c]
        if player == 0 or (r, c) in visited_local: return set(), set()
        
        q, group, liberties = [(r, c)], set(), set()
        visited_local.add((r, c))
        
        head = 0
        while head < len(q):
            curr_r, curr_c = q[head]; head += 1
            group.add((curr_r, curr_c))
            
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    if self.board[nr, nc] == 0 or self.board[nr, nc] == 2:
                        liberties.add((nr, nc))
                    elif self.board[nr, nc] == player and (nr, nc) not in visited_local:
                        visited_local.add((nr, nc)); q.append((nr, nc))
        return group, liberties

    def _check_and_remove_captured_stones(self, player_to_check):
        captured_stones = set()
        visited_for_capture = set()
        for r in range(9):
            for c in range(9):
                if self.board[r, c] == player_to_check and (r,c) not in visited_for_capture:
                    group, liberties = self._get_group(r, c, visited_for_capture)
                    if not liberties:
                        captured_stones.update(group)
        
        if captured_stones:
            for r_cap, c_cap in captured_stones:
                self.board[r_cap, c_cap] = 0
        return captured_stones

    def get_total_liberties(self, player):
        total_liberties = 0
        visited_for_liberty = set()
        for r in range(9):
            for c in range(9):
                if self.board[r,c] == player and (r,c) not in visited_for_liberty:
                    group, liberties = self._get_group(r, c, visited_for_liberty)
                    total_liberties += len(liberties)
        return total_liberties
    
    def _calculate_territory(self):
        p1_territory, p2_territory = 0, 0
        visited = set()
        for r in range(9):
            for c in range(9):
                if self.board[r,c] == 0 and (r,c) not in visited:
                    group, borders = set(), set()
                    q = [(r, c)]; visited.add((r, c))
                    head = 0
                    while head < len(q):
                        curr_r, curr_c = q[head]; head+=1
                        group.add((curr_r, curr_c))
                        for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if not (0 <= nr < 9 and 0 <= nc < 9): continue
                            if self.board[nr, nc] == 0 and (nr, nc) not in visited:
                                visited.add((nr, nc)); q.append((nr, nc))
                            elif self.board[nr, nc] != 0: borders.add(self.board[nr, nc])
                    border_colors = {b for b in borders if b in [1, -1]}
                    if len(border_colors) == 1:
                        owner = border_colors.pop()
                        if owner == 1: p1_territory += len(group)
                        else: p2_territory += len(group)
        return p1_territory, p2_territory
    
    def get_state_tensor(self):
        player_stones = np.where(self.board == self.current_player, 1, 0)
        opponent_stones = np.where(self.board == -self.current_player, 1, 0)
        turn_indicator = np.ones((9, 9))
        return np.stack([player_stones, opponent_stones, turn_indicator])
    
    def get_valid_moves(self):
        valid_moves = []
        original_board_state = np.copy(self.board)
        for r in range(9):
            for c in range(9):
                if self.board[r,c] == 0:
                    # 임시로 수를 둬봄
                    self.board[r,c] = self.current_player
                    captured_stones = self._check_and_remove_captured_stones(-self.current_player)
                    _, my_liberties = self._get_group(r,c, set())
                    # 보드 원상복구
                    self.board = np.copy(original_board_state)
                    # 자살수가 아니면 유효한 수
                    if my_liberties or captured_stones:
                        valid_moves.append((r,c))
        return valid_moves

    def get_winner(self):
        return self.winner

    def display_board(self):
        symbols = {1: 'O', -1: 'X', 0: '.', 2: 'N'}
        print("   " + " ".join([str(i) for i in range(9)]))
        print(" +" + "-" * 17 + "+")
        for r in range(9):
            print(f"{r}| {' '.join([symbols[self.board[r, c]] for c in range(9)])} |")
        print(" +" + "-" * 17 + "+")
        print(f"Current Player: {symbols.get(self.current_player, 'Unknown')}")
