import math
import copy

X = "X"
O = "O"
EMPTY = None
INFINITY = 1e3

class InvalidAction(Exception):
    "Raise when an action is not available"
    pass

def initial_state():
    "Return the initial state"
    return [[EMPTY] * 3 for _ in range(3)]

def player(board):
    "Returns palyer who has next turn on a board. X starts the game"
    x_counter , o_counter=0, 0
    
    for each_row in board:
        # row wise count of each X and O
        x_counter+= each_row.count(X)
        o_counter+= each_row.count(O)
        
    if o_counter < x_counter:
        return O
    return X

def actions(board):
    action = set()
    for row_idx, row in enumerate(board):
        for col_idx, value in enumerate(row):
            if value not in (X,O):
                action.add((row_idx, col_idx))
    return action

def result(board, action):
    if action not in actions(board):
        raise InvalidAction("You have peformed an invalid action")
    
    curr_player = player(board)
    new_board = copy.deepcopy(board)
    
    row_idx, col_idx = action
    
    new_board[row_idx][col_idx] = curr_player
    
    return new_board


def winner(board):
    Board_size = len(board)
    
    for each_row in board:
        if each_row.count(X) == Board_size:
            return X
        if each_row.count(O) == Board_size:
            return O
        
    for col_idx in  range(Board_size):
        col = [row[col_idx] for row in board]
        if col.count(X) == Board_size:
            return X
        if col.count(O) == Board_size:
            return O
        
    main_diag = [board[i][i] for i in range(Board_size)]
    anti_diag = [board[Board_size-i-1][i] for i in range(Board_size)]
    
    if main_diag.count(X) == Board_size \
        or anti_diag.count(X) == Board_size:
            return X
    
    if main_diag.count(O) == Board_size \
        or anti_diag.count(O)  == Board_size:
            return O
    
    return None


def terminal(board):
    if winner(board) in (X, O):
        return True
    
    if not any(map(lambda x: EMPTY in x,board)):
        return True
    
    return False


def utility(board):
    won =1
    lost = -1
    draw =0
    
    if winner(board) == X:
        return won
    elif winner(board) == O:
        return lost
    
    return draw

def minimax(board):
    
    if terminal(board):
        return None
    
    curr_player = player(board)
    
    
    if curr_player  ==X:
        can_outcome = -INFINITY
        
        for action in actions(board):
            possible_board = result( board, action)
            
            oppo_utility = min_value(possible_board)
            
            if oppo_utility > can_outcome:
                can_outcome = oppo_utility
                can_action  = action
    elif curr_player ==O:
        can_outcome = INFINITY
        
        for action in actions(board):
            possible_board = result(board, action)
            
            oppo_utility  = max_value(possible_board)
            
            if oppo_utility < can_outcome:
                
                can_outcome = oppo_utility
                can_action = action
                
    return  can_action            


def max_value(board):
    v = -INFINITY
    
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        result_board = result(board, action)
        v= max(v, min_value(result_board))
        
    return v

def min_value(board):
    v = INFINITY
    
    if terminal(board):
        return utility(board)
    
    for action in actions(board):
        result_board = result(board, action)
        v = min(v, max_value(result_board))
        
    return v 