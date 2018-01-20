from copy import deepcopy
import random

C_PUCT = 1.0

def init_tree(pos):
    return [pos, 0, 0.0, None, 1.0, None] # Position, N, W, Q, P, children list (None for leaf)

def MCTS_iteration(root):
    # select
    path = [root]
    while path[-1][-1] is not None:
        a_max = None
        c = C_PUCT * (path[-1][1] - 1) ** 0.5
        mult = 1 if path[-1][0].currentPlayer % 2 == 0 else -1
        for move, node in path[-1][-1].items():
            a = node[3] * mult + c * node[4] / (1 + node[1])
            if a_max is None or a > a_max:
                a_max = a
                move_max = move
        path.append(path[-1][-1][move_max])

    # expand and evaluate
    pos = path[-1][0]
    moves = pos.getMoves()
    v, p = evaluate(pos, moves)
    if len(moves) > 0:
        children = {}
        for move in moves:
            y = deepcopy(pos)
            y.makeMove(move)
            children[move] = [y, 0, 0.0, v, p[move], None]

        path[-1][-1] = children

    # backup
    for node in path:
        node[1] += 1
        node[2] += v
        node[3] = node[2] / node[1]

def evaluate(pos, moves):
    # TODO
    # should always output estimated score for players 0 and 2,
    # even though DNN always see the situation from the perspective of the current player
    tricks_left = sum(len(x) for x in pos.hands[pos.currentPlayer])
    return (pos.score + tricks_left / 2, {move: 1 / len(moves) for move in moves})

def visualize_tree(root, depth = 0, move = None):
    card_str = '**' if move is None else '%s%s' % ('SHDC'[move[0]], '  23456789TJQKA'[move[1]])
    print('%s%s %d %f %f' % ('    ' * depth, card_str, root[1], root[3], root[4]))
    if root[-1] is not None:
        moves = list(root[-1].keys())
        moves.sort(key = lambda move: root[-1][move][1], reverse = True)
        for move in moves: visualize_tree(root[-1][move], depth + 1, move)
