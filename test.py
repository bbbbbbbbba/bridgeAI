from bridge import BridgePosition
import MCTS
x = BridgePosition([[[2,3],[],[],[]],[[11,13],[],[],[]],[[12,14],[],[],[]],[[4,5],[],[],[]]])
tree = MCTS.init_tree(x)
for k in range(100): MCTS.MCTS_iteration(tree)
MCTS.visualize_tree(tree)
