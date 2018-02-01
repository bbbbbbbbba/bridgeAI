from bridge import BridgePosition
import MCTS
import numpy as np
import tensorflow as tf

feature_size = len(BridgePosition().to_tensor())
policy_num = 52

feature_columns = [tf.feature_column.numeric_column("x", shape = [feature_size])]
classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns,
                                        hidden_units = [1024, 1024, 1024],
                                        n_classes = policy_num,
                                        model_dir = "/tmp/double_dummy_model")

def self_play(visualize = False):
    tree = MCTS.init_tree(BridgePosition())
    data_x = []
    data_p = []
    data_v = []
    if visualize: tree[0].visualize()
    while len(tree[0].get_moves()) > 0:
        for k in range(100): MCTS.MCTS_iteration(tree)
        n_max = -1
        for move, node in tree[-1].items():
            if node[1] > n_max:
                n_max = node[1]
                move_max = move
        data_x.append(tree[0].to_tensor())
        # data_p.append([0] * policy_num)
        # data_p[-1][tree[0].move_to_int(move_max)] = 1
        data_p.append([tree[0].move_to_int(move_max)])
        data_v.append([-tree[0].score])
        if visualize:
            card_str = tree[0].move_to_str(move_max)
            print(card_str, end = ' ' if tree[0].cards_in_trick < 3 else '\n', flush = True)
        tree = tree[-1][move_max]
    return tree[0].score, [np.array(data_x), np.array(data_p), np.array(data_v) + tree[0].score]

data = [np.zeros((0, feature_size)), np.zeros((0, 1), dtype = int), np.zeros((0, 1))] # x, p, v
for i in range(1000):
    print(i)
    score, data_game = self_play(visualize = i % 100 == 0)
    for k in range(3): data[k] = np.vstack([data[k], data_game[k]])

train_num = data[0].shape[0] // 10 * 9

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": data[0][:train_num]}, y = data[1][:train_num],
    num_epochs = None, shuffle = True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {"x": data[0][train_num:]}, y = data[1][train_num:],
    num_epochs = 1, shuffle = False)

for k in range(10):
    classifier.train(input_fn = train_input_fn, steps = 200)
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print(accuracy_score)
