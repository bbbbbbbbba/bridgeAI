from bridge import BridgePosition
import MCTS
import numpy as np
import tensorflow as tf
import DNN
import time

feature_size = len(BridgePosition().to_tensor())
policy_num = 52

feature_columns = [tf.feature_column.numeric_column("x", shape = [feature_size])]
classifier = tf.estimator.Estimator(model_fn = DNN.model_fn,
                                    params = {
                                        'feature_columns': feature_columns,
                                        'hidden_units': [1024, 1024, 1024],
                                        'n_classes': policy_num,
                                        'value_weight': 0.1
                                    })

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
        data_p.append([tree[0].move_to_int(move_max)])
        data_v.append([-tree[0].score])
        if visualize:
            card_str = tree[0].move_to_str(move_max)
            print(card_str, end = ' ' if tree[0].cards_in_trick < 3 else '\n', flush = True)
        tree = tree[-1][move_max]
    return tree[0].score, [np.array(data_x), np.array(data_p), np.array(data_v) + tree[0].score]

import pickle
import os
if os.path.exists('data.pickle'):
    with open('data.pickle', 'rb') as fin: data = pickle.load(fin)
else:
    data = [np.zeros((0, feature_size)), np.zeros((0, 1), dtype = int), np.zeros((0, 1))] # x, p, v
    for i in range(1000):
        t0 = time.process_time()
        score, data_game = self_play(visualize = i % 100 == 0)
        print("Game #%d time:" % i, time.process_time() - t0)
        for k in range(3): data[k] = np.vstack([data[k], data_game[k]])
    with open('data.pickle', 'wb') as fout: pickle.dump(data, fout)

train_num = data[0].shape[0] // 10 * 9

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': data[0][:train_num]}, y = {'p': data[1][:train_num], 'v': data[2][:train_num]},
    num_epochs = None, shuffle = True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'x': data[0][train_num:]}, y = {'p': data[1][train_num:], 'v': data[2][train_num:]},
    num_epochs = 1, shuffle = False)

for k in range(10):
    t0 = time.process_time()
    classifier.train(input_fn = train_input_fn, steps = 200)
    print("Train time:", time.process_time() - t0)
    t0 = time.process_time()
    metrics = classifier.evaluate(input_fn = test_input_fn)
    print("Test time:", time.process_time() - t0)

    t0 = time.process_time()
    for i in range(train_num, train_num + 52):
        pos = BridgePosition.from_tensor(data[0][i])
        policy, value = data[1][i][0], data[2][i][0]
        pos.visualize(verbose = True)
        print(pos.move_to_str(pos.int_to_move(policy)), value)
        pred = classifier.predict(input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {'x': data[0][i:i+1]}, y = None,
            num_epochs = 1, shuffle = False, batch_size = 1)
        )
        t1 = time.process_time()
        pred = next(pred)
        print("Time:", time.process_time() - t1)
        pred_policy, pred_value = pred['class_ids'][0], pred['val'][0]
        print(pos.move_to_str(pos.int_to_move(pred_policy)), pred_value)
        print()
    print("Prediction time:", time.process_time() - t0)

    accuracy_score, mse_value = metrics['accuracy'], metrics['mse_value']
    print(accuracy_score, mse_value)
