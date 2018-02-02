import tensorflow as tf

def model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units = units, activation = tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    val = tf.layers.dense(net, 1, activation = None)

    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
            'val': val
        }
        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    loss_policy = tf.losses.sparse_softmax_cross_entropy(labels = labels['p'], logits = logits)
    value_weight = params['value_weight'] if 'value_weight' in params else 1.0
    loss_value = tf.losses.mean_squared_error(labels = labels['v'], predictions = val,
                                              weights = value_weight)
    total_loss = loss_policy + loss_value

    accuracy = tf.metrics.accuracy(labels = labels['p'],
                                   predictions = predicted_classes,
                                   name = 'acc_op')
    mse_value = tf.metrics.mean_squared_error(labels = tf.cast(labels['v'], tf.float32),
                                              predictions = val,
                                              name = 'mse_op')
    metrics = {'accuracy': accuracy, 'mse_value': mse_value}
    tf.summary.scalar('accuracy', accuracy[1])
    tf.summary.scalar('mse_value', mse_value[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss = total_loss, eval_metric_ops = metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate = 0.1)
    train_op = optimizer.minimize(total_loss, global_step = tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss = total_loss, train_op = train_op)
