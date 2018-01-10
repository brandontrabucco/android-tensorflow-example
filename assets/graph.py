import tensorflow as tf


BASEDIR = "G:/My Drive/Academic/Professional/Neural RPS/"
LEARNING_RATE = 0.001
LOSS_COLLECTION = "loss_collection"


def initialize_weights_cpu(
        name,
        shape,
        standard_deviation=0.01,
        decay_factor=None,
        collection=None):
    with tf.device("/cpu:0"):
        weights = tf.get_variable(
            name,
            shape,
            initializer=tf.truncated_normal_initializer(
                stddev=standard_deviation,
                dtype=tf.float32),
            dtype=tf.float32)
    if decay_factor is not None and collection is not None:
        weight_decay = tf.multiply(
            tf.nn.l2_loss(weights),
            decay_factor)
        tf.add_to_collection(collection, weight_decay)
    return weights


def initialize_biases_cpu(
        name,
        shape):
    with tf.device("/cpu:0"):
        biases = tf.get_variable(
            name,
            shape,
            initializer=tf.constant_initializer(1.0),
            dtype=tf.float32)
    return biases


with tf.Graph().as_default():


    weights_one = initialize_weights_cpu("weights_one", [3, 3])
    weights_two = initialize_weights_cpu("weights_two", [3, 3])
    weights_three = initialize_weights_cpu("weights_three", [3, 3])
    biases_one = initialize_biases_cpu("biases_one", [3, 1])
    biases_two = initialize_biases_cpu("biases_two", [3, 1])
    biases_three = initialize_biases_cpu("biases_three", [3, 1])
    input_tensor = tf.placeholder(tf.float32, shape=[3, 1], name="input_tensor")
    label_tensor = tf.placeholder(tf.float32, shape=[3, 1], name="label_tensor")


    layer_one = tf.nn.sigmoid(
        tf.matmul(
            weights_one,
            input_tensor) + biases_one)
    layer_two = tf.nn.sigmoid(
        tf.matmul(
            weights_two,
            layer_one) + biases_two)
    layer_three = tf.matmul(
        weights_three,
        layer_two) + biases_three
    output_tensor = tf.nn.softmax(
        layer_three,
        dim=0,
        name="output_tensor")


    loss_tensor = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=layer_three,
            labels=label_tensor),
        name="loss_tensor")
    backprop_op = tf.train.AdamOptimizer(
        learning_rate=LEARNING_RATE).minimize(
            loss_tensor,
            var_list=[
                weights_one,
                weights_two,
                weights_three,
                biases_one,
                biases_two,
                biases_three
            ], name="backprop")
    init_op = tf.global_variables_initializer()


    with tf.Session() as session:
        session.run(init_op)
        session.run(backprop_op, feed_dict={
            input_tensor: [[0], [0], [0]],
            label_tensor: [[0], [0], [0]]
        })
        tf.train.write_graph(session.graph_def, BASEDIR, "tf_graph.proto", as_text=False)
