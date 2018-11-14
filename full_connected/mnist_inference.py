import tensorflow as tf
Input_Node = 588
Output_Node = 2
Layer1_Node = 500

def get_weight_varible(shape, regularizer):
    weight = tf.get_variable(
        "weight", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.01) 
    )
    if(regularizer is not None):
        tf.add_to_collection(
            "losses",
            regularizer(weight)
        )
    return weight
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weight = get_weight_varible([Input_Node, Layer1_Node], regularizer)
        biases = tf.get_variable('biases', [Layer1_Node],
                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + biases)
    with tf.variable_scope('layer2'):
        weight = get_weight_varible([Layer1_Node, Output_Node], regularizer)
        biases = tf.get_variable('biases', [Output_Node],
                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.softmax(tf.matmul(layer1, weight) + biases)
    return layer2
    