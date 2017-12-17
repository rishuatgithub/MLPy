# Tensor Flow basic - Rishu Shrivastava

#import the tf canonical lib
import tensorflow as tf

#A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
#Let's build a simple computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output.
#One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
