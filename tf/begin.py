# Tensor Flow basic - Rishu Shrivastava

#import the tf canonical lib
import tensorflow as tf
#from __future__ import print_function

#A computational graph is a series of TensorFlow operations arranged into a graph of nodes.
#Let's build a simple computational graph. Each node takes zero or more tensors as inputs and produces a tensor as an output.
#One type of node is a constant. Like all TensorFlow constants, it takes no inputs, and it outputs a value it stores internally.
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

# creating a session and printing the computational nodes
sess = tf.Session()
print(sess.run([node1, node2]))


# adding two nodes
node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# Using placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
# lambda function sort of output using placeholder
print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# More trivial operations using placeholder
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
# initialize the variables. Without which the variables will not be executed
init = tf.global_variables_initializer()
sess.run(init) #runs the Session

#Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# calculate the loss in linear_model - reduce_sum
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


# Adjust the variables to have a Zero loss
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
