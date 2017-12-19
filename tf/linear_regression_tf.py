# Linear Regression model using tensorflow - Rishu Shrivastava
import tensorflow as tf

#Model parameters
W=tf.Variable([.3], dtype=tf.float32)
b=tf.Variable([-.3], dtype=tf.float32)
#Input and Output parameters
x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

linear_model=W*x + b

#calculate loss - Sum of squared error
loss = tf.reduce_sum(tf.square(linear_model - y))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
