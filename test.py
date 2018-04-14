import numpy as np
import random as rn
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

high_level_concepts = [
    0,  # Circle
    0,  # Vertical line
    0,  # Horizontal line
    0,  # Two lines
    0,  # Pointy
]

high_level_concepts = [
    [0.98, 0.1, 0.1, 0.1, 0.1],
    [0.2, 0.98, 0.3, 0.3, 0.5],
    [0.5, 0.1, 0.7, 0.2, 0.9],
    [0.5, 0.2, 0.2, 0.3, 0.7],
    [0.1, 0.9, 0.9, 0.8, 0.1],
    [0.4, 0.8, 0.8, 0.35, 0.3],
    [0.65, 0.3, 0.3, 0.1, 0.1],
    [0.1, 0.98, 0.98, 0.5, 0.6],
    [0.9, 0.2, 0.2, 0.2, 0.6],
    [0.9, 0.9, 0.2, 0.7, 0.3]
]
np_hlc = np.array(high_level_concepts, dtype=np.float32)

if 0:
    np_test = np.array([0.98, 0.1, 0.1, 0.1, 0.1])

    result = np.power(np_hlc - np_test, 2)

    result = np.add.reduce(result, 1)

    print(result)
    lala()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 5]))
b = tf.Variable(tf.zeros([5]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

y_see = (tf.nn.tanh(y) + 1) * 0.5
y = np_hlc - y_see
y_dist = tf.pow(y, 2)
y = tf.reduce_sum(y_dist, 1)
y = 1-tf.nn.softmax(y)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
    batch = mnist.train.next_batch(1)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("PREDICT")

correct = 0
random_correct = 0
for n in range(100):
    y_final = y.eval(feed_dict={x: [mnist.test.images[n]], y_: [mnist.test.labels[n]]})
    why = y_see.eval(feed_dict={x: [mnist.test.images[n]], y_: [mnist.test.labels[n]]})
    print("\n")
    print("I think it is: ", np.argmax(y_final))
    print("And it is a:   ", np.argmax(mnist.test.labels[n]))
    print("Becouse:")
    print("Circle:          ", why[0][0])
    print("Vertical line:   ", why[0][1])
    print("Horizontal line: ", why[0][2])
    print("Two lines:       ", why[0][3])
    print("Pointy:          ", why[0][4])

    if np.argmax(y_final) == np.argmax(mnist.test.labels[n]):
        correct += 1
    if np.argmax(y_final) == rn.randint(0, 9):
        random_correct +=1
print(correct)
print(random_correct)

# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# print(correct_prediction.eval(feed_dict={x: [mnist.test.images[0]], y_: [mnist.test.labels[0]]}))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
