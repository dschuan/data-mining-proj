import math
import tensorflow as tf
import numpy as np
import pylab as plt
import numpy
import pandas
from pathlib import Path

sources = ["cleveland","long_beach","switzerland"]
#sources = ["cleveland"]



LABELS = ['age',
'sex',
'cp',
'trestbps',
'chol',
'fbs',
'restecg',
'thalach',
'exang',
'oldpeak',
'slope',
'ca',
'thal',
'prediction']

def neuralNet(testX, testY, trainX = [], trainY = [], useTrainedModel = False, modelName = "model"):
	my_file = Path("./" + modelName+ ".ckpt.index")
	if not my_file.is_file():
		useTrainedModel = False
	NUM_FEATURES = testX.shape[1]

	NUM_CLASSES = 2
	beta = 10e-12
	learning_rate = 0.06
	epochs = 800
	batch_size = 8
	num_neurons = 10
	seed = 123

	trainX = (trainX- np.mean(trainX, axis=0))/ np.std(trainX, axis=0)
	testX = (testX- np.mean(testX, axis=0))/ np.std(testX, axis=0)


	Y_data = np.zeros((trainX.shape[0], NUM_CLASSES))
	Y_data[np.arange(trainX.shape[0]), trainY] = 1 #one hot matrix
	trainY = Y_data

	Y_data = np.zeros((testX.shape[0], NUM_CLASSES))
	Y_data[np.arange(testX.shape[0]), testY] = 1 #one hot matrix
	testY = Y_data



	tf.reset_default_graph()
	# Create the model
	x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
	y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

	# Build the graph for the deep net

	weights_1 = tf.Variable(tf.truncated_normal([NUM_FEATURES, num_neurons], stddev=1.0/math.sqrt(float(NUM_FEATURES))), name='weights')
	biases_1  = tf.Variable(tf.zeros([num_neurons]), name='biases')
	u_1 = tf.add(tf.matmul(x, weights_1), biases_1)
	output_1 = tf.nn.sigmoid(u_1)

	weights_2 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
	biases_2  = tf.Variable(tf.zeros([num_neurons]), name='biases')
	u_2 = tf.add(tf.matmul(output_1, weights_2), biases_2)
	output_2 = tf.nn.sigmoid(u_2)

	weights_3 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
	biases_3  = tf.Variable(tf.zeros([num_neurons]), name='biases')
	u_3 = tf.add(tf.matmul(output_2, weights_3), biases_3)
	output_3 = tf.nn.sigmoid(u_3)

	weights_4 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
	biases_4  = tf.Variable(tf.zeros([num_neurons]), name='biases')
	u_4 = tf.add(tf.matmul(output_3, weights_4), biases_4)
	output_4 = tf.nn.sigmoid(u_4)

	weights_5 = tf.Variable(tf.truncated_normal([num_neurons, num_neurons], stddev=1.0/math.sqrt(float(num_neurons))), name='weights')
	biases_5  = tf.Variable(tf.zeros([num_neurons]), name='biases')
	u_5 = tf.add(tf.matmul(output_4, weights_5), biases_5)
	output_5 = tf.nn.sigmoid(u_5)


	weights_6 = tf.Variable(tf.truncated_normal([num_neurons, NUM_CLASSES], stddev=1.0 / np.sqrt(num_neurons), dtype=tf.float32), name='weights_2')
	biases_6 = tf.Variable(tf.zeros([NUM_CLASSES]), dtype=tf.float32, name='biases_2')
	logits  = tf.matmul(output_5, weights_6) + biases_6

	#reg_loss = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) + tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)



	#reg_loss = reg_loss*beta



	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
	loss = tf.reduce_mean(cross_entropy)# + reg_loss

	# Create the gradient descent optimizer with the given learning rate.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.minimize(loss)

	correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
	accuracy = tf.reduce_mean(correct_prediction)

	saver = tf.train.Saver()

	with tf.Session() as sess:
		if(useTrainedModel):
			saver.restore(sess, "./"  + modelName + ".ckpt")
			predictions = sess.run(logits,{ x: testX})
			return np.argmax(predictions, axis=1)

		else:
			sess.run(tf.global_variables_initializer())
			train_acc = []
			test_acc = []
			for i in range(epochs):
				for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
					train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})
				train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))



				if i % 100 == 0:
					print('iter %d: accuracy %g'%(i, train_acc[i]))
				test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

			save_path = saver.save(sess, "./" + modelName+ ".ckpt")
			print("Model saved in path: %s" % save_path)

			max_accuracy = max(test_acc)

			print("Final Test Accuracy = ",max_accuracy, "at epoch ", test_acc.index(max_accuracy))

			# plot learning curves
			plt.figure(1)
			plt.plot(range(epochs), train_acc)
			plt.xlabel(str(epochs) + ' iterations')
			plt.ylabel('Train accuracy')

			plt.figure(2)
			plt.plot(range(epochs), test_acc)
			plt.xlabel(str(epochs) + ' iterations')
			plt.ylabel('test accuracy')
			plt.show()

			predictions = sess.run(logits,{ x: testX})
			return np.argmax(predictions, axis=1)
