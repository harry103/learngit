import tensorflow as tf
import numpy as np
import math

def read_data(filename,x,y):
	f = open(filename,'r')
	data_str = f.read()

	num = 0
	str_pos = 0
	data_list = []

	while num < (x * y):
		if data_str[str_pos] < '0' or data_str[str_pos] > '9':
			str_pos = str_pos + 1
			continue
		else:
			data_list.append(int(data_str[str_pos]))
			str_pos = str_pos + 1
			num = num + 1
	#print (data_list)
	data = np.array(data_list)
	data = data.reshape(x,y)
	
	return data

def main():
	x = read_data('e10000.txt',128,32)
	w = read_data('pi300.txt',256,1)
	x = 0.1 * x
	w = 0.1 * w

	#print (w)
	#print (x)

	#x = x.reshape(32,128)

	# w1 32 * 128 * 256
	w1 = w[0:128]
	for i in range(8):
		w1 = np.append(w1,w1,axis=1)
	w1 = w1.reshape(256,128)
	#for i in range(5):
		#w1 = np.append(w1,w1,axis=0)
	#print (w1.shape)
	
	# w2 32 256 2^22
	w2 = w   
	for i in range(22):
		w2 = np.append(w2,w2,axis=1)
	w2 = w2.reshape(int(math.pow(2,22)),256)
	#for i in range(5):
		#w2 = np.append(w2,w2,axis=0)
	#print (w2.shape)
	
	total_step = int(math.pow(2,17))
	batch_size = 32
	y = tf.matmul(w1, x[:,0].reshape(128,1)) 
	step_now = 0

	optimizer = tf.train.GradientDescentOptimizer(0.5)
	
	for step in range(total_step):
			loss = np.zeros(1)
		for batch in range(batch_size):
			a1 = tf.matmul(w1, x[:,batch_size].reshape(128,1))
			y1 = tf.nn.relu(a1)

			y2 = tf.matmul(w2,y1)
			loss = tf.add(loss,-tf.log(y2[step_now]))
			step_now = step_now + 1

	loss = tf.divide(loss,batch_size)




if __name__ == "__main__":
    main()







