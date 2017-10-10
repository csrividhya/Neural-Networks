'''
AUTHOR - Srividhya Chandrasekharan 
This is a Radial Basis function neural network to do function approximation of f(x) = 0.5 + 0.4(2*pi*x)
'''

import numpy as np
import math
import random
from random import uniform
from collections import namedtuple
import collections
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

pi = math.pi

input_pattern = namedtuple("input_pattern", "name, input, desired_op")


def y(x):
	angle = 2 * pi * x
	sine = math.sin(angle)
	h = (0.5 + (0.4 * sine))
	return h


def noise():
	return uniform(-0.1, 0.1)


def generate_input_data(N):
	data = []

	for i in range(1, N+1):
		x = uniform(0, 1)
		desired_op = (y(x) + noise())
		data.append(input_pattern(i, x, desired_op))
	return data


def pick_initial_centroids(k,data):
	n = 75
	count = 0
	pool = []  # to hold the generated random numbers
	median=[] #to hold the centroids


	while(count < k):
		random_index = random.randrange(0, n)

		while(random_index in pool):
			random_index = random.randrange(0, n)

		pool.append(random_index)
		median.append(data[random_index])
		count = count + 1

	print("\n Initial Centroids : " + str(median))
	return median

def find_dmax(median):
	dmax = eucledian(median[0][1],median[1][1])

	for i in range(0,len(median)):
		for j in range(0,len(median)):
			if i!=j:
				d = eucledian(median[i][1],median[j][1])
				if d >=dmax:
					dmax = d
	return dmax


def eucledian(x, y):
    temp = (y - x)
    return abs(temp)


def restimate_median(median, cluster,data):
    new_median = []
    count_cluster= [] #distribution of points in cluster
    n = len(median)

    for k in range(0, n):
        sum1 = [0.0] * n
        count = [0.0] * n
        avg1 = [0.0] * n

        new_record = []

        for i in range(0, len(data)):
        	if cluster[i] == k:
        		sum1[k] = sum1[k] + data[i].input
        		count[k] = count[k] + 1

        avg1[k] = (sum1[k] / count[k])
        count_cluster.append(count[k])

        new_record.append(k)
        new_record.append(avg1[k])
        new_record.append(999) #dummy desired output value

        new_median.append(new_record)

    return new_median,count_cluster


def assign_cluster(median,data):
	cluster = []
	for i in range(0, len(data)):
		min_dist = eucledian(data[i][1], median[0][1])
		centroid_index = 0
		for j in range(0, len(median)):
			distance = eucledian(data[i][1], median[j][1])
			if distance <= min_dist:
				min_dist = distance
				centroid_index = j
		cluster.append(centroid_index)
	return cluster

def determine_gaussian_centers(k,data):
	median = pick_initial_centroids(k,data)
	count_cluster=[]
	cluster=[]
	old_median = []
	iter = 0

	cluster = assign_cluster(median,data)
	iter=iter+1

	median,count_cluster= restimate_median(median, cluster,data)

	while(median != old_median):
		cluster=assign_cluster(median,data)
		old_median=median
		median,count_cluster = restimate_median(median, cluster,data)
		iter = iter + 1

	print("\n Number of iterations taken by K-Means are "+str(iter))
	print "\n Gaussian centers are:-"+str(median)
	return median,cluster,count_cluster

def compute_variance_using_dmax(median,k):
	dmax = find_dmax(median)
	variance = dmax/math.sqrt(k)
	variance = math.pow(variance,2)
	print "\n \n The common variance computed for all clusters are = ",variance
	return variance

def compute_variance(median,k,count_cluster,cluster,data):
	variance = []
	for j in range(0,k): #for each cluster
		sum = 0.0
		for i in range(0,len(data)): #for each input pattern
			if cluster[i] == j:
				sum = sum + eucledian(data[i][1],median[j][1])**2
		#sum = math.pow(sum,2)
		v = (sum/count_cluster[j])
		variance.append(v)
	print "\n \n The variances computed for the gaussian centers are = ",variance
	return variance


def get_mean_variance(variance,cluster_number):
	s = 0
	count = 0

	for i in range(0,len(variance)):
		if i!=cluster_number:
			s = s + variance[i]
			count=count+1
	return (s/count)

def adjust_variances(variance,common_variance,count_cluster):
	for i in range(0,len(count_cluster)):
		if count_cluster[i]==1.0:
			variance[i] = get_mean_variance(variance,i)
	return variance

def phi(sigma,center,x):
	temp = eucledian(center[1],x[1])
	return np.exp((-1/(2*sigma))*temp**2)

def generate_modified_input(N,variance,median,cluster,k,learning_rate,data):
	x = np.ones((N,k+1)) #x[i][j][0] - bias term
	
	i=0
	for i in range(0,N):
		for j in range(0,k):
			x[i][j] = phi(variance[j],median[j],data[i])
	return x

def linear_regression(x,w,learning_rate,data):
	epochs = 1
	w_old = w
	w_new = np.zeros(w.shape)
	
	while(epochs<=100):
		for i in range(0,len(data)):
			y = np.transpose(w_old).dot(x[i])
			d = data[i].desired_op
			temp = (learning_rate*(d-y)*x[i])
			w_new = w_old + temp
			w_old = w_new
		epochs = epochs+1
	
	return w_old

def y_after_regression(x,w,data):
	y=[]
	for i in range(0,len(data)):
			y.append(np.transpose(w).dot(x[i]))
	return y

	
def plotting_graph(k,learning_rate,y_got_from_regression):
	ip =[]
	op=[]

	for i in range(0, 1000):
		tempx = uniform(0, 1)
		ip.append(tempx)
		tempy = y(tempx)
		op.append(tempy)

	noisyx=[]
	noisyy=[]
	for i in range(0,len(data)):
		noisyx.append(data[i].input)
		noisyy.append(data[i].desired_op)

	x=[]
	for i in range(0,len(data)):
		x.append(data[i].input)
	
	plt.figure()
	plt.grid(True)
	plt.plot(x,y_got_from_regression,'ro',ms=8,label='Output generated by RBF',zorder=8)
	plt.plot(ip, op,'co',label='0.5+0.4sin(2*pi*x)', zorder=10)
	plt.plot(noisyx,noisyy,'g^',label='Noisy data used for regression', zorder=9)
	plt.title('Learning rate = '+str(learning_rate)+' number of centers = '+str(k))
	l = plt.legend()
	l.set_zorder(20)  # put the legend on top
	plt.show()


if __name__=="__main__":
	
	data = generate_input_data(75)

	N = 75 #fixed
	k = 4 #number of gaussian centers 
	
	learning_rate = np.array([0.01,0.02]) #eeta
	number_of_hidden_layers = np.array([2,4,7,11,16]) #k

	for i in range(0,len(number_of_hidden_layers)):
		k = number_of_hidden_layers[i]
		median,cluster,count_cluster= determine_gaussian_centers(k,data)
		
		common_variance = compute_variance_using_dmax(median,k)
		variance = compute_variance(median,k,count_cluster,cluster,data)
		variance = adjust_variances(variance,common_variance,count_cluster)
		
		#Uncomment this line to have a comman variance for the K gaussian centers
		'''
		for i in range(0,len(variance)):
			variance[i] =  common_variance
		print variance
		'''

		for j in range(0,len(learning_rate)):	
			x = generate_modified_input(N,variance,median,cluster,k,learning_rate,data)

			w = np.random.uniform(-1,2,size=(k+1))

			learned_weights = linear_regression(x,w,learning_rate[j],data)
			print "\n \n For learning_rate =",str(learning_rate[j])," and number of gaussian centers =  ",str(k)," Weights learned by regression are ",learned_weights

			y_got_from_regression = y_after_regression(x,learned_weights,data)
	
			plotting_graph(k,learning_rate[j],y_got_from_regression)