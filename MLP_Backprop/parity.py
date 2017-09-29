'''AUTHOR: SRIVIDHYA CHANDRASEKHARAN
# 1 of CSE 5526: Intro to Neural Networks. Goal is to design a Multi-layer perceptron with 4 input elements
in the Input layer, 4 neurons(4 binary bits => 16 input data patterns) in the hidden layer and one unit in the
output layer. Goal is to train the network to identify ODD parity (having oddnumber of 1's in binary 
representation). O/P is 1 for odd parity and 0 otherwise!'''
import numpy as np
from scipy.special import expit
from collections import namedtuple
from Data import *
import pickle

LEN = 16  # number of input data patterns
THRESHOLD = 0.05
ALPHA = 0.9  # used in momentum calculation

def phi(v):  # Logistic Sigmoid function
	return (1.0 / (1.0 + np.exp(-1*v)))
	#return expit(v)

def phiDash(v):  # first derivatve of the logistic sigmoid
	sigmoid = phi(v)
	return (sigmoid * (1 - sigmoid))

def E(desiredOutput, output):  # cumulative errorFunction of a single data point
	return ((0.5) * pow((desiredOutput- output), 2))

def e(d, y):  # computes error in classification for a single data pattern
	return (d - y)

def v(x, w):  # weighted sum of x and w
	return np.dot(x, w)

def y(wxs):  # computes output by computing activation potential
	v = np.sum(wxs)
	activation = phi(v)
	return activation

def delta(DesiredOutput , Wkj , ActualOutput,v_k):
	return (e(DesiredOutput,ActualOutput)*phiDash(v_k))

def generalizedDelta(Yj, X , Wkj , Wji , deltaK, v_j):
	return (phiDash(v_j)*(deltaK*Wkj))

def momentum(previous_weight_update):
	return (ALPHA*previous_weight_update)

def gradientDescent(data,LEARNING_RATE):
	epochs = 0

	#nth iteration:-
	'''
	Wji = np.random.uniform(-1,2,size=(4,5))
	f1 = open("wji",'w')
	pickle.dump(Wji,f1)
	f1.close()
	'''

	Wji = np.array([[-0.25319573,  1.74377211,  1.44500589,  0.54399743,  0.12785769],
       [ 1.72207905,  1.14686209,  0.99662104,  1.68118961, -0.8927832 ],
       [-0.83097461, -0.09198941,  0.68408733,  0.5605566 ,  1.3274079 ],
       [ 0.61897873, -0.42775785, -0.27232791,  0.11333034,  0.51092689]])

	#print "Weights of hidden neurons : \n",Wji
	#print "weight vectors of the 4 hidden neurons are: ",Wji
	
	#Wkj = np.random.uniform(-1,2,size=5)
	'''
	f2 = open("wkj",'w')
	pickle.dump(Wji,f2)
	f2.close()
	'''
	
	Wkj = np.array([-0.67737842, -0.41639604,  1.72141107,  1.46240335,  0.28432497])

	#print "\n \n Weight of output neuron : \n",Wkj,"\n\n"

	#n+1th iteration:-
	Wji_new=np.zeros((4,5)) #b/w i/p and hidden layers
	Wkj_new=np.zeros(5) #b/w hidden and o/p layers

	
	Yj = np.ones(5) #outputs of the hidden layer and input to the output layer!
	Yk = 0 #outputs of the output layer - need to compare with desired o/p 

	nCorrect = 0 #number of input patterns whose Error is less than 0.05
	
	
	prev_gradientJ = np.zeros((4,5))
	prev_gradientK = 0.	
	

	while(nCorrect<16):#abs(errorCurrent - errorPrevious)>0.05):
		nCorrect = 0 #reinitialization to 0
		totalError = 0.

		for i in range(0,LEN): #explore each of the 16 training data patterns

			#for hidden layer
			X = np.array(data[i].input)
			#print "\n\n Input pattern = ",X,"\n\n "

			D = data[i].desired_op

			wx = Wji*X
			#print "\n Weighted sum bruhs!! ",wx,"\n"
			
			#for each of the 4 neurons, find output:-
			for j in range(0,4):
				Yj[j] = y(wx[j])
			
			#print "\n Output of 4 hidden neurons for pattern ",X," is ",  Yj," where 5th element is BIAS bruh!"
			
			#for output layer calc
			wx = Wkj*Yj
			#print "Wkj ",Wkj
			#print "\n Weighted sum of the output layer \n",wx

			Yk = y(wx)

			#print "\n Yk = ", Yk
			
			#gradient descent rule for o/p using delta rule
			v_k = np.sum(Wkj*Yj)
			deltaK = delta(D,Wkj,Yk,v_k)
			#print "DeltaK ",deltaK
			gradientK = (LEARNING_RATE*deltaK*Yj)
			#print "gradientK ",gradientK
			Wkj_new = Wkj + gradientK #+ (ALPHA * prev_gradientK)

			prev_gradientK = gradientK
			
			#print "\n Output layer: New weight vector bruh!!\n ",Wkj_new
			
			#gradient descent rule for all 4 neurons in hidden layer using generalized delta rule
			v_j=np.zeros(4) #weighted sum of terms
			deltaJ=np.zeros((5,1))
			gradientJ=np.zeros((4,5))
			
			for j in range(0,4):
				v_j[j] = np.dot(X,Wji[j])
				deltaJ[j] = generalizedDelta(Yj, X , Wkj[j] , Wji[j] , deltaK, v_j[j])
				gradientJ[j] = (LEARNING_RATE*deltaJ[j]*X)
				Wji_new[j] = Wji[j] + gradientJ[j] # + (ALPHA * prev_gradientJ[j])
			
			#print "\n deltaJ ",deltaJ
			#print "\n gradientJ ",gradientJ
			prev_gradientJ = gradientJ
			
			#print "\n New hidden layer's weight vector is",Wji_new

			#updating the weight vectors here!!
			Wji = Wji_new  
			Wkj = Wkj_new 
			
			#print "\n Updated Wji \n", Wji
			#print "\n\n Updated Wkj\n ", Wkj,"\n"
	
			#calculate the error function for each data point
			#D is a single number 
			# The actual output should be a weighted sum of wkj and yk values!!

			errorCurrent =  E(D,Yk)
			totalError+=errorCurrent
			if errorCurrent<=0.05:
				nCorrect+=1
			
		#print "At epoch #",epochs," the Cumulative Error function value = ",totalError
		epochs+=1

	return epochs
	
if __name__=="__main__":
	learning_rate = np.arange(0.05,0.55,0.05)
	train = BinaryData(sys.argv[1])

	print "Learning rate VS Epochs after including momentum"
	
	for i in range(0,len(learning_rate)):
		rate = learning_rate[i]
		epochs = gradientDescent(train.data,rate)
		print "\nLearning rate = "+str(rate)+" Epochs = "+str(epochs)	
	
	