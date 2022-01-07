import numpy as np
import pandas as pd
import math
import sys

batch_size=30
weights={}
bias={}
grad={}

def softmax(z):
  a=np.exp(z-np.max(z))
  b=a/a.sum(axis=0)
  return b

def sigmoid(z):
  return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

def initialize(ip, L1,L2, op):
  weights['W1']=np.random.randn(L1,ip)*0.01
  weights['W2']=np.random.randn(L2,L1)*0.01
  weights['W3']=np.random.randn(op,L2)*0.01
  bias['b1']=np.zeros((L1,1))
  bias['b2']=np.zeros((L2,1))
  bias['b3']=np.zeros((op,1))
    
def forward_prop(a,w,b,activation):
  z=np.dot(w,a)+b
  if activation=='sigmoid':
    g=sigmoid(z)
  if activation=='softmax':
    g=softmax(z)
  return g,(w,b)

def forward_propagation(x_in):
  memory=[]
  memory.append(x_in)
  a=x_in
  for i in range(0,2):
    a_prev=a
    a,mem=forward_prop(a,weights['W'+str(i+1)],bias['b'+str(i+1)],'sigmoid')
    memory.append(a)
  op,mem=forward_prop(a,weights['W3'],bias['b3'],'softmax')
  memory.append(op)
  return op,memory

def compute_cost(a,y):
  cost_func=np.multiply(y ,np.log(a)) + np.multiply((1-y), np.log(1-a))
  cost = (-1/batch_size) * np.sum(cost_func)
  return cost

def d_sigmoid(z):
  return z*(1-z)

def backpropagation(memory,y_actual):
  a3=memory[3]
  a2=memory[2]
  a1=memory[1]
  a0=memory[0]

  w1=weights['W1']
  b1=bias['b1']
  w2=weights['W2']
  b2=bias['b2']
  w3=weights['W3']
  b3=bias['b3']

  dz3=a3-y_actual
  dw3=(1/batch_size)*(np.dot(dz3,a2.T))
  db3=(1/batch_size)*np.sum(dz3, axis=1, keepdims=True)

  dz2=(1/batch_size)*np.dot(w3.T,dz3) * d_sigmoid(a2)
  dw2=(1/batch_size)*(np.dot(dz2,a1.T))
  db2=(1/batch_size)*np.sum(dz2, axis=1, keepdims=True)

  dz1=(1/batch_size)*np.dot(w2.T,dz2) * d_sigmoid(a1)
  dw1=(1/batch_size)*(np.dot(dz1,a0.T))
  db1=(1/batch_size)*np.sum(dz1, axis=1, keepdims=True)

  grad['dW1']=dw1
  grad['db1']=db1
  grad['dW2']=dw2
  grad['db2']=db2
  grad['dW3']=dw3
  grad['db3']=db3

def update_parameters(learning_rate):
  w1=weights['W1']
  b1=bias['b1']
  w2=weights['W2']
  b2=bias['b2']
  w3=weights['W3']
  b3=bias['b3']

  dw1=grad['dW1']
  db1=grad['db1']
  dw2=grad['dW2']
  db2=grad['db2']
  dw3=grad['dW3']
  db3=grad['db3']

  w2=w2-learning_rate*dw2
  b2=b2-learning_rate*db2
  w3=w3-learning_rate*dw3
  b3=b3-learning_rate*db3
  w1=w1-learning_rate*dw1
  b1=b1-learning_rate*db1
  

  weights['W1']=w1
  weights['b1']=b1
  weights['W2']=w2
  weights['b2']=b2
  weights['W3']=w3
  weights['b3']=b3

def predict(y_computed):
  return np.argmax(y_computed, axis=0)

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[targets]
    return res.reshape(list(targets.shape)+[nb_classes])

x_n=pd.read_csv('train_image.csv')
y_n=pd.read_csv('train_label.csv')

x_n[784]=y_n
x=x_n.sample(n=10000,replace=False)
x=x.set_index([pd.Index([i for i in range(len(x))])])
y=x[784]
x=x.iloc[:,0:784]


#Training
initialize(784,500,100,10)
for j in range(0,50):
  batch_size=30
  print("Iteration ", j)
  for i in range(0,len(x),batch_size):
      if (i+batch_size) > len(x):
        batch_size=len(x)-i
      op,memory=forward_propagation(x[i:i+batch_size].T)
      y_new=get_one_hot(y[i:i+batch_size],10).reshape(batch_size,10)
      print(compute_cost(op,np.transpose(y_new)))
      backpropagation(memory, np.transpose(y_new))
      update_parameters(0.6)
