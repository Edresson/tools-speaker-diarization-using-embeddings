
import os
import sys
import numpy as np
seed = 0
import random
random.seed(seed)
np.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)

import tflearn
import tflearn.metrics

import librosa
import pickle
import math

mfcc_x = sys.argv[1]
mfcc_y = sys.argv[2]
mfcc_Pessoas_base2 = sys.argv[3]
mfcc_Cadastrados_base2  = sys.argv[4] 
mfcc_Pessoas_base1 = sys.argv[5]
mfcc_Cadastrados_base1  = sys.argv[6]

print( mfcc_Pessoas_base1)
mfcc_Pessoas_base3 = sys.argv[7]
mfcc_Cadastrados_base3  = sys.argv[8]

n_Entrada_taxa  = sys.argv[9]
n_Saida = sys.argv[10]
modelo = sys.argv[11]

with open(mfcc_x, 'rb') as f:
        X = pickle.load(f)
with open(mfcc_y, 'rb') as f:
        Y = pickle.load(f)
        
with open(mfcc_Pessoas_base2, 'rb') as f:
        pessoas_base2 = pickle.load(f)
        
with open(mfcc_Cadastrados_base2, 'rb') as f:
        cadastrados_base2 = pickle.load(f)

with open(mfcc_Pessoas_base1, 'rb') as f:
        pessoas_base1 = pickle.load(f)
        
with open(mfcc_Cadastrados_base1, 'rb') as f:
        cadastrados_base1 = pickle.load(f)


with open(mfcc_Pessoas_base3, 'rb') as f:
        pessoas_base3 = pickle.load(f)
        
with open(mfcc_Cadastrados_base3, 'rb') as f:
        cadastrados_base3 = pickle.load(f)

        
print(pessoas_base1,cadastrados_base1)
for i in range(len(X)):
        print(X[i][1],Y[i][1])
Aux = []

for i in  range(len(X)):
        Aux.append([])
        Aux[i] = X[i][0]
        
        
        

X= Aux        

Aux = []
for i in range(len(Y)):
    Aux.append(np.array(Y[i][0]).reshape(-1))

Y = Aux

#modelo

encoder = tflearn.input_data(shape=[None, 13,int(n_Entrada_taxa)])
encoder = tflearn.dropout(encoder,0.9)
encoder = tflearn.dropout(encoder,0.2)
encoder = tflearn.fully_connected(encoder, 40,activation='crelu')
decoder = tflearn.fully_connected(encoder, int(n_Saida), activation='linear')
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.0007,loss='mean_square', metric=None)#categorical_crossentropy
model = tflearn.DNN(net)
model.fit(X, Y, n_epoch=1000,run_id="auto_encoder", batch_size=128,shuffle=True, show_metric=False)

model.save(modelo)


encoding_model = tflearn.DNN(encoder, session=model.session)
a = encoding_model.predict([X[0]])




X = []

i=0

while i <len(cadastrados_base2):
                            
            X.append([encoding_model.predict([cadastrados_base2[i][0]])[0],cadastrados_base2[i][1]])
            i = i+1          



acertou = 0
tamanho = 0
V = []

for q in range(len(pessoas_base2)):            
                        
        a=[encoding_model.predict([pessoas_base2[q][0]])[0],pessoas_base2[q][1]]

        V.append(a)
                        
        

        
        
        
posI = 0

for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print("Base2(Locutores não conhecidos portugues:")
print("acertou: ", acertou,"de: ",tamanho) 
print(acertou/tamanho)


X = []

i=0


while i <len(cadastrados_base1):
                            
            X.append([encoding_model.predict([cadastrados_base1[i][0]])[0],cadastrados_base1[i][1]])
            i = i+1          



acertou = 0
tamanho = 0
V = []

for q in range(len(pessoas_base1)):            
                        
        a=[encoding_model.predict([pessoas_base1[q][0]])[0],pessoas_base1[q][1]]

        V.append(a)
                        
        

        
        
        
posI = 0

for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print('Base1(Locutores conhecidos):')
print("acertou: ", acertou,"de: ",tamanho) 
print(acertou/tamanho)


###base3
X = []

i=0

while i <len(cadastrados_base3):
                            
            X.append([encoding_model.predict([cadastrados_base3[i][0]])[0],cadastrados_base3[i][1]])
            i = i+1          



acertou = 0
tamanho = 0
V = []

for q in range(len(pessoas_base3)):            
                        
        a=[encoding_model.predict([pessoas_base3[q][0]])[0],pessoas_base3[q][1]]

        V.append(a)
                        
        

        
        
        
posI = 0

for j in range(len(V)):
        
        menordist = math.inf
        i =0
            
        while i < len(X):
                
                distancia = np.sqrt(sum([(xi-yi)**2 for xi,yi in zip(V[j][0],X[i][0])]))
                
                if distancia <  menordist:
                    menordist = distancia
                    posI= i
                i=i+1
        
        
        if(X[posI][1] == V[j][1]):
                acertou = acertou +1
                

       


tamanho = len(V)    
print('Base3(Locutores Librespeech):')
print("acertou: ", acertou,"de: ",tamanho) 
print(acertou/tamanho)



















