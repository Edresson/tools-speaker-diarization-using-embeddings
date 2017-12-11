#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import random
random.seed(5)

import tflearn
import os
import sys
import librosa
import speech_data as data
import numpy as np
import pickle
test_data = '/home/edresson/Pti-embbending/Encoder-MFCC/Automatizado/Bases/Segments-5s/Validacao/Base1/X/'
train_data = '/home/edresson/Pti-embbending/Encoder-MFCC/Automatizado/Bases/Segments-5s/Treino/Base1/X-2/'
working =''
# grab the speakers from the training directory
speakers = data.get_speakers(train_data)
number_classes = len(speakers)
#print(number_classes,speakers)
# create the MFCC arrays from the data for training
audio_files = os.listdir(working+train_data)

X = []
Y = []



        
try:
    
    with open('rna-treino_X-5s.txt', 'rb') as f:
           X = pickle.load(f)
       
    with open('rna-treino_Y-5s.txt', 'rb') as f:
          Y = pickle.load(f)

    with open('rna-teste_X-5s.txt', 'rb') as f:
           test = pickle.load(f)
       
    with open('rna-teste_Y-5s.txt', 'rb') as f:
          testY = pickle.load(f)
               
except:
    flag = 0 
    for f in audio_files:
        #a = data.one_hot_from_item(data.speaker(f), speakers)
        a = f.split('-')[0]
        if flag == 0:
            
            #print(a,f[:-4])
            flag =1
            
        Y.append(int(a))
        y, sr = librosa.load(train_data + f)
        X.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
    Aux =Y
    for i in range(len(Y)):
        aux = Y[i]
        Y[i] = [0]*number_classes
        Y[i][aux-1]= 1

    print(Aux[0],Y[0])
        
    test = []
    testY = []
    for f1 in os.listdir(test_data):
        y, sr = librosa.load(test_data + f1)
        a = f1.split('-')[0]
            
        testY.append(int(a))
        test.append(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    for i in range(len(testY)):
        aux = testY[i]
        testY[i] = [0]*number_classes
        testY[i][aux-1]= 1
    with open('rna-teste_X-5s.txt', 'wb') as f:
               pickle.dump(test, f)

    with open('rna-teste_Y-5s.txt', 'wb') as f:
               pickle.dump(testY, f)
               
    with open('rna-treino_X-5s.txt', 'wb') as f:
               pickle.dump(X, f)

    with open('rna-treino_Y-5s.txt', 'wb') as f:
               pickle.dump(Y, f)
               
        
        



# define the network and model
#tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)#iniciando grafico com os parametros num_cores: numero de cores da gpu ;gpu_memory_fraction: o uso de memória alocada 0.5 para 50%

tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)#iniciando grafico com os parametros num_cores: numero de cores da gpu ;gpu_memory_fraction: o uso de memória alocada 0.5 para 50%
net = tflearn.input_data(shape=[None, 13,216]) #Passando a data para o tf, 13 = n_mfcc; 44 = 1 segundo segment wave audio; None: o primeiro elemento deve ser None representando, o tamanho do lote.
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, number_classes*10,activation='elu')#64 neuronios para essa camada .
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, number_classes*19,activation='relu')

#net = tflearn.fully_connected(net, 64)#64 neuronios para essa camada .
#Outputs the input element scaled up by 1 / keep_prob. The scaling is so that the expected sum is unchanged.
 # 0.5 = probabilidade de que cada elemento seja mantido
net = tflearn.dropout(net, 0.6)
net = tflearn.fully_connected(net, number_classes, activation='softmax')#number_classes,  numero de locutores  para essa camada e 'softmax' nome ou função  de ativação para essa camada, default "linear"
#uma camada de regressão (a seguir à saída) é necessária como parte das operações de treinamento da estrutura.
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy',learning_rate=0.00005) # "adam" =  default gradient descent optimizer,loss= Função de perda utilizada por este otimizador de camada. Padrão: 'categorical_crossentropy'.

#criando a rede .
model = tflearn.DNN(net)
#trainando o modelo , x = input[Mfcc] , y = lista dos locutores, n_epoch = numero de etapas a serem executadas , show_metric=True: exibir a precisão a cada etapa , snapshot_step = 100 , terá 100 modelos instantâneos para cada etapa 
model.fit(X, Y, n_epoch=3000,shuffle=True, show_metric=True)



#salvar o modelo
#os.chdir('Modelos/')
#model.load('./Modelos/modelos/rna-tradicional-5s.tflearn')
model.save('rna-tradicional-1.tflearn')

# test the model using the testing directory

    
result = model.predict(test)
res=0
c = 0 
#aredondar o resultado
#print(result)
for i in range(len(result)):
    for x  in range(len(result[i])):
        result[i][x] = round(result[i][x])
        
        
#result  = [round(a) for a in x for x in result]

for f,r in zip(testY, result):
    #print(f,r)
    if np.all(f ==r):
         c = c + 1
    else:
        pass
        #print("Errou,era para ser:", f, " e foi: ",r)    


        
acc = float(c) / float(len(test))

print('Test set accuracy: %s' %str(acc))
