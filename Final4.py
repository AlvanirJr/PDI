# -*- coding: utf-8 -*-
# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as Dataset
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import preprocessing as pre
from torch.backends import cudnn
import cv2
from sklearn.metrics import confusion_matrix
#from pylab import *
 

#import schedule

file_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_path)


# torch.manual_seed(1)    # reproducible

# Hyper Parameters
epochs = 40         # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 130
LR = 0.01     # learning rate
DOWNLOAD_MNIST = False

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(         # input shape (1, 28, 28) nosso (1, 186, 186)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,            # n_filters
                kernel_size=3,              # filter size
                stride=2,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28) nosso (16, 63, 63)
            nn.ReLU(),                      # activation
#            ,    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
         #x = self.sigmoid(x)
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14) nosso (16, 63, 63)
            nn.Conv2d(6, 8, 3, 2, 1),     # output shape (32, 14, 14) nosso (32, 22, 22)
            nn.ReLU(),                      # activation
#            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 10, 5, 1, 1),
                nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 12, 3, 1, 1),        
                nn.ReLU(),
        )
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.out1 = nn.Linear(12 * 2 * 2, 30)   # fully connected layer, output 10 classes
        self.out2 = nn.Linear(30, 2)

        self.camada1 = None
        self.camada2 = None
        self.camada3 = None
        self.camada4 = None

    def forward(self, x):
        x = self.conv1(x)
        self.camada1 = x
        
        x = self.conv2(x)
        self.camada2 = x
        x = self.conv3(x)
        self.camada3 = x
        x = self.conv4(x)
        self.camada4 = x
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.pool3(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output1 = self.out1(x)
        output2 = self.out2(output1)
        return output2    # return x for visualization


class AppleCTDatasets(Data.Dataset):

    def __init__(self, lista, label, n_class=2):
        self.data = lista
        self.label = label
        self.n_class = n_class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img   = self.data[idx]
#        img        = scipy.misc.imread(img_name, mode='RGB')
        label_name = self.label[idx]
#        label      = np.load(label_name)
        #novo teste
        #label = img[:,:,0]-label

        sample = {'X': img, 'Y': label_name}

        return sample


def train(cnn):
    perda = []
    perdaval = []
    matriz = [[0,0], [0,0]]
    #roda as epocas para treinar a rede
    for epoch in range(epochs):

#        ts = time.time()

        ###Treina a rede utilizando a base de dados
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs, labels = (batch['X'].float()), (batch['Y'].float())


            outputs = cnn(inputs)
                        
            output = outputs.data.tolist()
            labels_matrix = labels.data.tolist()
            
            for i in range(len(output)):
                if(labels_matrix[i][0] == 1):
                    
                    if(output[i][0] > 0.5):
                        matriz[0][0] += 1
                    else:
                        matriz[0][1] += 1
                elif(labels_matrix[i][0] == 0):
                    
                    if(output[i][1] > 0.5):
                        matriz[1][1] += 1
                    else:
                        matriz[1][0] += 1
            perca = 0.
            loss = loss_func(outputs, labels)
            if(np.array(loss.item()).copy()) < 1:
                perca = np.array(loss.item()).copy()
            else:
                perca = 1
            perda.append(perca)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print()
            
        print("       |", "Lamp", " | ","Laptop", " |")
        print("____________________________")
        print("lamp |", matriz[0][0],  "     |      ",matriz[0][1], " |")
        print("laptop  |", matriz[1][0], "     |      " ,matriz[1][1], " |")
            
        print()
        ## Agora roda o valuation na rede com os pesos atuais, para checar o overfiting
        
        
        for iter, batch in enumerate(val_loader):

            inputs, labels = (batch['X'].float()), (batch['Y'].float())

            outputs = cnn(inputs)
            loss = loss_func(outputs, labels)
            if(np.array(loss.item()).copy()) < 1:
                perca = np.array(loss.item()).copy()
            else:
                perca = 1

            perdaval.append(perca)
            plt.plot(perda)
            plt.plot(perdaval)
            plt.show()

            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))
                #print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.data[0]))

        print("Finish epoch {}".format(epoch))
    return cnn
#        torch.save(fcn_model, model_path)

#        val(epoch)


def mostrar_camada(rede, size):
    np_array = rede
    np_array = np_array[0, :, :, :]

    # talvez dê erro nesse .cpu(), porque eu tava usando cuda, dai tinha que mudar de cuda pra cpu, dai acredito que só é retirar esse .cpu e continuar com o resto
    np_array = np_array.cpu().detach().numpy()
    for i in range(1):
        for j in range(size):
            plt.subplot2grid((1, size), (i, j)).imshow(np_array[i + j])

    plt.show()


def visualizar_camadas(data, rede, class_names, device ,num_images=12 ):
    was_training = rede.training
    rede.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(data):
            inputs, labels = (batch['X'].float()), (batch['Y'].float())
            #print(len(labels))
            #print(len(inputs))
            inputs = inputs.to(device)
            outputs = rede(inputs)
            #print(outputs)
            #matriz = matriz_confusion(labels, outputs)
            print('primeira camada')
            mostrar_camada(rede.camada1,6)
            print('segunda camada')
            mostrar_camada(rede.camada2,8)
            print('terceira camada')
            mostrar_camada(rede.camada3,10)
            print('quarta camada')
            mostrar_camada(rede.camada4,12)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.data[j])

                if images_so_far == num_images:
                    rede.train(mode=was_training)
        return rede.train(mode=was_training)
       # plot.show()
                
                
def matriz_confusion(labels, outputs):
    matriz  = [[0,0], [0,0]]
    labels = labels.data.tolist()
    outputs = outputs.data.tolist()
    #print(confusion_matrix(labels, outputs))
    #print(outputs)
    for i in range(len(outputs)):
        if(labels[i][0] == 1):
            #print(labels[i][0])
            if(outputs[i][0] > 0.5):
               # print(outputs[i][0])
                matriz[0][0] += 1
            else:
                matriz[0][1] += 1
        #print(label[0][])
        elif(labels[i][1] == 1):
            if(outputs[i][1] > 0.5):
                matriz[1][1] += 1
            else:
                matriz[1][0] += 1
        return matriz
 
base1 = []
base_label1 = []

base2 = []
base_label2 = []

for k in range(1, 61):
    if (k < 10):
        caminho = '101_ObjectCategories/lamp/image_000' + str(k) + '.jpg'
    #elif(k >= 100):
     #   caminho = '101_ObjectCategories/Faces/image_0' + str(k) + '.jpg'
    else:
        caminho = '101_ObjectCategories/lamp/image_00' + str(k) + '.jpg'

    img = cv2.imread(caminho, 0)
    normalize_img = pre.normalize(img)

    base1.append(normalize_img)
    base_label1.append([0., 1.])

for k in range(1, 81):
    if (k < 10):
        caminho = '101_ObjectCategories/laptop/image_000' + str(k) + '.jpg'
    #elif(k >= 100):
     #   caminho = '101_ObjectCategories/Faces_easy/image_0' + str(k) + '.jpg'
    else:
        caminho = '101_ObjectCategories/laptop/image_00' + str(k) + '.jpg'

    img = cv2.imread(caminho, 0)
    normalize_img = pre.normalize(img)

    base2.append(normalize_img)
    base_label2.append([1., 0.])


for i in range(len(base1)):
    base1[i] = (cv2.resize(base1[i], (186, 186)))

for i in range(len(base2)):
    base2[i] = (cv2.resize(base2[i], (186, 186)))

base_label1 = np.array(base_label1)
base_label2 = np.array(base_label2)

for i in range(len(base1)):
    base1[i] = np.array([base1[i]])

for i in range(len(base2)):
    base2[i] = np.array([base2[i]])

treino = np.concatenate((base1[0:int((len(base1)*0.8))], base2[0:int((len(base2)*0.8))]))
treino_label = np.concatenate((base_label1[0:int((len(base_label1)*0.8))], base_label2[0:int((len(base_label2)*0.8))]))

teste = np.concatenate((base1[int((len(base1)*0.8)):int((len(base1)*0.9))], base2[int((len(base2)*0.8)):int((len(base2)*0.9))]))
teste_label = np.concatenate((base_label1[int((len(base_label1)*0.8)):int((len(base_label1)*0.9))], base_label2[int((len(base_label2)*0.8)):int((len(base_label2)*0.9))]))

val = np.concatenate((base1[int((len(base1)*0.9)):], base2[int((len(base2)*0.9)):]))
val_label = np.concatenate((base_label1[int((len(base_label1)*0.9)):], base_label2[int((len(base_label2)*0.9)):]))



train_data = AppleCTDatasets(lista= treino, label= treino_label)
test_data = AppleCTDatasets(lista=teste, label=teste_label)
val_data = AppleCTDatasets(lista=val, label=val_label)


train_loader = Data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
teste_loader = Data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = Data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
use_gpu = False

cnn = CNN()
#print (cnn)
optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rede_treinada = train(cnn)
#print(rede_treinada)
classes = ["Lamp", "Laptop"]
visualizar_camadas(teste_loader, rede_treinada, classes, device)


"""

                if images_so_far == num_images:
                    net.train(mode=was_training)
                    return
        net.train(mode=was_training)


"""





















## following function (plot_with_labels) is for visualization, can be ignored if not interested
#from matplotlib import cm
#try: from sklearn.manifold import TSNE; HAS_SK = True
#except: HAS_SK = False; print('Please install sklearn for layer visualization')
#def plot_with_labels(lowDWeights, labels):
#    plt.cla()
#    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#    for x, y, s in zip(X, Y, labels):
#        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
#
#plt.ion()
## training and testing
#
#
#for epoch in range(EPOCH):
#    print("A")
#    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
#        print("Aqui")
#        print(b_x)
#        print(b_y)
#        output = cnn(b_x)[0]               # cnn output
#        loss = loss_func(output, b_y)   # cross entropy loss
#        optimizer.zero_grad()           # clear gradients for this training step
#        loss.backward()                 # backpropagation, compute gradients
#        optimizer.step()                # apply gradients
#
##        if step % 50 == 0:
##            test_output, last_layer = cnn(test_x)
##            pred_y = torch.max(test_output, 1)[1].data.numpy()
##            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
##            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
##            if HAS_SK:
##                # Visualization of trained flatten layer (T-SNE)
##                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
##                plot_only = 500
##                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
##                labels = test_y.numpy()[:plot_only]
##                plot_with_labels(low_dim_embs, labels)
#plt.ioff()
#
## print 10 predictions from test data
#test_output, _ = cnn(test_x[:10])
#pred_y = torch.max(test_output, 1)[1].data.numpy()
#print(pred_y, 'prediction number')
#print(test_y[:10].numpy(), 'real number')
