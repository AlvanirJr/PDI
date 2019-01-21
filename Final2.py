
import os


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.datasets as Dataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import numpy as np
from sklearn import preprocessing as pre
from torch.backends import cudnn
import cv2







epochs = 40
BATCH_SIZE = 130
LR = 0.01           


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(         # input shape (1, 28, 28) nosso (1, 186, 186)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=6,            # n_filters
                kernel_size=5,              # filter size
                stride=2,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28) nosso (16, 63, 63)
            nn.ReLU6(),                      # activation
#            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )#93
        self.conv2 = nn.Sequential(         
            nn.Conv2d(6, 10, 3, 1, 1),     
            nn.ReLU(),                     
#            nn.MaxPool2d(2),                
        )#86
        
        self.conv3 = nn.Sequential(         
            nn.Conv2d(10, 14, 3, 1, 1),     
            nn.ReLU(),                     
#            nn.MaxPool2d(2),                
        )#75
        
        self.conv4 = nn.Sequential(         
            nn.Conv2d(14, 20, 3, 2, 1),     
            nn.ReLU(),                     
#            nn.MaxPool2d(2),                
        )#30
        
        self.out1 = nn.Linear(20 * 46 * 46 , 50)   # fully connected layer, output 10 classes
        self.out2 = nn.Linear(50, 2)
        

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
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output1 = (self.out1(x))
        output2 = (self.out2(output1))
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
    matrix = [[0,0],[0,0]]

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
                        matrix[0][0] += 1
                    else:
                        matrix[0][1] += 1
                elif(labels_matrix[i][0] == 0):
                    
                    if(output[i][1] > 0.5):
                        matrix[1][1] += 1
                    else:
                        matrix[1][0] += 1
            
        
                        
            
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
        print("Lamp |", matrix[0][0],  "     |      ",matrix[0][1], " |")
        print("Laptop  |", matrix[1][0], "     |      " ,matrix[1][1] , " |")
            
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


def mostrar_camada(net, size):
    np_array = net
    np_array = np_array[0, :, :, :]

    # talvez dê erro nesse .cpu(), porque eu tava usando cuda, dai tinha que mudar de cuda pra cpu, dai acredito que só é retirar esse .cpu e continuar com o resto
    np_array = np_array.cpu().detach().numpy()
    for i in range(1):
        for j in range(size):
            plt.subplot2grid((1, size), (i, j)).imshow(np_array[i + j])

    plt.show()

def visualizar_camadas(data, rede, class_names, device ,num_images=6):

    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(data):
            inputs, labels = (batch['X'].float()), (batch['Y'].float())

            outputs = rede(inputs)
            
          
            print('Primeira camada')
            mostrar_camada(rede.camada1,6)
            print('Segunda camada')
            mostrar_camada(rede.camada2,10)
            print('Terceira camada')
            mostrar_camada(rede.camada3,14)
            print('Quarta camada')
            mostrar_camada(rede.camada4,20)
            
            
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j])
                
                #o = np.array(outputs.cpu().data[j])
                #plt.imshow(cv2.resize(inputs.cpu().data[j], (186, 186)))


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

optimizer = torch.optim.SGD(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rede_treinada = train(cnn)
classes = ["anchor", "barrel"]
visualizar_camadas(teste_loader, rede_treinada, classes, device)




















