
import numpy as np
import random
import torch
from torch.autograd import Variable
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# importing models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#import torchvision.transforms as transforms



class models:
    def __init__(self):
        #TODO: DEFINE INITIAL FUNCTION
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def baseline(X_train, y_train, X_test, y_test):


        np.unique(y_test, return_counts=True)[1][0] / len(y_test)
        #baseline error
        into_list = y_train.tolist()
        most_occurence = max(into_list, key=into_list.count)
        # 1 - error (because if incorrect we get 1)
        accuracy = 1 - np.sum((most_occurence - y_test)**2) / len(y_test)

        #f1 doesn't work, we don't have true positives since we only predict 0
        #y_pred = np.array([most_occurence for _ in y_test])
        #f1_s = f1_score(y_test, y_pred)
        f1_s = float('nan')
        return accuracy, f1_s


    def lr(X_train, y_train, X_test, y_test):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1_s = f1_score(y_test, y_pred)
        return accuracy, f1_s

    def gnb(X_train, y_train, X_test, y_test):
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1_s = f1_score(y_test, y_pred)
        return accuracy, f1_s



    def knnf(params):

        model = KNeighborsClassifier(**params)

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)
        #accuracy = model.score(X_test, y_test)
        #accuracy = metrics.accuracy_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)


        return f1



    def rf(X_train, y_train, X_test, y_test):
        model = RandomForestClassifier(max_depth=10, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = model.score(X_test, y_test)
        f1_s = f1_score(y_test, y_pred)
        return accuracy, f1_s



#### -------------------------------- ####

# models using mixuo
    def lr_mixup(X_train, y_train, X_test, y_test, augr, mixup, alpha, batch_size, lr_rate, epochs):

        # in general:
        #   - https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        # for mix up:
        #   - https://github.com/facebookresearch/mixup-cifar10

        input_dim = len(X_train[0,:])
        output_dim = len(np.unique(y_train))



        # initialize for pytorch
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.FloatTensor(y_train).float().long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.FloatTensor(y_test).float().long()




        trainloader = torch.utils.data.DataLoader(dataset=tuple(zip(X_train,y_train)), batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=tuple(zip(X_test,y_test)), batch_size=len(y_test), shuffle=False)
        # to create bathes more easily

        #Create model class
        class LogisticRegression(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(LogisticRegression, self).__init__()
                self.linear = torch.nn.Linear(input_dim, output_dim)

            def forward(self, x):
                outputs = torch.sigmoid( self.linear(x) )
                return outputs

        model = LogisticRegression(input_dim, output_dim)


        #instantiate the Loss Class

        if output_dim == 2:
            #binary
            criterion = torch.nn.BCEWithLogitsLoss()
            model = LogisticRegression(input_dim, output_dim-1) # also update model
        else:
            #multiclass
            criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy


        #instantiate the Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate) # multiclass



        # training the model
        def train(epoch):
            #print('\nEpoch: %d' % epoch)

            for batch_idx, (inputs, targets) in enumerate(trainloader): # importing batches
                #print('\nBatch: %d' % batch_idx)
                targets = targets.unsqueeze(1).float()

                # normal
                inputs, targets = map(Variable, (inputs, targets))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                ####

                # mixup
                # applying augmented ratio
                if mixup:
                    totalnr = inputs.shape[0]
                    augnr = round(augr*totalnr)
                    randomlist = random.sample(range(0, totalnr), augnr)
                    inputs, targets = inputs[randomlist,:], targets[randomlist,:]

                    # applying mixup
                    # TODO: define self
                    inputs, targets_a, targets_b, lam = self.mixup_data(inputs, targets, alpha)
                    # to make gradients
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                    optimizer.zero_grad() # initialize
                    outputs = model(inputs) # predicting
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam) # loss from mixup
                    loss.backward() # calculate gradient
                    optimizer.step() # parameter update


                # we can create progress bar if we want to

        def test(epochs):


            #only loop once through all y_test
            for batch_idx, (inputs, targets) in enumerate(testloader):
                targets = targets.unsqueeze(1).float() # necesary format
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets).data
                #_, y_pred = torch.max(outputs.data, 1) # multiclass

                #outputs = outputs.unsqueeze(1) # correct shape

                # into 0 and 1's
                y_pred = torch.round(outputs)

                total = targets.size(0)

                correct = (y_pred == targets).sum()
                accuracy = float(correct/total)
                f1_s = f1_score(y_test.detach().numpy(), y_pred.detach().numpy())

                #print("Epoch: {}. Batch: {}. Loss: {}. Accuracy: {}.".format(epoch, batch_idx, loss.item(), accuracy))

            return loss, accuracy, f1_s

        # start
        for epoch in range(0, epochs):
            train(epoch)
        loss, accuracy, f1_s = test(epoch)


        return accuracy, f1_s



    def nn_mixup(X_train, y_train, X_test, y_test, augr, mixup, alpha, batch_size, lr_rate, epochs):

        # in general:
        #   - https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
        # for mix up:
        #   - https://github.com/facebookresearch/mixup-cifar10

        input_dim = len(X_train[0,:])
        output_dim = len(np.unique(y_train))



        # initialize for pytorch
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.FloatTensor(y_train).float().long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.FloatTensor(y_test).float().long()




        trainloader = torch.utils.data.DataLoader(dataset=tuple(zip(X_train,y_train)), batch_size=batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(dataset=tuple(zip(X_test,y_test)), batch_size=len(y_test), shuffle=False)
        # to create bathes more easily

        #Create model class
        class Net(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(Net, self).__init__()
                self.fc1 = torch.nn.Linear(input_dim, 32)
                self.fc2 = torch.nn.Linear(32,64)
                self.fc3 = torch.nn.Linear(64,output_dim)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                outputs = torch.sigmoid( self.fc3(x) )
                return outputs




        model = Net(input_dim, output_dim)


        #instantiate the Loss Class

        if output_dim == 2:
            #binary
            criterion = torch.nn.BCEWithLogitsLoss()
            model = Net(input_dim, output_dim-1) # also update model
        else:
            #multiclass
            criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy


        #instantiate the Optimizer Class
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate) # multiclass



        # training the model
        def train(epoch):
            #print('\nEpoch: %d' % epoch)

            for batch_idx, (inputs, targets) in enumerate(trainloader): # importing batches
                #print('\nBatch: %d' % batch_idx)
                targets = targets.unsqueeze(1).float()

                # normal
                inputs, targets = map(Variable, (inputs, targets))
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                ####

                # mixup
                # applying augmented ratio
                if mixup:
                    totalnr = inputs.shape[0]
                    augnr = round(augr*totalnr)
                    randomlist = random.sample(range(0, totalnr), augnr)
                    inputs, targets = inputs[randomlist,:], targets[randomlist,:]

                    # applying mixup
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha)
                    # to make gradients
                    inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

                    optimizer.zero_grad() # initialize
                    outputs = model(inputs) # predicting
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam) # loss from mixup
                    loss.backward() # calculate gradient
                    optimizer.step() # parameter update


                # we can create progress bar if we want to

        def test(epochs):


            #only loop once through all y_test
            for batch_idx, (inputs, targets) in enumerate(testloader):
                targets = targets.unsqueeze(1).float() # necesary format
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets).data
                #_, y_pred = torch.max(outputs.data, 1) # multiclass

                #outputs = outputs.unsqueeze(1) # correct shape

                # into 0 and 1's
                y_pred = torch.round(outputs)

                total = targets.size(0)

                correct = (y_pred == targets).sum()
                accuracy = float(correct/total)
                f1_s = f1_score(y_test.detach().numpy(), y_pred.detach().numpy())

                #print("Epoch: {}. Batch: {}. Loss: {}. Accuracy: {}.".format(epoch, batch_idx, loss.item(), accuracy))

            return loss, accuracy, f1_s

        # start
        for epoch in range(0, epochs):
            train(epoch)
        loss, accuracy, f1_s = test(epoch)

        return accuracy, f1_s




