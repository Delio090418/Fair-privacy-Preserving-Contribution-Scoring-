import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import torch.nn as nn
import math
from modelos import MLP,CNN,CNN_brain,LogisticRegressionModel, resnet,resnetpcam
from tqdm import tqdm
from torchvision.transforms import ToTensor
import random
import torch.optim as optim
from data_configuration import data_clients, commun_test
# from isic import non_iid_data,commun_test_isic 
# from PCam import non_iid_data_pcam, common_test_pcam
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Subset

# torch.cuda.is_available()
# # dev = "cpu"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
dev = torch.device(dev)
dev_four = dev #torch.device("cpu" if torch.cuda.is_available() else "cpu")

#[dev_one, dev_two, dev_three, dev_one, dev_two, dev_three,dev_one, dev_two, dev_three, dev_one, dev_two, dev_three,dev_one, dev_two, dev_three]

class Client:
    def __init__(self, model_type, dataset, device):
        self.device = device
        self.data_loader = dataset["train_loader"]
        self.test=dataset["test_loader"]
        self.model_type=model_type
        #self.testing_cli=dataset[1]
        self.model = self.init_model(model_type).to(self.device)
        self.criterion, self.optimizer=self.opt_and_cri(model_type)
        # self.criterion = self.op_cr[0]
        # self.optimizer = self.op_cr[1]

        self.client_size = len(self.data_loader.dataset) + len(self.test.dataset)


    def opt_and_cri(self,model_type):
        cr1=nn.CrossEntropyLoss()#nn.functional.cross_entropy
        # cr2=nn.BCELoss()
        op1=torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.001)#optim.SGD(self.model.parameters(), lr=0.01)
        # op2=optim.SGD(self.model.parameters(), lr=0.01)
        if model_type == "MLP":
            return cr1, op1 
        elif model_type == "CNN":
            return cr1, op1
        elif model_type == "CNN_brain":# or self.model_type == "LOG_S":
            return cr1, op1
        elif model_type == "resnet":# or self.model_type == "LOG_S":
            return cr1, op1
        elif model_type == "resnetpcam":
            return nn.BCEWithLogitsLoss(), op1
        else:
            raise ValueError("Unsupported Criterion and Optimizer")
        
   
    def init_model(self, model_type):
        if model_type == "MLP":
            return MLP()
        elif model_type == "CNN":
            return CNN()
        elif model_type == "CNN_brain":
            return CNN_brain()
        elif model_type == "resnet":
            return resnet()
        elif model_type == "resnetpcam":
            return resnetpcam(num_classes=1)
        else:
            raise ValueError("Unsupported model type")

    
    def fit(self, num_epochs):#fit(self, num_epochs test,plot=False):
        # Training loop
        #lenght=len(self.data_loader)
        self.model.to(self.device)  # Ensure model is on the correct GPU
        #batch_losses = []  # Store batch-wise loss
        #epoch_losses = [] 
        #val_losses = []
        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            #epoch_loss = 0  # Accumulate epoch loss
            #num_batches = len(self.data_loader)
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                if self.model_type == "resnetpcam":
                    target = target.view(output.shape)
                    target = target.float()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                # batch_losses.append(loss.item())  # Store batch loss
                # epoch_loss += loss.item()
            # epoch_losses.append(epoch_loss / num_batches)
            # # Evaluate on Validation Set
            # self.model.eval()  # Set model to evaluation mode
            # epoch_val_loss = 0

            # with torch.no_grad():
            #     for inputs, targets in test:
            #         inputs, targets = inputs.to(dev), targets.to(dev)
            #         outputs = self.model(inputs).squeeze(dim=1)
            #         loss_val = self.criterion(outputs, targets)
            #         epoch_val_loss += loss_val.item()

            # val_losses.append(epoch_val_loss / len(test))  # Store average validation loss

        # if plot==True:
        #     plt.figure(figsize=(8, 4))
        #     plt.plot(range(1, num_epochs + 1), epoch_losses, label="Training Loss", color="blue")
        #     plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", color="red")            
        #     plt.xlabel("Epochs")
        #     plt.ylabel("Loss")
        #     plt.title("Epoch-wise Training Loss")
        #     plt.legend()
        #     plt.show()

    # def _evaluate_on_device(self, model_state_dict, test_loader, model_type, device):
    #     model = self.init_model(model_type).to(device)
    #     model.load_state_dict(model_state_dict)
    #     model.eval()
    #     correct = 0
    #     total = 0

    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             data, target = data.to(device), target.to(device)
    #             outputs = model(data)
    #             if model_type == "resnetpcam":
    #                 outputs = torch.sigmoid(outputs).squeeze()
    #                 predicted = (outputs >= 0.5).float()
    #                 target = target.float().squeeze()
    #             else:
    #                 _, predicted = torch.max(outputs, dim=1)
    #             correct += (predicted == target).sum().item()
    #             total += target.size(0)

    #     return correct, total
    
    # def evaluation_parallel(self, test_set):
    #     num_samples = len(test_set.dataset)
    #     device_count = torch.cuda.device_count()
    #     indices = list(range(num_samples))
    #     chunk_size = num_samples // device_count
    #     devices = [f"cuda:{i}" for i in range(device_count)]

    #     model_state = self.model.state_dict()  # share weights to each worker
    #     loaders = []

    #     for i in range(device_count):
    #         subset = Subset(test_set.dataset, indices[i*chunk_size : (i+1)*chunk_size])
    #         loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
    #         loaders.append(loader)

    #     results = []
    #     with ThreadPoolExecutor(max_workers=device_count) as executor:
    #         futures = [
    #             executor.submit(self._evaluate_on_device, model_state, loader, self.model_type, devices[i % len(devices)])
    #             for i, loader in enumerate(loaders)
    #         ]
    #         results = [f.result() for f in futures]

    #     correct = sum(r[0] for r in results)
    #     total = sum(r[1] for r in results)
    #     return correct / total if total > 0 else 0.0
    
    def evaluation(self,test_set):
        device = next(self.model.parameters()).device
        self.model.to(device)  # Get model device
        self.model.eval()
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for data, target in test_set:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                if self.model_type == "resnetpcam":
                    outputs = torch.sigmoid(outputs).squeeze()  # should be shape [batch_size]
                    predicted = (outputs >= 0.5).float()
                    target = target.float().squeeze()
                else: 
                    _, predicted = torch.max(outputs, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:
                    break
        return correct/total
    
    def get_model_grads(self):
        return [param.grad for param in self.model.parameters()]
    
    # def evaluate(self, test_loader, threshold=5000):
    #     if len(test_loader.dataset) < threshold:
    #         return self.evaluation(test_loader)
    #     else:
    #         return self.evaluation_parallel(test_loader)
        
    
    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def load_state_dict(self, weights):
        self.model.load_state_dict(weights)

    def parameters(self):
        return self.model.parameters()

class federation:
    def __init__(self, model_type, dataset_name, num_clients, partition_type='N-IID',alpha=0.5,dev_four=dev):
        self.alpha=alpha
        self.num_clients=num_clients
        self.model_type=model_type
        self.dataset_name=dataset_name
        self.partition_type=partition_type
        #self.sizes=client_sizes
        self.model = self.init_model(model_type).to(dev_four)
        self.datos=self.load_dataset(self.dataset_name,self.num_clients)
        self.datasets=self.datos[0]
        self.test_global=self.datos[1]
        self.lista_clientes=[Client(model_type,self.datasets[i],dev) for i in range(self.num_clients)]



    def load_dataset(self, dataset_name, n_cli):
        if dataset_name == "mnist":
            data_mnist=data_clients(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            #list_loader=[data_mnist[i]["test_loader"] for i in data_mnist.keys()]
            test_mnist=commun_test(dataset_name)
            return data_mnist, test_mnist
        elif dataset_name == "cifar10":
            data_cifar=data_clients(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            #list_loader=[data_cifar[i]["test_loader"] for i in data_cifar.keys()]
            test_cifar=commun_test(dataset_name)
            return data_cifar, test_cifar
        elif dataset_name=="BRAIN":
            data_brain =data_clients(data_name=dataset_name,num_clients=n_cli, alpha=self.alpha,partition_type=self.partition_type)
            test_brain = commun_test(dataset_name)
            return data_brain, test_brain
        # elif dataset_name=="isic":
        #     data=non_iid_data(n_clients=self.num_clients, alpha=self.alpha)
        #     test=commun_test_isic(data)
        #     return data, test
        # elif dataset_name=="pcam":
        #     data=non_iid_data_pcam(n_clients=self.num_clients, alpha=self.alpha)
        #     test=common_test_pcam(data)
        #     return data, test
        else:
            raise ValueError("Unsupported dataset")
        


    def init_model(self, model_type):
        if model_type == "MLP":
            return MLP()
        elif model_type == "CNN":
            return CNN()
        elif model_type == "resnetpcam":
            return resnetpcam(num_classes=1)
        elif model_type == "CNN_brain":
            return CNN_brain()
        elif model_type == "resnet":
            return resnet(num_classes=8)
        else:
            raise ValueError("Unsupported model type")


    def total_size(self):
        size = 0
        for client in self.lista_clientes:
            size += client.client_size
        return size


    def aggregate(self, List_clients):
        pesos_cliente = [client.parameters() for client in List_clients]
        data = []
        for params_clients in zip(*pesos_cliente):
            data.append(sum(param.data for param in params_clients) / len(List_clients))
        return data

    def weighted_aggre(self, List_clientes, device=dev_four):
        List_clients = [client.model.to(device) for client in List_clientes]
        pesos_cliente = [client.parameters() for client in List_clients] 
        total_data = sum(cliente.client_size for cliente in List_clientes)
        data_amount = [participant.client_size for participant in List_clientes]
        data = []
        for params_clients in zip(*pesos_cliente):
            order_para = []
            for param, amout in zip(params_clients, tuple(data_amount)):
                weighted_param = (amout / total_data) * param.data
                order_para.append(weighted_param.to(device))  # Ensure all tensors are on dev_four
            data.append(sum(order_para))
        return data
    


    # def federated_averaging(self, num_iterations, conjunto_cli, num_epochs=20):
    #     from concurrent.futures import ProcessPoolExecutor
    #     conj_clientes = [self.lista_clientes[i] for i in conjunto_cli]
    #     iteration = 0
    #     while iteration <= num_iterations - 1:
    #         # Parallel training
    #         with ProcessPoolExecutor(max_workers=len(set(devices))) as executor:
    #             futures = []
    #             for client in conj_clientes:
    #                 client.model.to(client.device)
    #                 futures.append(executor.submit(client.fit, num_epochs))
    #             for f in futures:
    #                 f.result()
    #         tqdm.write(f"fed iteration {iteration}")
    #         params = self.weighted_aggre(conj_clientes)
    #         self.set_parameters(params)
    #         iteration += 1

    def federated_averaging(self, num_iterations, conjunto_cli,num_epochs=20):
        conj_clientes = [self.lista_clientes[i] for i in conjunto_cli]
        iteration = 0
        while iteration <= num_iterations - 1:
            for client in conj_clientes:
                weights_modelo = self.model.state_dict()
                client.load_state_dict(weights_modelo)
                client.fit(num_epochs)
            tqdm.write(f"fed iteration {iteration}")
            params = self.weighted_aggre(conj_clientes)
            self.set_parameters(params)
            iteration += 1
        #return self.evaluacion(test_set)
    
    # def _evaluate_on_device(self, model_state_dict, test_loader, model_type, device):
    #     model = self.init_model(model_type).to(device)
    #     model.load_state_dict(model_state_dict)
    #     model.eval()
    #     correct = 0
    #     total = 0

    #     with torch.no_grad():
    #         for data, target in test_loader:
    #             data, target = data.to(device), target.to(device)
    #             outputs = model(data)
    #             if model_type == "resnetpcam":
    #                 outputs = torch.sigmoid(outputs).squeeze()
    #                 predicted = (outputs >= 0.5).float()
    #                 target = target.float().squeeze()
    #             else:
    #                 _, predicted = torch.max(outputs, dim=1)
    #             correct += (predicted == target).sum().item()
    #             total += target.size(0)

    #     return correct, total
    
    # def evaluation_global_parallel(self, test_set):
        # num_samples = len(test_set.dataset)
        # device_count = torch.cuda.device_count()
        # indices = list(range(num_samples))
        # chunk_size = num_samples // device_count
        # devices = [f"cuda:{i}" for i in range(device_count)]

        # model_state = self.model.state_dict()  # share weights to each worker
        # loaders = []

        # for i in range(device_count):
        #     subset = Subset(test_set.dataset, indices[i*chunk_size : (i+1)*chunk_size])
        #     loader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)
        #     loaders.append(loader)

        # results = []
        # with ThreadPoolExecutor(max_workers=device_count) as executor:
        #     futures = [
        #         executor.submit(self._evaluate_on_device, model_state, loader, self.model_type, devices[i % len(devices)])
        #         for i, loader in enumerate(loaders)
        #     ]
        #     results = [f.result() for f in futures]

        # correct = sum(r[0] for r in results)
        # total = sum(r[1] for r in results)
        # return correct / total if total > 0 else 0.0
    
        
    def evaluation_global(self,test_set,device):
        self.model.to(device)    # Get model device 
        self.model.eval()
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for data, target in test_set:
                data, target = data.to(device), target.to(device)
                outputs = self.model(data)
                if self.model_type== "resnetpcam":
                    outputs = torch.sigmoid(outputs).squeeze()  # should be shape [batch_size]
                    predicted = (outputs >= 0.5).float()
                    target = target.float().squeeze()
                else: 
                    _, predicted = torch.max(outputs, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                i+=1
                if i > 5:
                    break
        return correct/total
    
    # def evaluate_global(self, test_loader, threshold=5000):
    #     if len(test_loader.dataset) < threshold:
    #         return self.evaluation_global(test_loader)
    #     else:
    #         return self.evaluation_global_parallel(test_loader)
    
   



    def set_parameters(self,params):
        for model_parametro, param in zip(self.model.parameters(), params):
            model_parametro.data = param

    def reset_parameters(self):
        # self.normalizer.reset_parameters()
        # self.layers.reset_parameters()
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def parametros(self):
        data=[param.data for param in self.model.parameters()]
        return data


if __name__ == "__main__":
    # data=non_iid_data(6,0.2)[0]
    # #data_for_clients("",3,0.5,partition_type="IID")
    # #data=dirichlet_partition(3, 0.5)
    # cli=Client("resnet",data)
    # cli.fit(1)
    # eval=cli.evaluation(cli.test)
    # print(eval)
    fed=federation("CNN","cifar10",3,dev_four=dev)
    clients=fed.lista_clientes
    #fed.federated_averaging(1,[0,1,2],1)
    clients[0].fit(1)
    eva_1=clients[0].evaluation(clients[0].test)
    #eva_2=fed.evaluation_global(fed.test_global)
    print(eva_1)
    #print(eva_2)




    # mlp = Client("MLP","MNIST",0)
    # mlp.fit(10)
    # test_set=mlp.testing_cli
    # eva=mlp.evaluation(test_set)
    # print(eva)