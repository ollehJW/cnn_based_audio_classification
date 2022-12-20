import os
import time
import numpy as np
import torch


class SupervisedTraining(object):
    """Trainer for Supervised Learning
    Parameters
    ----------
    epoch : int
        max epoch. Defaults to 50.
    result_model_path : str
        Path to best model.

    model : torch.model
        Model
    train_dataloader : torch.DataLoader
        Dataloader for train dataset
    valid_dataloader : torch.DataLoader
        Dataloader for valid dataset
    criterion : torch.loss
        Loss function
    optimizer : torch.optim
        Optimizer

        
    """
    def __init__(self, 
                epoch=50, 
                result_model_path='', 
                ):

        # about training strategy
        self.epoch = epoch
        self.result_path = result_model_path

     # main function of train, **train loop
    def train(self, model, train_dataloader, val_dataloader, criterion, optimizer, gpu = True, seed = 1004):
        """
        Conduct supervised learning
        Parameters
        ----------
        model
            pytorch-base model
        train_dataloader
            train dataloader
        val_dataloader
            validation dataloader
        Returns
        --------
        """
        
        indexing_seed = np.random.randint(0, 100000, self.epoch)
        print("Indexing Seed: {}".format(indexing_seed))

        # update instance variables
        self.model = model 
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.gpu = gpu
        self.criterion = criterion
        self.optimizer = optimizer
        
        # update model 
        self._model = model

        # gpu setting
        if gpu:
            print("GPU can be used.")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self._model.to(device)

        else:
            device = torch.device("cpu")

        min_valid_acc = 0

        # training
        
        for e in range(self.epoch):
            train_loss = 0.0
            correct = 0
            total = 0
            model.train()     # Optional when not using Model Specific layer
            for data in self.train_dataloader:
                if torch.cuda.is_available():
                    images, labels = data['spectrogram'].float().to(device), data['target'].float().to(device)
        
                optimizer.zero_grad()
                target = model(images)
                loss = criterion(target,torch.argmax(labels, dim=1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(target, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).float().sum()
                total += labels.size(0)
            
            train_acc = correct / total
            correct = 0
            total = 0
            valid_loss = 0.0
            model.eval()     # Optional when not using Model Specific layer
            for data in self.val_dataloader:
                if torch.cuda.is_available():
                    data, labels = data['spectrogram'].float().to(device), data['target'].float().to(device)
        
                target = model(data)
                loss = criterion(target,torch.argmax(labels, dim=1))
                valid_loss = loss.item() * data.size(0)
                _, predicted = torch.max(target, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).float().sum()
                total += labels.size(0)
            valid_acc = correct / total

            print(f'Epoch {e+1} \t Training Loss: {train_loss / len(train_dataloader)} \t Training Acc: {train_acc} \t\t Validation Loss: {valid_loss / len(val_dataloader)} \t Validation Acc: {valid_acc}')
            if min_valid_acc < valid_acc:
                print(f'Validation Accuracy Increased({min_valid_acc:.6f}--->{valid_acc:.6f}) \t Saving The Model')
                min_valid_acc = valid_acc
                # Saving State Dict
                torch.save(model.state_dict(), os.path.join(self.result_path, 'Best_model.pth'))

            self.train_dataloader.dataset.file_indexing(seed = indexing_seed[e])
            self.val_dataloader.dataset.file_indexing(seed = indexing_seed[e])