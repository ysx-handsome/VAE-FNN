# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.autograd import Variable
import sys
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchsummary import summary
import random

torch.manual_seed(0)


#prefix = sys.argv[1]
regu = 1e-4 #float(sys.argv[2]) #for NN
regu2 = 1e-4 #float(sys.argv[3]) #for decoder weights
regu1 = -1 #for encoder weights
#regu = 0.0005
layer1 = 5 # decoder layers
nodes1 = 32 # decoder nodes
layer2 = 1 # encoder layers
nodes2 =  32 # encoder nodes
mc_samples = 200
batch_size = 512  # batch size is 512 for initial fit
epochs = 100
epsilon_std = 1.0
##noise = 0.2/np.sqrt(2) # for x, 0.1**2 #
prior_mean = 0
prior_var = 0.5
laplace = False
noisex = 0.05 # {0.05,0.1,0.2}
#beta = float(sys.argv[3])
sy = 0.2

"""
X_train = np.loadtxt("simu_nn2/nn_5_32_train_0.2_0.2_" +repeat +".txt")
X_train = X_train[0:n,:]
X_train[:,2:4] = X_train[:,0:2] + np.random.normal(size=(X_train.shape[0],2)) * noisex
X_train[:,4] = X_train[:,5] + np.random.normal(size=X_train.shape[0]) * sy
test_dat = np.loadtxt("simu_nn2/nn_5_32_test_" + repeat +".txt")
best_predict = np.zeros((test_dat.shape[0], 2))
"""

results = np.zeros((3, 7)) #NN_train_ise, NN_test_ise, train_ise, t`rain_ll, test_ise, NN_test_iae, test_iae
latent_dim = 2
ise_min = 1e8

class FNN_model(pl.LightningModule):
    def __init__(self,layer, nodes, activ ='relu', input_dim = 1, output_dim = 1, regu = -1, alpha = 0.3):
        super(FNN_model,self).__init__()
        self.layer=layer
        self.nodes=nodes
        self.activ=activ
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.regu=regu
        self.alpha=alpha
        self.bulld_model()


    def bulld_model(self):
        self.linear1 = nn.Linear(self.input_dim, self.nodes)
        self.linears = nn.ModuleList([nn.Linear(self.nodes,self.nodes) for i in np.arange(self.layer)])
        self.linear2 = nn.Linear(self.nodes, self.output_dim)

        if self.activ=='leakyrelu':
            self.activ1=nn.LeakyReLU()
            self.activs=nn.ModuleList([nn.LeakyReLU() for i in np.arange(self.layer)])
            self.activ2=nn.LeakyReLU()

        else:
            self.activ1=nn.ReLU()
            self.activs=nn.ModuleList([nn.ReLU() for i in np.arange(self.layer)])
            self.activ2=nn.ReLU()

    def forward(self,inputs):
        outs=[]
        outs.append(self.linear1(inputs))
        outs.append(self.activ1(outs[-1]))
        for i in range(self.layer):
            outs.append(self.linears[i](outs[-1]))
            outs.append(self.activs[i](outs[-1]))
        outs.append(self.linear2(outs[-1]))
        outs.append(self.activ2(outs[-1]))
        print(outs[-2][:20])
        return outs[-1]

    def configure_optimizers(self):
        ada = torch.optim.Adam(self.parameters(),lr=0.003, betas=(0.9, 0.999),eps=1e-8 ,weight_decay=0.0004, amsgrad=False)
        return ada

    def training_step(self, train_batch, batch_idx): #mse
        x,y=train_batch
        yPred=self.forward(x)
        yPred=yPred.squeeze(-1)
        loss=F.mse_loss(yPred,y)
        self.log('train_loss',loss)
        return loss

    def validation_step(self,val_batch,batch_id):
        pass


class changeNoise(Callback):
    def __init__(self, noisey, noise):
        super(changeNoise, self).__init__()
        self.noisey = noisey
        self.noise = noise

    def on_train_epoch_end(self,trainer, pl_module):

        if trainer.current_epoch > 19:
            self.noisey=Variable(trainer.logged_metrics.get('mise2'))


noisey = Variable(torch.FloatTensor([0.1]))
noise = Variable(torch.FloatTensor([noisex**2]))
noiseparam = changeNoise(noisey, noise)


class LossLayer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(*args, **kwargs)

    def forward(self, inputs):
        #z_mu0, z_log_var0, z, x_pred, x1, y1]
        mu, log_var, z, fz, w, y = inputs
        y = y.unsqueeze(1)
        w = w.unsqueeze(1)
        #print('noisey :',noisey)
        if laplace:
            reconstruction_loss = torch.sum(torch.square(y - fz), axis=-1) / noisey / 2 + torch.sum(torch.abs(w - z), axis=-1) / noise
        else:
            reconstruction_loss = torch.sum(torch.square(y - fz), axis=-1) / noisey / 2 + torch.sum(torch.square(w - z),axis=-1) / noise / 2
        prior_loss = 1.5 * torch.log(1 + torch.square(z - prior_mean) / prior_var / 2)  # v = 2
        prior_loss = torch.sum(prior_loss, axis=-1)
        post_loss = .5 * (torch.square(mu - z) / torch.exp(log_var) + log_var)
        post_loss = torch.sum(post_loss, axis=-1)
        return reconstruction_loss + prior_loss - post_loss


class WeightLayer(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(WeightLayer, self).__init__(*args, **kwargs)

    def forward(self, loss):
        log_weight =-loss.detach()
        log_weight -= torch.max(log_weight, axis=1, keepdims=True).values
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, axis=1, keepdims=True)
        return weight




class Lambda(pl.LightningModule):
    def __init__(self,LAMBDA):
        super(Lambda, self).__init__()
        self.lam=LAMBDA

    def forward(self, input):
        return self.lam(input)



class VAE(pl.LightningModule):

    def __init__(self,fnn_model,model_mu1,model_var1,model_mu2,model_var2,Lambda,LossLayer,WeightLayer):
        super(VAE,self).__init__()
        self.fnn_model=fnn_model
        self.model_mu1=model_mu1
        self.model_var1=model_var1
        self.model_mu2=model_mu2
        self.model_var2=model_var2
        self.Lambda=Lambda
        self.losslayer=LossLayer
        self.weightlayer=WeightLayer

    def configure_optimizers(self):
        ada = torch.optim.Adam(self.parameters(),lr=0.003, betas=(0.9, 0.999),eps=1e-8, weight_decay=0.0004, amsgrad=False)
        return ada

    def training_step(self,train_batch,batch_idx):
        X,y=train_batch
        z_mu0,z_log_var0,z,x_pred,x0_pred,x1,y1=self.predict(X)
        vae_loss = self.losslayer([z_mu0, z_log_var0, z, x_pred, x1, y1])
        weight = self.weightlayer(vae_loss)

        loss = torch.sum(vae_loss * torch.square(weight), axis=1)
        reconstruction_loss0 = (torch.sum(torch.square(y[:, :, latent_dim:(latent_dim + 1)] - x0_pred),
                                      axis=-1)) / noisey / 2
        reconstruction_loss0 = torch.sum(reconstruction_loss0 * (weight - torch.square(weight)), axis=1)
        yPred=self.forward(X)
        m2=self.mise2(y,yPred,weight)
        if laplace:
            loss=torch.mean(loss + reconstruction_loss0, axis=0) + torch.log(noise) * latent_dim + torch.log(noisey) / 2
        else:
            loss=torch.mean(loss + reconstruction_loss0, axis=0) + torch.log(noise) * latent_dim / 2 + torch.log(noisey) / 2

        self.log_dict({'train_loss':loss,'mise2':m2})
        return loss

    def mise2(self,yTrue,yPred,weight):

        var_y = torch.sum(torch.square(yTrue[:, :, latent_dim:(latent_dim + 1)] - yPred[:, :, latent_dim:(latent_dim + 1)]),
                      axis=-1)
        return torch.mean(torch.sum(var_y * weight, axis=1))

    def predict(self,X,run_type='train'):
        x, x2, x1, y1 = X
        eps1 = torch.tensor(data=torch.normal(mean=0,std=epsilon_std, size=(x.shape[0], mc_samples, 1)))
        eps2 = torch.tensor(data=torch.normal(mean=0,std=epsilon_std, size=(x.shape[0], mc_samples, 1)))
        z_mu1 = self.model_mu1(x)
        z_log_var1 = self.model_var1(x)
        z_sigma1 = self.Lambda(z_log_var1)
        #print(z_sigma1.shape,eps1.shape,z_mu1.shape)
        z_sigma1=z_sigma1.unsqueeze(-1)
        z_eps1 = torch.bmm(eps1,z_sigma1)
        z_mu1=z_mu1.unsqueeze(-1)
        z1 = torch.add(z_mu1, z_eps1)
        x2=x2.unsqueeze(1)

        input2 = torch.concat([x2.repeat(1,mc_samples,1), z1],dim=2)
        z_mu2 = self.model_mu2(input2)
        z_log_var2 = self.model_var2(input2)
        z_sigma2 = self.Lambda(z_log_var2)
        z_eps2 = torch.multiply(z_sigma2, eps2)
        z2 = torch.add(z_mu2, z_eps2)
        z = torch.concat([z1, z2],dim=2)
        x_pred = self.fnn_model(z)
        z_mu1 = z_mu1.repeat(1,mc_samples,1)
        z_log_var1 = z_log_var1.unsqueeze(1)
        z_log_var1 = z_log_var1.repeat(1,mc_samples,1)
        z_mu = torch.concat([z_mu1, z_mu2],dim=2)
        z_log_var = torch.concat([z_log_var1, z_log_var2],dim=2)
        z_mu0 = z_mu.detach()
        z_log_var0 = z_log_var.detach()
        z0 = z.detach()
        x0_pred = self.fnn_model(z0)

        if run_type=='predict':
            return z,x_pred
        else:
            return z_mu0,z_log_var0,z,x_pred,x0_pred,x1,y1

    def forward(self,X):
        z,x_pred=self.predict(X,'predict')
        output=torch.concat([z,x_pred],dim=2)
        return output

class FNN_Dataset(torch.utils.data.Dataset):

    def __init__(self,path='',excel=None):

        if excel:
            nplist=excel
        else:
            nplist = pd.read_excel(path)
        self.length=nplist.shape[0]
        nplist = nplist.T.to_numpy()
        self.x=nplist[latent_dim:2*latent_dim].T
        self.y=nplist[2*latent_dim]

        self.x=np.array(self.x)
        self.x=torch.FloatTensor(self.x)
        self.y=np.array(self.y)
        self.y=torch.FloatTensor(self.y)
        self.p=True
    def get_length(self):
        return self.length

    def __len__(self):

        return self.length

    def __getitem__(self, idx):

        return self.x[idx],self.y[idx]


class VAE_Dataset(torch.utils.data.Dataset):
    def __init__(self, path='',excel=None):
        if excel:
            nplist=excel
        else:
            nplist = pd.read_excel(path)
        self.length = nplist.shape[0]
        nplist = nplist.to_numpy()
        latent_dim=2

        self.x = nplist[:,latent_dim:(latent_dim*2+1)]
        self.x2 = nplist[:,(latent_dim+1):(latent_dim*2+1)]
        self.x1 = nplist[:,latent_dim:(latent_dim*2)]
        self.y1 = nplist[:,latent_dim*2]
        self.y = nplist[:,latent_dim:(latent_dim*2+1)]

        self.x = np.array(self.x)
        self.x = torch.FloatTensor(self.x)
        self.x2=np.array(self.x2)
        self.x2=torch.FloatTensor(self.x2)
        self.x1 = np.array(self.x1)
        self.x1 = torch.FloatTensor(self.x1)
        self.y1 = np.array(self.y1)
        self.y1 = np.expand_dims(self.y1,axis=1)
        self.y1 = torch.FloatTensor(self.y1)

        self.y = np.array(self.y)
        self.y = np.expand_dims(self.y,axis=1)
        self.y = torch.FloatTensor(self.y)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return [self.x[idx],self.x2[idx],self.x1[idx],self.y1[idx]],self.y[idx]

def get_test_date(path='',excel=None):
    if excel:
        nplist=excel
    else:
        nplist = pd.read_excel(path)
    nplist = nplist.T.to_numpy()
    latent_dim = 2
    x_test = nplist[latent_dim:latent_dim*2].T
    y_test = nplist[latent_dim*2]
    x_test = np.array(x_test)
    x_test = torch.FloatTensor(x_test)
    y_test = np.array(y_test)
    return x_test,y_test


def main():
    for i in np.arange(1):

        path='simu_train.xlsx'
        fnn_train_dataset=FNN_Dataset(path=path)
        train_loader=torch.utils.data.DataLoader(fnn_train_dataset,batch_size=5000,shuffle=True)
        model0 = FNN_model(layer1, nodes1, input_dim=latent_dim, activ='relu', regu=regu)
        model0_trainer=pl.Trainer(max_epochs=3)
        model0_trainer.fit(model0,train_loader)
        #print('fit_end')

        raise ValueError
        path='simu_test.xlsx'
        x_test,y_test=get_test_date(path)
        model0.eval()
        with torch.no_grad():
            pred=model0(x_test)
            pred=np.array(pred)
            print(pred)
            results[i,1]=np.mean((pred.transpose()-y_test)**2)
            #results[i,0]=

        #print('FNN MSE: ',results[i,1])

        #raise ValueError('stop!')
        global noisey
        if sy < 0.3:
            noisey = Variable(torch.FloatTensor([0.1]))
        else:
            noisey = Variable(torch.FloatTensor([0.2]))

        model1=FNN_model(layer1, nodes1, input_dim = latent_dim, activ='relu', regu = regu2)
        weights = model0.state_dict()
        #print(weights)
        model1.load_state_dict(weights)


        model_mu1=FNN_model(layer2, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1)
        model_var1=FNN_model(layer2-1, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1)
        model_mu2=FNN_model(layer2-1, nodes2, activ='relu', regu = regu1, input_dim = latent_dim+1, output_dim = 1)
        model_var2 = FNN_model(layer2 - 1, nodes2, activ='relu', regu=regu1, input_dim=latent_dim + 1, output_dim=1)
        Lambda=lambda t: torch.exp(.5*t)
        vae_loss=LossLayer()
        weight=WeightLayer()
        vae=VAE(model1,model_mu1,model_var1,model_mu2,model_var2,Lambda,vae_loss,weight)
        vae_trainer=pl.Trainer(callbacks=[noiseparam],max_epochs=3) #max_epochs=epoches

        vae_path='simu_train.xlsx'
        vae_train_dataset=VAE_Dataset(path=vae_path)
        vae_train_loader=torch.utils.data.DataLoader(vae_train_dataset,batch_size=512)
        vae_trainer.fit(vae,vae_train_loader)
        metric_history=vae_trainer.logged_metrics
        #print('metric',metric_history)
        results[i,3]=metric_history['train_loss']
        results[i,2]=metric_history['mise2']

        #print(model1.state_dict())

        path='simu_test.xlsx'
        x_test, y_test = get_test_date(path)
        model1.eval()
        with torch.no_grad():
            pred=model1(x_test)
            pred=np.array(pred)
            print(pred)
            results[i,4]=np.mean((pred.transpose()-np.array(y_test))**2)
            results[i,6]=np.mean(np.abs(pred.transpose()-y_test))
           # results[i,5]=np.mean(np.abs(pred.transpose()-y_test))

        #print('vae mse: ',results[i,4],'vae mae: ',results[i,6])



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


