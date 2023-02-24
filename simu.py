import pandas as pd
import numpy as np
from model import  *
from simu_generator import simu_generator
import random

random.seed(0)

simu_times=5
n=8000
dim=2
sigma_y=0.2



def train_loop(train_df,test_df):
    fnn_train_dataset = FNN_Dataset(excel=train_df)
    train_loader = torch.utils.data.DataLoader(fnn_train_dataset, batch_size=512)
    model0 = FNN_model(layer1, nodes1, input_dim=latent_dim, activ='relu', regu=regu)
    model0_trainer = pl.Trainer(max_epochs=100)
    model0_trainer.fit(model0, train_loader)
    print('fit_end')
    x_test, y_test = get_test_date(test_df)
    model0.eval()
    with torch.no_grad():
        pred = model0(x_test)
        pred = np.array(pred)
        results[i, 1] = np.mean((pred.transpose() - y_test) ** 2)
        # results[i,0]=

    print('FNN MSE: ', results[i, 1])

    # raise ValueError('stop!')
    global noisey
    if sy < 0.3:
        noisey = Variable(torch.FloatTensor([0.1]))
    else:
        noisey = Variable(torch.FloatTensor([0.2]))

    model1 = FNN_model(layer1, nodes1, input_dim=latent_dim, activ='relu', regu=regu2)
    weights = model0.state_dict()
    model1.load_state_dict(weights)

    model_mu1 = FNN_model(layer2, nodes2, activ='relu', regu=regu1, input_dim=latent_dim + 1, output_dim=1)
    model_var1 = FNN_model(layer2 - 1, nodes2, activ='relu', regu=regu1, input_dim=latent_dim + 1, output_dim=1)
    model_mu2 = FNN_model(layer2 - 1, nodes2, activ='relu', regu=regu1, input_dim=latent_dim + 1, output_dim=1)
    model_var2 = FNN_model(layer2 - 1, nodes2, activ='relu', regu=regu1, input_dim=latent_dim + 1, output_dim=1)
    Lambda = lambda t: torch.exp(.5 * t)
    vae_loss = LossLayer()
    weight = WeightLayer()
    vae = VAE(model1, model_mu1, model_var1, model_mu2, model_var2, Lambda, vae_loss, weight)
    vae_trainer = pl.Trainer(callbacks=[noiseparam], max_epochs=epochs)  # max_epochs=epoches

    vae_path = 'simu_train.xlsx'
    vae_train_dataset = VAE_Dataset(vae_path)
    vae_train_loader = torch.utils.data.DataLoader(vae_train_dataset, batch_size=512)
    vae_trainer.fit(vae, vae_train_loader)
    metric_history = vae_trainer.logged_metrics
    print('metric', metric_history)
    results[i, 3] = metric_history['train_loss']
    results[i, 2] = metric_history['mise2']

    path = 'simu_test.xlsx'
    x_test, y_test = get_test_date(path)
    model1.eval()
    with torch.no_grad():
        pred = model1(x_test)
        pred = np.array(pred)
        results[i, 4] = np.mean((pred.transpose() - np.array(y_test)) ** 2)
        results[i, 6] = np.mean(np.abs(pred.transpose() - y_test))
    # results[i,5]=np.mean(np.abs(pred.transpose()-y_test))

    print('vae mse: ', results[i, 4], 'vae mae: ', results[i, 6])


for i in range(simu_times):

    generator=simu_generator(dim,n,sigma_y)
    x, w, y, yy=generator.generate_data()
    path=''
    simu_data=pd.DataFrame()

    simu_data['x1']=x[:,0]
    simu_data['x2']=x[:,1]
    simu_data['w1']=w[:,0]
    simu_data['w2']=w[:,1]
    simu_data['y']=yy
    simu_data['pure_y']=y



