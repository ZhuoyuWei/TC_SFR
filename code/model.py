from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import numpy as np
import pandas as pd
import numpy as np
import os
import sys

def read_ground_truth(filename):
    df=pd.read_csv(filename)
    print(df.columns)
    #print('######')
    locations=set(df["LocationID"])
    #print(locations)

    df["DateTime"]=pd.to_datetime(df["DateTime"])
    #print(df["DateTime"])
    df["Value"].astype('float')

    loc2values={}

    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        #print('$$$$$$$$$$$')
        #print(sub_df.head())

        loc2values[loc]=sub_df

    return df, loc2values

def read_ground_truth_withoutfloat(filename):
    df=pd.read_csv(filename)
    print(df.columns)
    #print('######')
    locations=set(df["LocationID"])
    #print(locations)

    df["DateTime"]=pd.to_datetime(df["DateTime"])
    #print(df["DateTime"])
    #df["Value"].astype('float')

    loc2values={}

    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        #print('$$$$$$$$$$$')
        #print(sub_df.head())

        loc2values[loc]=sub_df

    return df, loc2values



def convert_local2values(df):
    locations=set(df["LocationID"])
    loc2values={}
    for loc in locations:
        sub_df=df[df["LocationID"]==loc]
        sub_df=sub_df.sort_values(by='DateTime',ignore_index=True)
        loc2values[loc]=sub_df
    return loc2values


from tqdm import tqdm, trange


class RNNForTimeSeries(nn.Module):
    def __init__(self, hidden_size, layer_num):
        super(RNNForTimeSeries, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=layer_num,
                           batch_first=True)
        self.dense = nn.Linear(hidden_size * layer_num, 1)
        self.loss_fct = nn.MSELoss()

    def _step(self, input, batch_size):

        _, (hn, cn) = self.rnn(input)
        hn = hn.reshape((batch_size, -1))

        logits = self.dense(hn)

        return logits

    '''
    def forward(self,input,target=None):
        input_size=input.size()
        batch_size=input_size[0]

        _,(hn,cn)=self.rnn(input)
        hn=hn.reshape((batch_size,-1))

        logits=self.dense(hn)

        if target is not None:
            loss=self.loss_fct(logits,target)
            return logits,loss
        else:
            return logits
    '''

    def forward(self, input, target=None, max_length=0):
        input_size = input.size()
        batch_size = input_size[0]

        input = input.unsqueeze(-1).float()

        if target is not None:
            target_size = target.size()
            max_length = target_size[-1]
            target = target.unsqueeze(-1).float()

        assert max_length > 0

        outputs = []

        keep_input = input
        total_loss = 0.
        for i in range(max_length):
            # print('debug {}'.format(input.shape))
            logits = self._step(input, batch_size)
            outputs.append(logits)
            if target is not None:
                loss = self.loss_fct(logits, target[:, i])
                total_loss += loss
                input = torch.cat([input, target[:, i:i + 1]], dim=1)
            else:
                input = torch.cat([input, logits.unsqueeze(1)], dim=1)

        if target is not None:
            return outputs, total_loss
        else:
            return outputs

    '''
    def decodering(self,input,steps):
        outputs=[]
        keep_input=input
        for i in range(steps):
            logits=self.forward(input)
            outputs.append(logits)
            input = torch.cat([input,logits.unsequeeze(1)],dim=1)

        return outputs
    '''


class TSDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, X, Y=None):
        'Initialization'
        self.X = X
        self.Y = Y

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # x=self.X.iloc[index]
        # y=self.Y.iloc[index]

        x = self.X[index]
        if self.Y is not None:
            y = self.Y[index]
            return x, y

        else:
            return x


def ftrain(train_input, train_target,
           Epoch, batch_size, lr,
           beta1, beta2, weight_decay,
           save_model_path='model.json'):
    trainset = TSDataset(X=train_input, Y=train_target)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=None,
                             batch_sampler=None, num_workers=0, collate_fn=None,
                             pin_memory=False, drop_last=False, timeout=0,
                             worker_init_fn=None)

    model = RNNForTimeSeries(50, 3)
    if torch.cuda.is_available():
        model.cuda()

    optimizer = Adam(model.parameters(), lr, betas=(beta1, beta2),
                     weight_decay=weight_decay)

    train_iterator = trange(Epoch, desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(trainloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            (x, y) = batch
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # print(x.device)
            # print(x)

            optimizer.zero_grad()
            # optimizer.cuda()

            logits, loss = model(input=x, target=y)

            if torch.is_tensor(loss):
                loss.backward()
                optimizer.step()

    torch.save(model.state_dict(), save_model_path)

def fpredict(test_input,
             batch_size,
           model,
           max_decode_length=40):
    testset = TSDataset(X=test_input)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=None,
                             batch_sampler=None, num_workers=0, collate_fn=None,
                             pin_memory=False, drop_last=False, timeout=0,
                             worker_init_fn=None)


    if torch.cuda.is_available():
        model.cuda()



    epoch_iterator = tqdm(testloader, desc="Iteration")

    res=None

    for step, batch in enumerate(epoch_iterator):

            x = batch
            if torch.cuda.is_available():
                x = x.cuda()



            logits= model(input=x,max_length=max_decode_length)
            logits=np.concatenate([x.detach().numpy() for x in logits],1)
            print(logits.shape)

            if res is None:
                res=logits
            else:
                res=np.concatenate((res,logits))

    return res

def load_model(filename,seqlen=50,layers=3):
    model = RNNForTimeSeries(seqlen, layers)
    model.load_state_dict(torch.load(filename))
    if torch.cuda.is_available():
        model.cuda()
    return model

def loading_models_rnn(model_dir,locations):
    local2models={}
    for local in locations:
        model=load_model(os.path.join(model_dir,'{}_model.json'.format(local)))
        local2models[local]=model
    return local2models




# generate sequence data for local by slide window
def produce_data(df, train_len, test_len=40):
    total_len = train_len + test_len
    loc2df = convert_local2values(df)
    loc2data = {}
    for loc in loc2df:
        sub_df = loc2df[loc]
        values = sub_df['Value'].to_numpy()

        data_list = []
        for i in range(values.size - total_len):
            data = values[i:i + total_len]
            data_list.append(data)

        loc2data[loc] = np.array(data_list)

    return loc2data

if __name__=='__main__':
    df, loc2values = read_ground_truth(r'/vc_data/zhuwe/jupyter_sever_logs/tc_sfr/data/2008_2018_tenyears')
    df = df[df['Value'].notnull()]
    df = df.sort_values(by=['DateTime'], ignore_index=True)

    x_len = 60
    y_len = 40

    loc2data = produce_data(df, x_len, y_len)

    model_dir = '/vc_data/zhuwe/jupyter_sever_logs/tc_sfr/wdata/rnn_models'

    istrain=(sys.argv[1].lower() == 'train')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    local2models={}
    if not istrain:
        locations=['BNDN5','ARWN8','TCCC1','CARO2','ESSC2','NFDC1','LABW4','CLNK1','TRAC2','NFSW4']
        local2models=loading_models_rnn(model_dir,locations)




    for loc in loc2data:
        data = loc2data[loc]


        train_size = int(data.shape[0] * 0.9)
        train = data[:train_size, :]
        dev = data[train_size:, :]

        train_x = train[:, :x_len]
        train_y = train[:, x_len:]

        dev_x = dev[:, :x_len]
        dev_y = dev[:, x_len:]

        if istrain:
            ftrain(train_input=train_x,
               train_target=train_y,
               Epoch=5,
               batch_size=128,
               lr=4e-5,
               beta1=0.9,
               beta2=0.999,
               weight_decay=0,
               save_model_path=os.path.join(model_dir, '{}_model.json'.format(loc)))

        else:
            res=fpredict(test_input=dev_x,
                 batch_size=128,
                 model=local2models[loc],
                 max_decode_length=40)
            print(res)





