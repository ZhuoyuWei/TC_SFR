from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import numpy as np

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
                input = torch.cat([input, logits.unsequeeze(1)], dim=1)

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
    model.cuda()

    optimizer = Adam(model.parameters(), lr, betas=(beta1, beta2),
                     weight_decay=weight_decay)

    train_iterator = trange(Epoch, desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(trainloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            (x, y) = batch
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
           save_model_path='model.json',
           max_decode_length=40):
    testset = TSDataset(X=test_input)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=None,
                             batch_sampler=None, num_workers=0, collate_fn=None,
                             pin_memory=False, drop_last=False, timeout=0,
                             worker_init_fn=None)

    model = RNNForTimeSeries(50, 3)
    model.load_state_dict(torch.load(save_model_path))
    model.cuda()



    epoch_iterator = tqdm(testloader, desc="Iteration")

    res=None

    for step, batch in enumerate(epoch_iterator):

            x = batch
            x = x.cuda()
            #y = y.cuda()


            logits= model(input=x,max_length=max_decode_length)
            logits=logits.to_numpy()

            if res is None:
                res=logits
            else:
                res=np.concatenate((res,logits))

    return res





