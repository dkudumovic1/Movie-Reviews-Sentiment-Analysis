#%%
import pandas as pd
from utils import preprocessing
from utils import feature_extraction
import torch 
import numpy as np
from torch.utils.data import (
    DataLoader, TensorDataset
) 
import torch.nn as nn
import os
import matplotlib.pyplot as plt

x_train = pd.read_csv('x_train.csv', converters = {'review': str})
x_test = pd.read_csv('x_test.csv', converters = {'review': str})
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

#%%
#GLOVE
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'glove.6B.200d.txt')
word2vec_output_file = 'glove.6B.200d' +'.word2vec'

model = feature_extraction.load_glove_model(filepath, word2vec_output_file)
tokenizer, dictionary = feature_extraction.get_dictionary(x_train)
embedding = feature_extraction.get_glove_embedding_BiLSTM(model,dictionary)


print(f'Length of vocabulary is {len(dictionary)}')

X_train_indices = tokenizer.texts_to_sequences(x_train['review'])
X_test_indices = tokenizer.texts_to_sequences(x_test['review'])

#%%
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    

#%%
def padding(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

x_train_pad = padding(np.array(X_train_indices),100)
x_test_pad = padding(np.array(X_test_indices),100)

#%%
#TENSOR DATASETS, DATALOADERS
batch_size = 64

train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train['sentiment'].to_numpy()))
test_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test['sentiment'].to_numpy()))

train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

#%%
output_dim = 1
#NETWORK DEFINITION
class SentimentRNN(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embedding).float(),requires_grad=True)
        
        #lstm
        self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                           num_layers=no_layers, batch_first=True, bidirectional = True)
        
        self.maxpool = nn.MaxPool1d(4) # Where 4 is kernal size
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
    #    # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
  
        
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers * 2,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden


#%%
no_layers = 2
vocab_size = len(dictionary)
embedding_dim = 200
hidden_dim = 256
lr=0.0005

#%%
model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
model.to(device)
print(model)

#LOSS AND OPTIMISATION FUNCTIONS
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# function to predict accuracy
def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()


#TRAINING
#%%
clip = 5
epochs = 3
train_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state 
    h = model.init_hidden(batch_size)
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)   
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        
        model.zero_grad()
        output,h = model(inputs,h)
        
        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        train_losses.append(loss.item())
        # calculating accuracy
        accuracy = acc(output,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
    epoch_train_loss = np.mean(train_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_tr_acc.append(epoch_train_acc)
        
    print(f'Epoch {epoch+1}') 
    print(f'train_loss : {epoch_train_loss} val_loss : {0}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {0}')
    if epoch_train_loss <= train_loss_min:
        torch.save(model.state_dict(), './state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(train_loss_min,epoch_train_loss))
        train_loss_min = epoch_train_loss
    print(25*'==')



#%%
#testing
model.load_state_dict(torch.load('./state_dict.pt'), strict=False)

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)
TP = 0
TN = 0
FP = 0
FN = 0

model.eval()
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    
    
    for i in range(0, len(pred)):
        prediction = pred.cpu().detach().numpy()[i]
        true = labels[i]
        
        if prediction == 1 and true == 1 :
            TP += 1
        elif prediction == 0 and true == 0 :
            TN += 1
        elif prediction == 1 and true == 0 :
            FP += 1
        elif prediction == 0 and true == 1:
            FN += 1
            
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
precision = TP / (TP + FP)
print("Precision: ", precision)
recall = TP / (TP + FN)
print("Recal:", recall)
print("F1", 2 * (precision * recall) / (precision + recall))
# %%
#neutral dataset
neutral = pd.read_csv('neutral_dataset.csv', converters = {'Phrase': str})

#GLOVE
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, 'glove.6B.200d.txt')
word2vec_output_file = 'glove.6B.200d' +'.word2vec'

model = feature_extraction.load_glove_model(filepath, word2vec_output_file)
tokenizer, dictionary = feature_extraction.get_dictionary(x_train)
embedding = feature_extraction.get_glove_embedding_BiLSTM(model,dictionary)

print(f'Length of vocabulary is {len(dictionary)}')

neutral_indices = tokenizer.texts_to_sequences(neutral['Phrase'])
neutral_pad = padding(np.array(neutral_indices),100)

batch_size = 64

neutral_data = TensorDataset(torch.from_numpy(neutral_pad), torch.from_numpy(neutral['Sentiment'].to_numpy()))
neutral_loader = DataLoader(neutral_data, shuffle=False, batch_size=batch_size, drop_last=True)

#%%
#predictions
model = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
model.to(device)
print(model)

#LOSS AND OPTIMISATION FUNCTIONS
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.load_state_dict(torch.load('./state_dict.pt'), strict=False)

#%%
test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)
TP = 0
TN = 0
FP = 0
FN = 0
NP = 0 #neutral positive
NN = 0 #neutral negative

model.eval()
for inputs, labels in neutral_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h = model(inputs, h)
#    test_loss = criterion(output.squeeze(), labels.float())
#    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
#    correct_tensor = pred.eq(labels.float().view_as(pred))
#    correct = np.squeeze(correct_tensor.cpu().numpy())
    
    
    for i in range(0, len(pred)):
        prediction = pred.cpu().detach().numpy()[i]
        true = labels[i]
        
        if prediction == 1 and (true == 1 or true == 0):
            FP += 1
        elif prediction == 1 and (true == 3 or true == 4):
            TP += 1
        elif prediction == 0 and (true == 1 or true == 0) :
            TN += 1
        elif prediction == 0 and (true == 3 or true == 4):
            FN += 1
        elif prediction == 1 and true == 2:
            NP += 1
        elif prediction == 0 and true == 2:
            NN += 1
            

test_acc = (TP+TN)/(TP+TN+FP+FN)
print("Test accuracy: {:.3f}%".format(test_acc*100))
print("TP: ", TP)
print("TN: ", TN)
print("FP: ", FP)
print("FN: ", FN)
print("NP: ", NP)
print("NN: ", NN)
#precision = TP / (TP + FP)
#print("Precision: ", precision)
#recall = TP / (TP + FN)
#print("Recal:", recall)
#print("F1", 2 * (precision * recall) / (precision + recall))
# %%
