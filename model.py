import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp
import Constants

def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1,1,seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    #print(previous_mask)
    #print(seqs)
    masked_seq = previous_mask * seqs.data.float()
    #print(masked_seq.size())

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq,PAD_tmp],dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2,masked_seq.long(),float('-inf'))
    
    return masked_seq

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)).cuda()
    values = torch.from_numpy(sparse_mx.data).cuda()
    shape = torch.Size(sparse_mx.shape)
    return torch.cuda.sparse.FloatTensor(indices, values, shape)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, opt, dropout=0.1, tie_weights=False):
        super(RNNModel, self).__init__()
        ntoken = opt.user_size
        ninp = opt.d_word_vec
        nhid = opt.d_inner_hid
        # network part
        self.use_network = opt.network
        #print(self.use_network)
        if opt.network:
            adj = sp.coo_matrix((opt.net[2], (opt.net[0], opt.net[1])),
                            shape=(ntoken, ntoken),
                            dtype=np.float32)
            adj = normalize(adj)
            self.adj = sparse_mx_to_torch_sparse_tensor(adj)
            self.adj = self.adj.cuda()
            self.adj_list = opt.net_dict
            self.net_emb=torch.from_numpy(opt.embeds).float().cuda()
            self.nnl1 = 25
            self.nnl2 = 10
            self.gcn1 = nn.Linear(nhid, nhid)
            self.gcn2 = nn.Linear(self.net_emb.size(1), nhid)
            self.mem_size = 3
        # end network part

        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        # positional embedding
        self.pos_emb = opt.pos_emb
        if self.pos_emb:
            self.pos_dim = 8
            self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        
        if self.pos_emb:
            self.rnn = getattr(nn, rnn_type)(ninp+self.pos_dim, nhid)
        else:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid)
        
        if opt.network:
            self.decoder = nn.Linear(nhid+nhid, ntoken)#
        else:
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.user_size = ntoken

    def neighbor_sampling(self, nodes, num_neighbor):
        sample = np.zeros((nodes.shape[0],num_neighbor),dtype=int)
        for i in range(nodes.shape[0]):
            sample[i,0]=nodes[i]
            sample[i,1:]=np.random.choice(self.adj_list[nodes[i]], num_neighbor-1, replace=True)
        return sample

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if self.use_network:
            self.gcn1.bias.data.fill_(0)
            self.gcn1.weight.data.uniform_(-initrange, initrange)
            self.gcn2.bias.data.fill_(0)
            self.gcn2.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, generate=False):
        if not generate:
            input = input[:, :-1]
        # network part
        if self.use_network:
            nb1 = self.neighbor_sampling(input.data.cpu().numpy().reshape(-1), self.nnl1) #(batch*len)*nnl1
            nb2 = self.neighbor_sampling(nb1.reshape(-1), self.nnl2) #(batch*len*nnl1)*nnl2
            
            nf2 = nn.functional.relu(self.gcn2(self.net_emb[nb2,:]).mean(dim=1).view(-1,self.nnl1,self.nhid)) #(batch*len)*nnl1*nhid
            nf1 = nn.functional.relu(self.gcn1(nf2).mean(dim=1).view(input.size(0),input.size(1),self.nhid))

            # memory
            net_emb = torch.cat([torch.zeros((input.size(0),self.mem_size-1,self.nhid)).cuda(),nf1],dim=1) # batch*(len+mem)*nhid nf1 #self.net_emb[input,:]
            net_emb= torch.stack([net_emb[:,i:input.size(1)+i,:] for i in range(self.mem_size)], dim=2).view(input.size(0)*input.size(1),-1,self.nhid)# batch*len*5*nhid
            #cat#net_emb= torch.cat([net_emb[:,i:input.size(1)+i,:].contiguous().view(input.size(0)*input.size(1),self.nhid) for i in range(self.mem_size)],dim=1)# batch*len*5*nhid
            net_emb = net_emb.mean(dim=1).contiguous().view(input.size(0),input.size(1),self.nhid)

        # embedding
        emb = self.drop(self.encoder(input)) # without network
        #emb = self.drop(torch.cat([self.encoder(input),nf1],dim=2))

        batch_size = input.size(0)
        max_len = input.size(1)
        if self.use_network:
            outputs = Variable(torch.zeros(max_len, batch_size, self.nhid+self.nhid)).cuda()#
        else:
            outputs = Variable(torch.zeros(max_len, batch_size, self.nhid)).cuda()
        hidden = Variable(torch.zeros(batch_size, self.nhid)).cuda()
        #hidden_c = Variable(torch.zeros(batch_size, self.nhid)).cuda()
        for t in range(0, max_len):
            # GRU
            if self.pos_emb:
                hidden = self.rnn(torch.cat([emb[:,t,:],self.drop(self.pos_embedding(torch.ones(batch_size).long().cuda()*t))],dim=1), hidden)
            else:
                hidden = self.rnn(emb[:,t,:], hidden)
            #LSTM
            #hidden, hidden_c = self.rnn(torch.cat([emb[:,t,:],self.pos_embedding(torch.ones(batch_size).long().cuda()*t)],dim=1), (hidden,hidden_c))
            if self.use_network:
                outputs[t] = torch.cat([hidden,net_emb[:,t,:]],dim=1)#nf1[:,t,:]
            else:
                outputs[t] = hidden


            
        outputs = outputs.transpose(0,1).contiguous()#b*l*v
        outputs = self.drop(outputs)
        decoded = self.decoder(outputs.view(outputs.size(0)*outputs.size(1), outputs.size(2)))
        result = decoded.view(outputs.size(0), outputs.size(1), decoded.size(1)) + torch.autograd.Variable(get_previous_user_mask(input, self.user_size),requires_grad=False)
        #print(result.size())
        
        return result.view(-1,decoded.size(1)), hidden

class RRModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_model):
        super(RRModel, self).__init__()
        self.rnn_model = rnn_model

    def forward(self, input, start_size=5, RL_train=False, generate=False):
        if not RL_train:
            return self.rnn_model.forward(input,generate)
        max_len = 500#2*input.size(1) #alert
        batch_size = input.size(0)
        # greedy simulation
        outputs_prob = Variable(torch.zeros(batch_size, max_len)).cuda()
        outputs_id = Variable(torch.zeros(batch_size, max_len)).cuda() # with start ids
        if self.rnn_model.use_network:
            nfs = Variable(torch.zeros(batch_size, max_len, self.rnn_model.nhid)).cuda()
        
        emb = self.rnn_model.drop(self.rnn_model.encoder(input)) # without network
        hidden = Variable(torch.zeros(batch_size, self.rnn_model.nhid)).cuda()

        for t in range(0, max_len):
            if t<start_size:
                step_input = emb[:,t,:]
                # net emb
                if self.rnn_model.use_network:
                    nb1 = self.rnn_model.neighbor_sampling(input[:,t].data.cpu().numpy().reshape(-1), self.rnn_model.nnl1) #(batch*1)*nnl1
                    nb2 = self.rnn_model.neighbor_sampling(nb1.reshape(-1), self.rnn_model.nnl2) #(batch*len*nnl1)*nnl2        
                    nf2 = nn.functional.relu(self.rnn_model.gcn2(self.rnn_model.net_emb[nb2,:]).mean(dim=1).view(-1,self.rnn_model.nnl1,self.rnn_model.nhid)) #(batch*len)*nnl1*nhid
                    nf1 = nn.functional.relu(self.rnn_model.gcn1(nf2).mean(dim=1).view(input.size(0),self.rnn_model.nhid))
                    nfs[:,t,:] = nf1

                if self.rnn_model.pos_emb:
                    hidden = self.rnn_model.rnn(torch.cat([step_input,self.rnn_model.drop(self.rnn_model.pos_embedding(torch.ones(batch_size).long().cuda()*t))],dim=1), hidden)
                else:
                    hidden = self.rnn_model.rnn(step_input, hidden)
                #hidden, hidden_c = self.rnn_model.rnn(torch.cat([step_input,self.rnn_model.pos_embedding(torch.ones(batch_size).long().cuda()*t)],dim=1), (hidden,hidden_c))
                outputs_id[:,t] = input[:,t]
                outputs_prob[:,t] = torch.zeros(batch_size)
                continue

            # net emb
            if self.rnn_model.use_network:
                nb1 = self.rnn_model.neighbor_sampling(outputs_id[:,t-1].long().cpu().numpy(), self.rnn_model.nnl1) #(batch*1)*nnl1
                nb2 = self.rnn_model.neighbor_sampling(nb1.reshape(-1), self.rnn_model.nnl2) #(batch*len*nnl1)*nnl2    
                nf2 = nn.functional.relu(self.rnn_model.gcn2(self.rnn_model.net_emb[nb2,:]).mean(dim=1).view(-1,self.rnn_model.nnl1,self.rnn_model.nhid)) #(batch*len)*nnl1*nhid
                nf1 = nn.functional.relu(self.rnn_model.gcn1(nf2).mean(dim=1).view(input.size(0),self.rnn_model.nhid))
                nfs[:,t,:] = nf1
                net_emb = nfs[:,t-self.rnn_model.mem_size+1:t+1,:].mean(dim=1)

            step_input = self.rnn_model.encoder(outputs_id[:,t-1].long())
            if self.rnn_model.pos_emb:
                hidden = self.rnn_model.rnn(torch.cat([step_input,self.rnn_model.drop(self.rnn_model.pos_embedding(torch.ones(batch_size).long().cuda()*t))],dim=1), hidden)
            else:
                hidden = self.rnn_model.rnn(step_input, hidden)
            #hidden,hidden_c = self.rnn_model.rnn(torch.cat([step_input,self.rnn_model.pos_embedding(torch.ones(batch_size).long().cuda()*t)],dim=1), (hidden,hidden_c))
            if self.rnn_model.use_network:
                output = torch.cat([hidden,net_emb],dim=1)#net_emb nf1
            else:
                output = hidden
            decoded = self.rnn_model.decoder(output) #b*v
            result = decoded + torch.autograd.Variable(torch.zeros(batch_size,self.rnn_model.user_size).cuda().scatter_(1,outputs_id[:,:t].long(),float('-inf')),requires_grad=False)
            #top1 = result.data.max(1)[1]
            top1 = torch.multinomial(torch.nn.functional.softmax(result,dim=1),1).squeeze()
            #print(top1.size())
            outputs_id[:,t]=top1
            #print(torch.gather(torch.nn.functional.log_softmax(result,dim=1),1,top1.unsqueeze(1)).squeeze())
            log_probs=torch.nn.functional.log_softmax(result,dim=1)
            #print(log_probs.size())
            #outputs_prob[:,t]=log_probs[:,Constants.EOS].mul((top1==Constants.EOS).float())+((1-log_probs[:,Constants.EOS].exp()).log()).mul((top1!=Constants.EOS).float())
            if top1.dim()==0:
                top1=top1.unsqueeze(0)
            outputs_prob[:,t]=torch.gather(log_probs,1,top1.unsqueeze(1)).squeeze()
        return outputs_id, outputs_prob
