''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
from torch.autograd import Variable
import Constants
import logging
import pickle
class Options(object):
    
    def __init__(self, data_name = 'twitter'):
        #data options.
        #data_name = 'twitter'
        #train file path.
        self.train_data = 'data/'+data_name+'/cascade.txt'
        #valid file path.
        self.valid_data = 'data/'+data_name+'/cascadevalid.txt'
        #test file path.
        self.test_data = 'data/'+data_name+'/cascadetest.txt'

        self.u2idx_dict = 'data/'+data_name+'/u2idx.pickle'

        self.idx2u_dict = 'data/'+data_name+'/idx2u.pickle'
        #save path.
        self.save_path = ''

        self.batch_size = 32

        self.net_data = 'data/'+data_name+'/edges.txt'

        self.embed_dim = 64
        self.embed_file = 'data/'+data_name+'/dw'+str(self.embed_dim)+'.txt'

class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, data_name, data=0, load_dict=True, cuda=True, batch_size=32, shuffle=True, test=False, with_EOS=True, loadNE=True): #data = 0 for train, 1 for valid, 2 for test
        self.options = Options(data_name)
        self.options.batch_size = batch_size
        self._u2idx = {}
        self._idx2u = []
        self.data = data
        self.test = test
        self.with_EOS = with_EOS
        if not load_dict:
            self._buildIndex()
            with open(self.options.u2idx_dict, 'wb') as handle:
                pickle.dump(self._u2idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(self.options.idx2u_dict, 'wb') as handle:
                pickle.dump(self._idx2u, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(self.options.u2idx_dict, 'rb') as handle:
                self._u2idx = pickle.load(handle)
            with open(self.options.idx2u_dict, 'rb') as handle:
                self._idx2u = pickle.load(handle)
            self.user_size = len(self._u2idx)
        self._train_cascades,train_len = self._readFromFile(self.options.train_data)
        self._valid_cascades,valid_len = self._readFromFile(self.options.valid_data)
        self._test_cascades,test_len = self._readFromFile(self.options.test_data)
        self.train_size = len(self._train_cascades)
        self.valid_size = len(self._valid_cascades)
        self.test_size = len(self._test_cascades)
        print("training set size:%d   valid set size:%d  testing set size:%d" % (self.train_size, self.valid_size, self.test_size))
        print(self.train_size+self.valid_size+self.test_size)
        print((train_len+valid_len+test_len+0.0)/(self.train_size+self.valid_size+self.test_size))
        print(self.user_size-2)
        self.cuda = cuda
        #self.test = test
        if self.data == 0:
            self._n_batch = int(np.ceil(len(self._train_cascades) / batch_size))
        elif self.data == 1:
            self._n_batch = int(np.ceil(len(self._valid_cascades) / batch_size))
        else:
            self._n_batch = int(np.ceil(len(self._test_cascades) / batch_size))

        self._batch_size = self.options.batch_size

        self._iter_count = 0

        if loadNE:
            self._adj_list = self._readNet(self.options.net_data)
            self._adj_dict_list=self._readNet_dict_list(self.options.net_data)
            self._embeds = self._load_ne(self.options.embed_file,self.options.embed_dim)

        self._need_shuffle = shuffle

        if self._need_shuffle:
            random.shuffle(self._train_cascades)

    def _buildIndex(self):
        #compute an index of the users that appear at least once in the training and testing cascades.
        opts = self.options

        train_user_set = set()
        valid_user_set = set()
        test_user_set = set()

        lineid=0
        for line in open(opts.train_data):
            lineid+=1
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(line)
                    print(chunk)
                    print(lineid)
                train_user_set.add(user)

        for line in open(opts.valid_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                valid_user_set.add(user)

        for line in open(opts.test_data):
            if len(line.strip()) == 0:
                continue
            chunks = line.strip().split()
            for chunk in chunks:
                user, timestamp = chunk.split(',')
                test_user_set.add(user)

        user_set = train_user_set | valid_user_set | test_user_set

        pos = 0
        self._u2idx['<blank>'] = pos
        self._idx2u.append('<blank>')
        pos += 1
        self._u2idx['</s>'] = pos
        self._idx2u.append('</s>')
        pos += 1

        for user in user_set:
            self._u2idx[user] = pos
            self._idx2u.append(user)
            pos += 1
        opts.user_size = len(user_set) + 2
        self.user_size = len(user_set) + 2
        print("user_size : %d" % (opts.user_size))

    def _readNet(self, filename):
        adj_list=[[],[],[]]
        n_edges = 0
        # add self edges
        for i in range(self.user_size):
            adj_list[0].append(i)
            adj_list[1].append(i)
            adj_list[2].append(1)
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            n_edges+=1
            adj_list[0].append(self._u2idx[nodes[0]])
            adj_list[1].append(self._u2idx[nodes[1]])
            adj_list[2].append(1) # weight
        print('edge:')
        print(n_edges/2)
        return adj_list

    def _readNet_dict_list(self, filename):
        adj_list={}
        # add self edges
        for i in range(self.user_size):
            adj_list.setdefault(i,[i]) # [i] or []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            nodes = line.strip().split(',')
            if nodes[0] not in self._u2idx.keys() or nodes[1] not in self._u2idx.keys():
                continue
            adj_list[self._u2idx[nodes[0]]].append(self._u2idx[nodes[1]])
            adj_list[self._u2idx[nodes[1]]].append(self._u2idx[nodes[0]])
        return adj_list

    def _load_ne(self, filename, dim):
        embed_file=open(filename,'r')
        line = embed_file.readline().strip()
        dim = int(line.split()[1])
        embeds = np.zeros((self.user_size,dim))
        for line in embed_file.readlines():
            line=line.strip().split()
            if line[0] not in self._u2idx.keys():
                print("cannot find this user")
                print(line[0])
                continue
            embeds[self._u2idx[line[0]],:]= np.array(line[1:])
        return embeds

    def _readFromFile(self, filename):
        """read all cascade from training or testing files. """
        total_len = 0
        t_cascades = []
        for line in open(filename):
            if len(line.strip()) == 0:
                continue
            userlist = []
            chunks = line.strip().split()
            for chunk in chunks:
                try:
                    user, timestamp = chunk.split(',')
                except:
                    print(chunk)
                if user in self._u2idx:
                    userlist.append(self._u2idx[user])

            if len(userlist) > 1 and len(userlist)<=500:
                total_len+=len(userlist)
                if self.with_EOS:
                    userlist.append(Constants.EOS)
                t_cascades.append(userlist)
        return t_cascades,total_len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def pad_to_longest(insts):
            ''' Pad the instance to the max seq length in batch '''

            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])
        
            inst_data_tensor = Variable(
                torch.LongTensor(inst_data), volatile=self.test)

            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()

            return inst_data_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            if self.data == 0:
                seq_insts = self._train_cascades[start_idx:end_idx]
            elif self.data == 1:
                seq_insts = self._valid_cascades[start_idx:end_idx]
            else:
                seq_insts = self._test_cascades[start_idx:end_idx]
            seq_data = pad_to_longest(seq_insts)

            return seq_data
        else:

            if self._need_shuffle:
                random.shuffle(self._train_cascades)
                #random.shuffle(self._test_cascades)

            self._iter_count = 0
            raise StopIteration()
