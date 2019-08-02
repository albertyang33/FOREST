'''
This script handling the training process.
'''

import argparse
import math
import time
import metrics
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import  Constants
from model import RNNModel
from model import RRModel
from Optim import ScheduledOptim
from DataLoader import DataLoader

def get_performance(crit, pred, gold, smoothing=False, num_class=None):
    ''' Apply label smoothing if needed '''

    # TODO: Add smoothing
    if smoothing:
        assert bool(num_class)
        eps = 0.1
        gold = gold * (1 - eps) + (1 - gold) * eps / num_class
        raise NotImplementedError

    loss = crit(pred, gold.contiguous().view(-1))

    pred = pred.max(1)[1]

    gold = gold.contiguous().view(-1)
    n_correct = pred.data.eq(gold.data)
    n_correct = n_correct.masked_select(gold.ne(Constants.PAD).data).sum().float()

    return loss, n_correct

def get_performance_rl(pred, pred_probs, tgt, expect):
    # reward = -|(log) tgt - (log) pred|
    # loss = -log prob * (reward-expect)
    # update expect reward prob*reward
    pred_probs = torch.cat([pred_probs,torch.zeros(pred.size(0),1).cuda()],dim=1) #manully add EOS prob log(1)=0
    pred_size=torch.argmax((pred==Constants.EOS).float(),dim=1)#.cpu().numpy()
    pred_size=pred_size+(pred_size==0).long()*1000#pred.size(1) #no eos, the first is start
    tgt_size=torch.argmax((tgt==Constants.EOS).float(),dim=1)#.cpu().numpy()
    reward = pred_size.float().log()-tgt_size.float().log()
    reward = - reward.mul(reward) # square of log-transformed loss
    reward_sum = reward.sum().cpu().numpy()
    reward = reward.cpu().numpy()

    pred_size = pred_size.cpu().numpy()

    seq_prob = Variable(torch.zeros(pred_probs.size(0))).cuda()
    for i in range(pred_probs.size(0)):
        for j in range(min(pred_size[i]+1,pred_probs.size(1))):
            seq_prob[i] += pred_probs[i][j]

    expect=0#-0.8 
    loss = -seq_prob.mul(torch.Tensor(reward-expect).cuda()).sum()
    expect = seq_prob.exp().mul(torch.Tensor(reward).cuda()).sum().detach().cpu().numpy()

    return loss, expect, reward_sum

def train_epoch(model, training_data, crit, optimizer,RL_setting,expect,):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0.0
    n_total_words = 0.0
    n_total_correct = 0.0

    expect_new = 0.0
    reward = 0.0
    batch_num = 0.0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words

        batch_num += tgt.size(0)

        # additional RL training
        if RL_setting:
            optimizer.zero_grad()
            pred_ids, pred_probs = model(tgt,RL_train=True)

            # backward
            loss_rl, expect_batch, reward_batch = get_performance_rl(pred_ids, pred_probs, tgt, expect) # with start ids
            loss_rl.backward()
            expect_new += expect_batch
            reward += reward_batch
            # update parameters
            optimizer.step()
            optimizer.update_learning_rate()
        
        #if not RL_setting:
        # forward
        optimizer.zero_grad()
        pred, *_ = model(tgt,RL_train=False)

        # backward
        loss, n_correct = get_performance(crit, pred, gold)
        loss.backward()

        # update parameters
        optimizer.step()
        optimizer.update_learning_rate()

        # note keeping
        n_total_correct += n_correct
        total_loss += loss.data[0]

    return total_loss/n_total_words, n_total_correct/n_total_words, expect_new/batch_num, reward/batch_num

def eval_epoch(model, validation_data, crit):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_total_words = 0
    n_total_correct = 0

    reward = 0.0
    batch_num = 0.0

    for batch in tqdm(
            validation_data, mininterval=2,
            desc='  - (Validation) ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]

        # forward
        pred, *_ = model(tgt,RL_train=False)
        loss, n_correct = get_performance(crit, pred, gold)

        # note keeping
        n_words = gold.data.ne(Constants.PAD).sum().float()
        n_total_words += n_words
        n_total_correct += n_correct
        total_loss += loss.data[0]

        pred_ids, pred_probs = model(tgt,RL_train=True)

        # backward
        loss_rl, expect_batch, reward_batch = get_performance_rl(pred_ids, pred_probs, tgt, expect=0) # with start ids
        reward += reward_batch
        batch_num += tgt.size(0)

    return total_loss/n_total_words, n_total_correct/n_total_words, reward/batch_num

def test_epoch(model, test_data, k_list=[1,5,10,20,50,100]):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    scores = {}
    for k in k_list:
        scores['hits@' + str(k)] = 0
        scores['map@' + str(k)] = 0
    n_total_words = 0

    reward = 0.0
    batch_num = 0.0

    for batch in tqdm(
            test_data, mininterval=2,
            desc='  - (Test) ', leave=False):

        # prepare data
        tgt = batch
        gold = tgt[:, 1:]

        # forward
        pred, *_ = model(tgt,RL_train=False)
        scores_batch, scores_len = metrics.portfolio(pred.detach().cpu().numpy(), gold.contiguous().view(-1).detach().cpu().numpy(), k_list)
        n_total_words += scores_len
        for k in k_list:
            scores['hits@' + str(k)] += scores_batch['hits@' + str(k)] * scores_len
            scores['map@' + str(k)] += scores_batch['map@' + str(k)] * scores_len        

        pred_ids, pred_probs = model(tgt,RL_train=True)

        for i in range(10):
            loss_rl, expect_batch, reward_batch = get_performance_rl(pred_ids, pred_probs, tgt, expect=0) # with start ids
            reward += reward_batch
            batch_num += tgt.size(0)

    for k in k_list:
        scores['hits@' + str(k)] = scores['hits@' + str(k)] / n_total_words
        scores['map@' + str(k)] = scores['map@' + str(k)] / n_total_words
    return scores, reward / batch_num

def train(model, training_data, validation_data, test_data, crit, optimizer, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    expect = 0
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        if epoch_i<opt.warmup:
            RL_setting = False
        else:
            RL_setting = True

        start = time.time()
        train_loss, train_accu, expect, reward = train_epoch(model, training_data, crit, optimizer,RL_setting,expect)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time()-start)/60))

        # validation
        start = time.time()
        valid_loss, valid_accu, reward = eval_epoch(model, validation_data, crit)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, Reward: {rwd:3.8f} , '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu, rwd=reward,
                    elapse=(time.time()-start)/60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if epoch_i>=opt.warmup-1 or epoch_i % 5==4:
            # test
            scores, reward_test = test_epoch(model, test_data)
            print('  - (Test) ')
            for metric in scores.keys():
                print(metric+' '+str(scores[metric]))
            print('reward '+str(reward_test)+'\n')

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))

def main():
    torch.set_num_threads(4)
    ''' Main function'''
    parser = argparse.ArgumentParser()

    #parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=16)

    #parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_inner_hid', type=int, default=64)

    parser.add_argument('-n_warmup_steps', type=int, default=1000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='model')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

    parser.add_argument('-no_cuda', action='store_true')

    parser.add_argument('-network', type=int, default=0) # use social network; need features or deepwalk embeddings as initial input
    parser.add_argument('-pos_emb', type=int, default=1)
    parser.add_argument('-warmup', type=int, default=10) # warmup epochs
    parser.add_argument('-notes', default='')
    parser.add_argument('-data_name', default='twitter')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model
    if opt.network==1:
        opt.network = True
    else:
        opt.network = False
    if opt.pos_emb==1:
        opt.pos_emb = True
    else:
        opt.pos_emb = False
    print(opt.notes)
    

    #========= Preparing DataLoader =========#
    train_data = DataLoader(opt.data_name, data=0, load_dict=True, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=opt.network)
    valid_data = DataLoader(opt.data_name, data=1, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=opt.network)
    test_data = DataLoader(opt.data_name, data=2, batch_size=opt.batch_size, cuda=opt.cuda, loadNE=opt.network)

    opt.user_size = train_data.user_size
    if opt.network:
        opt.net = train_data._adj_list
        opt.net_dict = train_data._adj_dict_list
        opt.embeds = train_data._embeds

    #========= Preparing Model =========#
    #print(opt)

    decoder = RNNModel('GRUCell', opt)
    RLLearner = RRModel(decoder)
    #print(transformer)

    optimizer = ScheduledOptim(
        optim.Adam(
            RLLearner.parameters(),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)


    def get_criterion(user_size):
        ''' With PAD token zero weight '''
        weight = torch.ones(user_size)
        weight[Constants.PAD] = 0
        weight[Constants.EOS] = 1
        return nn.CrossEntropyLoss(weight, size_average=False)

    crit = get_criterion(train_data.user_size)

    if opt.cuda:
        decoder = decoder.cuda()
        RLLearner = RLLearner.cuda()
        crit = crit.cuda()

    train(RLLearner, train_data, valid_data, test_data, crit, optimizer, opt)

if __name__ == '__main__':
    main()
