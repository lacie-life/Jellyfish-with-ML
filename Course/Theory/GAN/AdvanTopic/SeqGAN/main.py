from __future__ import print_function
from math import ceil
import numpy as np
import sys
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torchtext as tt

import generator
import discriminator
import helpers


CUDA = True
VOCAB_SIZE = 5000
MAX_SEQ_LEN = 30
START_LETTER = 0
BATCH_SIZE = 32
MLE_TRAIN_EPOCHS = 4   # 100
ADV_TRAIN_EPOCHS = 4   # 50
POS_NEG_SAMPLES = 1000  # 10000

GEN_EMBEDDING_DIM = 32
GEN_HIDDEN_DIM = 32
DIS_EMBEDDING_DIM = 64
DIS_HIDDEN_DIM = 64

# oracle_samples_path = './oracle_samples.trc'
# oracle_state_dict_path = './oracle_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_gen_path = './gen_MLEtrain_EMBDIM32_HIDDENDIM32_VOCAB5000_MAXSEQLEN20.trc'
# pretrained_dis_path = './dis_pretrain_EMBDIM_64_HIDDENDIM64_VOCAB5000_MAXSEQLEN20.trc'

src = tt.data.Field(tokenize=tt.data.utils.get_tokenizer("basic_english"),
                    fix_length=MAX_SEQ_LEN,
                    lower=True)

dataset = tt.data.TabularDataset(
    path='jokes.csv', format='csv',
    fields=[('id', None), ('text', src)])

src.build_vocab(dataset, max_size=VOCAB_SIZE)

src_itr = tt.data.BucketIterator(dataset=dataset,
                                 batch_size=BATCH_SIZE,
                                 sort_key=lambda x: len(x.text),
                                 device=torch.device("cuda:0"))


class BatchLoader:
    def __init__(self, dl, x_field):
        self.dl, self.x_field = dl, x_field

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_field)
            yield x.t()


train_loader = BatchLoader(src_itr, 'text')

vocab_max = 0
for i, batch in enumerate(train_loader):
    _max = torch.max(batch)
    if _max > vocab_max:
        vocab_max = _max

VOCAB_SIZE = vocab_max.item() + 1

inv_vocab = {v: k for k, v in src.vocab.stoi.items()}

# sentence = ['at', 'a', 'dinner', 'party']
# for w in sentence:
#     v = src.vocab[w]
#     print(v)
#     print(inv_vocab[v])

def train_generator_MLE(gen, gen_opt, oracle, real_data_samples, epochs):
    """
    Max Likelihood Pretraining for the generator
    """
    for epoch in range(epochs):
        print('epoch %d : ' % (epoch + 1), end='')
        sys.stdout.flush()
        total_loss = 0

        for i, batch in enumerate(real_data_samples):
            inp, target = helpers.prepare_generator_batch(batch, start_letter=START_LETTER,
                                                          gpu=CUDA)
            gen_opt.zero_grad()
            loss = gen.batchNLLLoss(inp, target)
            loss.backward()
            gen_opt.step()

            total_loss += loss.data.item()

            if (i / BATCH_SIZE) % ceil(
                            ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                print('.', end='')
                sys.stdout.flush()

        # each loss in a batch is loss per sample
        total_loss = total_loss / ceil(POS_NEG_SAMPLES / float(BATCH_SIZE)) / MAX_SEQ_LEN

        # sample from generator and compute oracle NLL
        oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

        print(' average_train_NLL = %.4f, oracle_sample_NLL = %.4f' % (total_loss, oracle_loss))
    torch.save(gen, 'netG_MLE.pt')


def train_generator_PG(gen, gen_opt, oracle, dis, num_batches):
    """
    The generator is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """

    for batch in range(num_batches):
        s = gen.sample(BATCH_SIZE*2)        # 64 works best
        inp, target = helpers.prepare_generator_batch(s, start_letter=START_LETTER, gpu=CUDA)
        rewards = dis.batchClassify(target)

        gen_opt.zero_grad()
        pg_loss = gen.batchPGLoss(inp, target, rewards)
        pg_loss.backward()
        gen_opt.step()

    # sample from generator and compute oracle NLL
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                                   start_letter=START_LETTER, gpu=CUDA)

    print(' oracle_sample_NLL = %.4f' % oracle_loss)
    torch.save(gen, 'netG_RL.pt')
    torch.save(dis, 'netD_RL.pt')


def train_discriminator(discriminator, dis_opt, real_data_samples, generator, oracle, d_steps, epochs):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from generator (negative).
    Samples are drawn d_steps times, and the discriminator is trained for epochs epochs.
    """

    # generating a small validation set before training (using oracle and generator)
    pos_val = oracle.sample(100)
    neg_val = generator.sample(100)
    val_inp, val_target = helpers.prepare_discriminator_data(pos_val, neg_val, gpu=CUDA)

    for d_step in range(d_steps):
        s = helpers.batchwise_sample(generator, POS_NEG_SAMPLES, BATCH_SIZE)
        for epoch in range(epochs):
            print('d-step %d epoch %d : ' % (d_step + 1, epoch + 1), end='')
            sys.stdout.flush()
            total_loss = 0
            total_acc = 0

            for i, batch in enumerate(real_data_samples):
                inp, target = helpers.prepare_discriminator_data(batch, s, gpu=CUDA)
                dis_opt.zero_grad()
                out = discriminator.batchClassify(inp)
                loss_fn = nn.BCELoss()
                loss = loss_fn(out, target)
                loss.backward()
                dis_opt.step()

                total_loss += loss.data.item()
                total_acc += torch.sum((out>0.5)==(target>0.5)).data.item()

                if (i / BATCH_SIZE) % ceil(ceil(2 * POS_NEG_SAMPLES / float(
                        BATCH_SIZE)) / 10.) == 0:  # roughly every 10% of an epoch
                    print('.', end='')
                    sys.stdout.flush()

            total_loss /= ceil(2 * POS_NEG_SAMPLES / float(BATCH_SIZE))
            total_acc /= float(2 * POS_NEG_SAMPLES)

            val_pred = discriminator.batchClassify(val_inp)
            print(' average_loss = %.4f, train_acc = %.4f, val_acc = %.4f' % (
                total_loss, total_acc, torch.sum((val_pred>0.5)==(val_target>0.5)).data.item()/200.))
    torch.save(dis, 'netD_dis.pt')


# MAIN
if __name__ == '__main__':
    oracle = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA, oracle_init=True)
    # oracle.load_state_dict(torch.load(oracle_state_dict_path))
    # oracle_samples = torch.load(oracle_samples_path).type(torch.LongTensor)
    # a new oracle can be generated by passing oracle_init=True in the generator constructor
    # samples for the new oracle can be generated using helpers.batchwise_sample()

    gen = generator.Generator(GEN_EMBEDDING_DIM, GEN_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)
    dis = discriminator.Discriminator(DIS_EMBEDDING_DIM, DIS_HIDDEN_DIM, VOCAB_SIZE, MAX_SEQ_LEN, gpu=CUDA)

    if CUDA:
        oracle = oracle.cuda()
        gen = gen.cuda()
        dis = dis.cuda()
        # oracle_samples = oracle_samples.cuda()

    # GENERATOR MLE TRAINING
    print('Starting Generator MLE Training...')
    gen_optimizer = optim.Adam(gen.parameters(), lr=1e-2)
    train_generator_MLE(gen, gen_optimizer, oracle, train_loader, MLE_TRAIN_EPOCHS)

    # torch.save(gen.state_dict(), pretrained_gen_path)
    # gen.load_state_dict(torch.load(pretrained_gen_path))

    # PRETRAIN DISCRIMINATOR
    print('\nStarting Discriminator Training...')
    dis_optimizer = optim.Adagrad(dis.parameters())
    train_discriminator(dis, dis_optimizer, train_loader, gen, oracle, 50, 3)

    # torch.save(dis.state_dict(), pretrained_dis_path)
    # dis.load_state_dict(torch.load(pretrained_dis_path))

    # ADVERSARIAL TRAINING
    print('\nStarting Adversarial Training...')
    oracle_loss = helpers.batchwise_oracle_nll(gen, oracle, POS_NEG_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN,
                                               start_letter=START_LETTER, gpu=CUDA)
    print('\nInitial Oracle Sample Loss : %.4f' % oracle_loss)

    for epoch in range(ADV_TRAIN_EPOCHS):
        print('\n--------\nEPOCH %d\n--------' % (epoch+1))
        # TRAIN GENERATOR
        print('\nAdversarial Training Generator : ', end='')
        sys.stdout.flush()
        train_generator_PG(gen, gen_optimizer, oracle, dis, 1)

        # TRAIN DISCRIMINATOR
        print('\nAdversarial Training Discriminator : ')
        train_discriminator(dis, dis_optimizer, train_loader, gen, oracle, 5, 3)
        torch.save(gen, 'netG_adv_{}.pt'.format(epoch))
        torch.save(dis, 'netD_adv_{}.pt'.format(epoch))

    sentences = gen.sample(32)
    for i in range(sentences.shape[0]):
        for j in range(sentences.shape[1]):
            print(inv_vocab[sentences[i][j].item()], end=' ')
        print('\n')

