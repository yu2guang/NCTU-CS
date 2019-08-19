import torch.nn as nn
from torch import optim
import time
import numpy as np

import models
import params
from prepare_data import preData, tenseLoader


if __name__ == '__main__':

    # 4 tense into pairs
    preData(params.PRE_YET)

    # open file to store
    file_loss = open('./%d/loss.txt'%params.NUM, 'w', encoding='UTF-8')
    file_KL = open('./%d/KL.txt'%params.NUM, 'w', encoding='UTF-8')
    file_bleu = open('./%d/bleu.txt'%params.NUM, 'w', encoding='UTF-8')
    file_bleuTest1 = open('./%d/bleuTest1.txt' % params.NUM, 'w', encoding='UTF-8')

    # train
    encoder = models.EncoderRNN().cuda()
    decoder = models.DecoderRNN().cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=params.LR)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=params.LR)

    cond_embed = nn.Embedding(4, params.COND_EMBEDDING_SIZE).cuda()
    cond_embed_optimizer = optim.Adam(cond_embed.parameters(), lr=params.LR)

    for epoch_i in range(1, params.EPOCH+1):
        print('\n#####Start epoch %d#####'%epoch_i, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('Training...')

        # load data
        train_data = tenseLoader('train')
        train_len = train_data.len()

        params.KL_W = 0.0
        total_bleu = 0.0
        total_loss = 0.0
        total_KL = 0.0
        bleu_400 = 0.0
        loss_400 = 0.0
        KL_400 = 0.0
        random_list=np.random.permutation(train_len)
        for train_i in range(1, train_len+1): #, train_len+1
            input_tensor, input_cond, target_tensor, target_cond = train_data.getPair(random_list[train_i-1])
            input, output, target, use_TF, bleu4, loss, KLloss = models.train(input_tensor, target_tensor, input_cond, target_cond, encoder, decoder, cond_embed, encoder_optimizer, decoder_optimizer, cond_embed_optimizer)
            models.KL_anneal(train_i)

            total_bleu += bleu4
            total_loss += loss
            total_KL += KLloss
            bleu_400 += bleu4
            loss_400 += loss
            KL_400 += KLloss
            file_bleu.write('%f\n' % (bleu4))
            file_loss.write('%f\n' % (loss))
            file_KL.write('%f\n' % (KLloss))

            if(train_i%400==0):
                print('vocab %d/%d: %s/%s(%s), bleu=%4f, loss=%4f, KL=%4f' % (train_i, train_len, input, output, target, bleu_400/400.0, loss_400/400.0, KL_400/400.0))
                bleu_400 = 0.0
                loss_400 = 0.0
                KL_400 = 0.0

        total_bleu /= train_len
        total_loss /= train_len
        total_KL /= train_len
        print('epoch %d: bleu=%4f, loss=%4f, KL=%4f'%(epoch_i,total_bleu,total_loss,total_KL))

        models.lr_decay(encoder_optimizer)
        models.lr_decay(decoder_optimizer)
        models.lr_decay(cond_embed_optimizer)
        models.TF_ratio()

        # test
        avg_bleu4 = models.test1(epoch_i, encoder, decoder, cond_embed)
        file_bleuTest1.write('%f\n' % avg_bleu4)

        models.test2(decoder, cond_embed)

        print('#####End epoch %d#####' % epoch_i, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # close file
    file_loss.close()
    file_KL.close()
    file_bleu.close()
    file_bleuTest1.close()

