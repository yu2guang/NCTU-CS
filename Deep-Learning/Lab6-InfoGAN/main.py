import torch as t
import time

import dataloader
import params
import models
import run

if __name__ == '__main__':

    # t.cuda.manual_seed(547600)  # gpu

    # open file to store
    file_Gloss = open('%s/Gloss.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')
    file_Dloss = open('%s/Dloss.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')
    file_Qloss = open('%s/Qloss.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')
    file_proT = open('%s/proT.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')
    file_proF1 = open('%s/proF1.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')
    file_proF2 = open('%s/proF2.txt' % params.RESULT_FILE, 'w', encoding='UTF-8')

    # load data
    train_loader = dataloader.loadTrainData()

    # net
    netG = models.Generator(inplace=True).cuda()
    netDQFront = models.DQFront(inplace=True).cuda()
    netD = models.Discriminator().cuda()
    netQ = models.Q().cuda()

    # init weight
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    netG.apply(weights_init)
    netDQFront.apply(weights_init)
    netD.apply(weights_init)
    netQ.apply(weights_init)

    # optimizer
    optimG = t.optim.Adam([{'params': netG.parameters()}, {'params': netQ.parameters()}], 
                            lr=params.LR_GEN_Q)
    optimD = t.optim.Adam([{'params': netDQFront.parameters()}, {'params': netD.parameters()}], 
                            lr=params.LR_DIS)

    # fixed test noise
    testNoise = dataloader.create_continuous_noise(params.C_SIZE)

    for epoch_i in range(1, params.EPOCH+1):
        print('\nepoch', epoch_i)
        print('START:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('loss=(G, D, Q), probability=(proT, proF1, proF2)')

        # train
        run.train(train_loader, netG, netDQFront, netD, netQ, optimG, optimD, epoch_i, testNoise,
                  file_Gloss, file_Dloss, file_Qloss, file_proT, file_proF1, file_proF2)


    # close file
    file_Gloss.close()
    file_Dloss.close()
    file_Qloss.close()
    file_proT.close()
    file_proF1.close()
    file_proF2.close()
