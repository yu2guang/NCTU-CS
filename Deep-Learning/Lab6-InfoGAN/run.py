import torch as t
import torchvision.utils as vutils

import params
import dataloader as dl


def train(train_loader, netG, netDQFront, netD, netQ, optimG, optimD, epoch_i, testNoise, file_Gloss, file_Dloss, file_Qloss, file_proT, file_proF1, file_proF2):
    data_len = len(train_loader)
    G_loss_part, D_loss_part, Q_loss_part, Q_acc_part = 0, 0, 0, 0
    T_pro_part, F1_pro_part, F2_pro_part = 0, 0, 0

    for i, (batch_x, batch_y) in enumerate(train_loader, 0):
        batch_i = i+1
        batch_size = batch_x.size(0)

        real_label = t.full((batch_size,), 1).cuda()
        fake_label = t.full((batch_size,), 0).cuda()

        # real data
        batch_x = batch_x.cuda()

        # noise
        noise, cate_noise = dl.random_noise(batch_size)

        # # forward
        # G_out = netG(noise)
        # D_real_out = netD(netDQFront(batch_x))
        # D_fake_out = netD(netDQFront(G_out))
        # Q_out = netQ(netDQFront(G_out))

        ############################
        # (1) Update D network
        ############################

        optimD.zero_grad()
        netD.zero_grad()

        # train with real
        D_real_out = netDQFront(batch_x)
        D_real_out = netD(D_real_out)
        T_pro_part += D_real_out.sum().item()
        D_real_loss = params.BCE(D_real_out, real_label)

        # train with fake
        G_out = netG(noise)
        D_fake_out = netDQFront(G_out)
        D_fake_out = netD(D_fake_out)
        F1_pro_part += D_fake_out.sum().item()
        D_fake_loss = params.BCE(D_fake_out, fake_label)

        # update
        D_loss = D_real_loss + D_fake_loss
        D_loss_part += D_loss.item()
        D_loss.backward(retain_graph=True)
        optimD.step()


        ############################
        # (2) Update G & Q network
        ############################
        optimG.zero_grad()
        netG.zero_grad()
        netQ.zero_grad()

        # train with D
        D_G_out = netDQFront(G_out)
        D_G_out = netD(D_G_out)
        D_G_loss = params.BCE(D_G_out, real_label)

        # train with Q
        Q_out = netDQFront(G_out).view(-1, 8192)
        Q_out = netQ(Q_out)
        Q_loss = params.CE(Q_out, cate_noise)
        Q_loss_part += Q_loss.item()

        ## Q accuracy
        Q_out_index = t.max(Q_out, 1)[1]
        Q_hat = sum(Q_out_index == cate_noise).item()
        Q_acc_part += Q_hat

        # update
        G_loss = D_G_loss + Q_loss
        G_loss_part += G_loss.item()

        G_loss = 10*D_G_loss + 100*Q_loss
        G_loss.backward()
        optimG.step()

        # fake data after updating G
        G_out2 = netG(noise)
        D_G_out2 = netDQFront(G_out2)
        D_G_out2 = netD(D_G_out2)
        F2_pro_part += D_G_out2.sum().item()

        if(batch_i%params.PRINT_EVERY==0):
            G_loss_part /= params.PRINT_EVERY
            D_loss_part /= params.PRINT_EVERY
            Q_loss_part /= params.PRINT_EVERY
            Q_acc_part /= (params.PRINT_EVERY*batch_size)
            T_pro_part /= (params.PRINT_EVERY*batch_size)
            F1_pro_part /= (params.PRINT_EVERY*batch_size)
            F2_pro_part /= (params.PRINT_EVERY*batch_size)
            print('batch %d/%d: loss=(%4f, %4f, %4f), probability=(%4f, %4f, %4f), Q_accuracy=%4f'
                    % (batch_i, data_len, G_loss_part, D_loss_part, Q_loss_part, T_pro_part, F1_pro_part, F2_pro_part, Q_acc_part))

            # write file
            file_Gloss.write('%f\n' % (G_loss_part))
            file_Dloss.write('%f\n' % (D_loss_part))
            file_Qloss.write('%f\n' % (Q_loss_part))
            file_proT.write('%f\n' % (T_pro_part))
            file_proF1.write('%f\n' % (F1_pro_part))
            file_proF2.write('%f\n' % (F2_pro_part))

            G_loss_part, D_loss_part, Q_loss_part, Q_acc_part = 0, 0, 0, 0
            T_pro_part, F1_pro_part, F2_pro_part = 0, 0, 0

            # test
            test(netG, testNoise, epoch_i, batch_i)


    # do checkpointing
    t.save(netG.state_dict(), '%s/netG_%d.pkl' % (params.RESULT_FILE, epoch_i))
    t.save(netDQFront.state_dict(), '%s/netDQFront_%d.pkl' % (params.RESULT_FILE, epoch_i))
    t.save(netD.state_dict(), '%s/netD_%d.pkl' % (params.RESULT_FILE, epoch_i))
    t.save(netQ.state_dict(), '%s/netQ_%d.pkl' % (params.RESULT_FILE, epoch_i))


def test(netG, noise, epoch_i, batch_i):
    netG.eval()

    numbers, _ = dl.create_categorical_noise(params.C_SIZE, True)
    outGs = []
    with t.no_grad():
        for i in range(10):  # different noise
            for j in range(10):  # different number
                noise_cat = t.cat((noise[i], numbers[j]), dim=0).view(1, 64, 1, 1)
                noise_cat = noise_cat.type(t.FloatTensor).cuda()
                outG = netG(noise_cat)
                outGs.append(outG)

    images = t.cat(outGs, dim=0)
    vutils.save_image(images,
                      '%s/testImg_%d_%d.png' % (params.RESULT_FILE, epoch_i, batch_i),
                      normalize=True,
                      nrow=10)

    netG.train()
