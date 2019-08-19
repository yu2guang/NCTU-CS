import torch as t
import torch.nn as nn
import random
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from prepare_data import SOS_token, EOS_token, tenseLoader
import params


def asc2str(ascTensor):
    res = ''
    for c in range(len(ascTensor) - 1):
        c_tmp = ascTensor[c].item()
        res += chr(c_tmp + 96)
    return res


# compute BLEU-4 score
# between output and truth
def compute_bleu(output, reference):
    cc = SmoothingFunction()
    return sentence_bleu([reference], output, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc.method1)


def lr_decay(optim):
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr'] * params.LR_DECAY


def KL_anneal(iter_i):
    params.KL_W = min(0.5, params.KL_W + 0.00025)


def TF_ratio():
    params.TEACHING_FORCE_RATIO = max(0.5, params.TEACHING_FORCE_RATIO - 0.025)


def sample_z(mu, log_var, en_out_len):
    eps = t.randn(en_out_len).view(1,1,-1).cuda()
    return mu + t.exp(log_var / 2) * eps


def compute_loss(z_mu, z_var, recon_loss):
    KLloss = 0.5 * t.sum(t.exp(z_var) + z_mu ** 2 - z_var - 1.)
    loss = recon_loss + params.KL_W * KLloss
    return loss, KLloss


class EncoderRNN(nn.Module):
    def __init__(self):
        super(EncoderRNN, self).__init__()
        self.hidden_size = params.HIDDEN_SIZE - params.COND_EMBEDDING_SIZE

        self.embedding = nn.Embedding(params.VOCAB_DICT_SIZE, params.HIDDEN_SIZE).cuda()
        self.gru = nn.GRU(params.HIDDEN_SIZE, params.HIDDEN_SIZE).cuda()

        self.mulinear = nn.Linear(params.HIDDEN_SIZE, params.LATENT_SIZE).cuda()
        self.varlinear = nn.Linear(params.HIDDEN_SIZE, params.LATENT_SIZE).cuda()

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def reparameterization_trick(self, hidden):
        mean = self.mulinear(hidden)
        variance = self.varlinear(hidden)
        en_outputs = sample_z(mean, variance, params.LATENT_SIZE).view(1, 1, -1).cuda()

        return mean, variance, en_outputs

    def initHidden(self):
        return t.zeros((1, 1, self.hidden_size)).cuda()


# Decoder
class DecoderRNN(nn.Module):
    def __init__(self):
        super(DecoderRNN, self).__init__()
        self.first_pass = True

        self.embedding = nn.Embedding(params.VOCAB_DICT_SIZE, params.HIDDEN_SIZE).cuda()
        self.linear40 = nn.Linear(params.LATENT_SIZE + params.COND_EMBEDDING_SIZE, params.HIDDEN_SIZE)
        self.gru = nn.GRU(params.HIDDEN_SIZE, params.HIDDEN_SIZE).cuda()
        self.out = nn.Linear(params.HIDDEN_SIZE, params.VOCAB_DICT_SIZE)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, z):
        output = self.embedding(input).view(1, 1, -1)
        # output = funct.relu(output)
        if (self.first_pass):
            z = self.linear40(z)
            self.first_pass = False
        output, hidden = self.gru(output, z)
        output = self.out(output[0])
        # output = self.softmax(output) NLLloss
        return output, hidden


def train(input_tensor, target_tensor, input_cond, target_cond, encoder, decoder, cond_embed, en_optim, de_optim, cond_embed_optim):
    en_hidden = encoder.initHidden()
    decoder.first_pass = True

    en_optim.zero_grad()
    de_optim.zero_grad()
    cond_embed_optim.zero_grad()
    input_len = input_tensor.size(0)
    target_len = target_tensor.size(0)

    # ----------sequence to sequence part for encoder----------#
    input_cond = cond_embed(input_cond).view(1, 1, -1)
    en_hidden = t.cat([en_hidden, input_cond], 2)  ###CVAE
    for di in range(input_len):
        en_hidden = encoder(input_tensor[di], en_hidden)

    mean, variance, en_outputs = encoder.reparameterization_trick(en_hidden)

    # ----------sequence to sequence part for decoder----------#
    de_input = t.tensor([[SOS_token]]).cuda()
    target_cond = cond_embed(target_cond).view(1, 1, -1)
    de_hidden = t.cat([en_outputs, target_cond], 2)  ###CVAE
    use_teacher_forcing = True if random.random() < params.TEACHING_FORCE_RATIO else False
    output_tensor = []
    recon_loss = 0
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_len):
            de_output, de_hidden = decoder(de_input, de_hidden)
            topv, topi = de_output.topk(1)
            output_tensor.append(topi.squeeze().item())
            de_input = target_tensor[di]  # Teacher forcing
            recon_loss += params.CRITERATION(de_output, target_tensor[di])
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_len):
            de_output, de_hidden = decoder(de_input, de_hidden)
            topv, topi = de_output.topk(1)
            de_input = topi.squeeze().detach()  # detach from history as input
            output_tensor.append(de_input.item())
            recon_loss += params.CRITERATION(de_output, target_tensor[di])
            if de_input.item() == EOS_token:
                break

    loss, KLloss = compute_loss(mean, variance, recon_loss)
    loss.backward()

    en_optim.step()
    de_optim.step()
    cond_embed_optim.step()

    input = asc2str(input_tensor.view(1, -1).squeeze())
    target = asc2str(target_tensor.view(1, -1).squeeze())
    if len(output_tensor) != 0:
        output_tensor = t.cuda.LongTensor(output_tensor)
        output = asc2str(output_tensor.view(-1))
        bleu4 = compute_bleu(output, target)
    else:
        bleu4 = 0

    return input, output, target, use_teacher_forcing, bleu4, loss.item(), KLloss.item()


def test1(epoch_i, encoder, decoder, cond_embed):
    print('\nTesting 1...')
    test_data = tenseLoader('test')
    test_len = test_data.len()
    avg_bleu4 = 0.0
    for test_i in range(1, test_len + 1):
        input_tensor, input_cond, target_tensor, target_cond = test_data.getPair(test_i - 1)
        decoder.first_pass = True
        en_hidden = encoder.initHidden()

        input_len = input_tensor.size(0)
        target_len = target_tensor.size(0)

        # ----------sequence to sequence part for encoder----------#
        input_cond_embed = cond_embed(input_cond).view(1, 1, -1)
        en_hidden = t.cat([en_hidden, input_cond_embed], 2)  ###CVAE
        for di in range(input_len):
            en_hidden = encoder(input_tensor[di], en_hidden)

        mean, variance, en_outputs = encoder.reparameterization_trick(en_hidden)

        # ----------sequence to sequence part for decoder----------#
        de_input = t.tensor([[SOS_token]]).cuda()
        target_cond_embed = cond_embed(target_cond).view(1, 1, -1)
        de_hidden = t.cat([en_outputs, target_cond_embed], 2)  ###CVAE
        output_tensor = []
        for di in range(target_len):
            de_output, de_hidden = decoder(de_input, de_hidden)
            topv, topi = de_output.topk(1)
            de_input = topi.squeeze().detach()  # detach from history as input
            output_tensor.append(de_input.item())
            if de_input.item() == EOS_token:
                break

        input = asc2str(input_tensor.view(1, -1).squeeze())
        target = asc2str(target_tensor.view(1, -1).squeeze())

        if len(output_tensor) != 0:
            output_tensor = t.cuda.LongTensor(output_tensor)
            output = asc2str(output_tensor.view(-1))
            bleu4 = compute_bleu(output, target)
        else:
            bleu4 = 0

        avg_bleu4 += bleu4

        print('vocab %d/%d: %s/%s(%s), %s->%s, bleu=%4f' % (
            test_i, test_len, input, output, target, params.COND[input_cond.item()], params.COND[target_cond.item()], bleu4))

    avg_bleu4 /= 10.0
    print('bleu4 avg score=%4f' % (avg_bleu4))
    if (avg_bleu4 > 0.5):
        # t.save(encoder.state_dict(),
        #        "./{num}/EncoderRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))
        # t.save(decoder.state_dict(),
        #        "./{num}/DecoderRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))
        # t.save(cond_embed.state_dict(),
        #        "./{num}/EmbedRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))
        t.save(encoder,
               "./{num}/EncoderRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))
        t.save(decoder,
               "./{num}/DecoderRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))
        t.save(cond_embed,
               "./{num}/EmbedRNN{num}_{score}_{e}.pkl".format(num=params.NUM, score=int(avg_bleu4 * 10), e=epoch_i))

    return avg_bleu4

def test2(decoder, cond_embed):
    print('\nTesting 2...')
    print('>', params.COND)
    for test_i in range(20):
        sample_G = t.randn(params.LATENT_SIZE).view(1,1,-1).cuda()

        # ----------sequence to sequence part for decoder----------#
        de_input = t.tensor([[SOS_token]]).cuda()

        output4 = []
        for cond in range(4):
            decoder.first_pass = True
            target_cond = t.tensor(cond).cuda()
            target_cond = cond_embed(target_cond).view(1, 1, -1)
            de_hidden = t.cat([sample_G, target_cond], 2)  ###CVAE

            output_tensor = []
            for di in range(30):
                de_output, de_hidden = decoder(de_input, de_hidden)
                topv, topi = de_output.topk(1)
                de_input = topi.squeeze().detach()  # detach from history as input
                output_tensor.append(de_input.item())
                if de_input.item() == EOS_token:
                    break

            if len(output_tensor) != 0:
                output_tensor = t.cuda.LongTensor(output_tensor)
                output = asc2str(output_tensor.view(-1))
                output4.append(output)
            else:
                output4.append('')

        print(output4)