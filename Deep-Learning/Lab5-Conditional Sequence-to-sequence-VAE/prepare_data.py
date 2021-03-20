from __future__ import unicode_literals, print_function, division
from io import open
import torch as t

SOS_token = 0
EOS_token = 27


def preData(pre):
    if(pre==False):
        lines = open('./data/train.txt', encoding='utf-8').read().strip().split('\n')
        train_file = open('./data/train2.txt', 'w', encoding='UTF-8')

        groups = [[word for word in l.split(' ')] for l in lines]
        for g in groups:
            for i in range(4):
                train_file.write('%s %d %s %d\n'%(g[i], i, g[i], i))

        train_file.close()


def read_data(mode):

    lines = open('./data/%s2.txt'%mode, encoding='utf-8').read().strip().split('\n')

    groups = []
    j = 0
    for l in lines:
        temp = []
        for i, word in enumerate(l.split(' ')):
            if((i==0)or(i==2)):
                asc = [ord(c) - 96 for c in word]
            else:
                asc = [int(word)]
            temp.append(asc)
        groups.append(temp)
        j+=1

    return groups


class tenseLoader:
    def __init__(self, mode):

        self.mode = mode
        self.groups = read_data(mode)

        print("> Found %d lines...(%s)" % (len(self.groups), mode))

    def len(self):
        return len(self.groups)

    def getPair(self, index):

        input = self.groups[index][0]
        input.append(EOS_token)
        input = t.tensor(input, dtype=t.long).view(-1, 1).cuda()

        input_cond = t.tensor(self.groups[index][1]).cuda()

        target = self.groups[index][2]
        target.append(EOS_token)
        target = t.tensor(target, dtype=t.long).view(-1, 1).cuda()

        target_cond = t.tensor(self.groups[index][3]).cuda()

        return input, input_cond, target, target_cond
