from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


#################################
#################################
train_flag = True
teacher_forcing_ratio = 0.5
learning_rate = 0.001
hidden_size = 256
n_iters = 10000
print_every = 500
random.seed = 1234

SOS_token = 0
EOS_token = 1
max_length = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

use_cuda = torch.cuda.is_available()
#################################
#################################



#################################
otput_file_name = "teacher_forcing_ratio=" + str(teacher_forcing_ratio) +\
                  "learning_rate=" + str(learning_rate) +\
                  "max_length=" + str(max_length) +\
                  "hidden_size=" + str(hidden_size) +\
                  "n_iters=" +  str(n_iters) +\
                  "print_every=" + str(print_every)

print("\noutput_filename=", otput_file_name)

print("\nHYPER_PARAMS:")
print("teacher_forcing_ratio=", teacher_forcing_ratio)
print("learning_rate=", learning_rate)
print("max_length=", max_length)
print("hidden_size=", hidden_size)
print("n_iters=", n_iters)
print("print_every=", print_every)
#################################



class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang  = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang  = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < max_length and \
        len(p[1].split(' ')) < max_length and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print("input laguage:", input_lang.name, ", vocab size", input_lang.n_words)
    print("output language:", output_lang.name,", vocab size", output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=max_length):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_output, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result

def variablesFromPair(pair):
    input_variable = variableFromSentence(input_lang, pair[0])
    target_variable = variableFromSentence(output_lang, pair[1])
    return (input_variable, target_variable)


'''
About the train method defined below

#################################################
####### Example conditions
#################################################

### We will assume that the

- input_variable: torch.LongTensor containing the indices of the words that
                  appear in the input sequence.

    Example a phrase of 7 words [123, 124, 101, 203, 2488, 5, 1]
                                (1 indicates end of the sequence)

- target cariable: torch.LongTensor containing the indices of the words that
                   appear in the input sequence.

    Example a phrase of 6 words [77, 78, 508, 1461, 4, 1]
                                (1 indicates end of the sequence).

   Notice that the indices of the input and output do not correspond to the same 
   language. For example 123 can correspond to the word "hello" in the input
   and 123 can correspond to the word "comer" in the output.


##### EXAMPLE

Let us consider the following inputs to the function

-  max_length = 10

-  input_variable
                Variable containing:
                  123
                  124
                  101
                  203
                 2488
                    5
                    1
                [torch.LongTensor of size 7x1]

- target_variable
                Variable containing:
                   77
                   78
                  508
                 1461
                    4
                    1
                [torch.LongTensor of size 6x1]

Then some of the intermediate quantities defined during the execution
of the train function and its values will be

-> SOS_token (defined as global variable)
---> SOS_token is 0

-> EOS_token (defined as global variable)
---> EOS_token is 1

-> input_length = input_variable.size()[0]
---> input_length --> 7

-> encoder_hidden = encoder.initHidden()
---> [torch.FloatTensor of size 1x1x256]

-> encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
---> [torch.FloatTensor of size 10x256]

-> decoder_output
---> [torch.FloatTensor of size 1x2925]

-> decoder_hidden
--->[torch.FloatTensor of size 1x1x256]

-> decoder_attention
---> [torch.FloatTensor of size 1x10]

-> topv, topi = decoder_output.data.topk(1)

---> topv (highest value in the tensor)  => -7.5980 [torch.FloatTensor of size 1x1]
    
    Notice this is the same (with some loss on decimal precision) as 
        np.max(decoder_output.data.numpy()) => -7.5979848

---> topi (index of the highest value in the tensor) => 1138 [torch.LongTensor of size 1x1]
         Notice this is the same as  np.argmax(decoder_output.data.numpy())=> 1138

-> Let us consider di is 0

-> criterion(decoder_output, target_variable[di])
---> Variable containing:
     7.9717
     [torch.FloatTensor of size 1]

-> target_variable[di]
---> Variable containing:
     77
     [torch.LongTensor of size 1]

-> decoder_output[0,77]
--->  Variable containing:
      -7.9717
      [torch.FloatTensor of size 1]


#################################################
####### About the loss and the criterion
#################################################

-> type(criterion)
---> torch.nn.modules.loss.NLLLoss

It seems that doing loss += criterion(decoder_output, target_variable[di]) we add
to the loss the strength of the connection of the decoder_output corresponding to
the input word.

We can use the loss with a FloatTensor and a LongTensor
    nn.NLLLoss(torch.FloatTensor,torch.LongTensor)

Notice that for an example with 
- input_variable [123, 124, 101, 203, 2488, 5, 1].T [torch.LongTensor of size 7x1]
- target_variable [77, 78, 508, 1461, 4, 1].T torch.LongTensor of size 6x1]

    di                                              => 0
    target_variable                                 => [torch.LongTensor of size 7x1]
    target_variable[di]                             => [torch.LongTensor of size 1]
    target_variable[di].numpy                       => 77
    decoder_output                                  => [torch.FloatTensor of size 1x2925]
    decoder_output[0,77]                            => -7.9717
    criterion(decoder_output, target_variable[di])  => 7.9717


#################################################
####### About encoder_outputs torch.Variable
#################################################

### Notice that the  encoder_outputs=Variable(torch.zeros(max_length, encoder.hidden_size)) 
### At the end of the "for ei in range(input_length)", the encoding loop, we have
### The following values in the encoder_outputs Variable

-> encoder_outputs
---> Variable containing:
    -0.0900 -0.1959 -0.0805  ...   0.3015  0.5032 -0.2351
     0.5738  0.0672 -0.3086  ...   0.3827 -0.0882  0.0687
     0.4895  0.1718  0.0472  ...   0.1263 -0.3028 -0.1114
              ...             ⋱             ...          
     0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
     0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
     0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
    [torch.FloatTensor of size 10x256]

### Notice since the target_variable has length 6 the row 7 contains all zeros
-> encoder_outputs[0:7,:]
---> Variable containing:
    -0.0900 -0.1959 -0.0805  ...   0.3015  0.5032 -0.2351
     0.5738  0.0672 -0.3086  ...   0.3827 -0.0882  0.0687
     0.4895  0.1718  0.0472  ...   0.1263 -0.3028 -0.1114
              ...             ⋱             ...          
     0.5111 -0.2856  0.0155  ...  -0.1554  0.2339  0.2660
     0.6114 -0.4986 -0.1778  ...   0.2450 -0.2981  0.0170
     0.0000  0.0000  0.0000  ...   0.0000  0.0000  0.0000
    [torch.FloatTensor of size 7x256]

#################################################
####### About decoder_output torch.Variable
#################################################

### Notice we have an input vocabulary of 4489 words and an output vocabulary
### of 2925 words

-> decoder_output
---> Variable containing:
     -8.1107 -8.0629 -7.9517  ...  -8.0785 -8.0255 -7.9740
     [torch.FloatTensor of size 1x2925]
'''


def train(input_variable,
          target_variable,
          encoder,
          decoder,
          encoder_optimizer,
          decoder_optimizer,
          criterion,
          max_length=max_length):
    
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    #encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder.forward(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    # In the first time step of the decoding process, the decoder_input is the SOS_token
    # This explicitly tells the decoder that a new sentence is starting.
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    # The decoder is initiallized with the hidden state of the encoder
    # This hidden state should capture the relevant info of the input_variable
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    


    # Without teacher forcing: use its own predictions as the next input
    for di in range(target_length):

        decoder_output, decoder_hidden, decoder_attention = decoder.forward(
                         decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        
        topv, topi = decoder_output.data.topk(1)

        # get an int value instead of a tensor
        ni = topi[0][0] 
        # use the int (word index) to define a new Variable, this new Variable is the
        # predicted sequence value in this time step, this value will be the input
        # of the decoder in the next time step. 
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        # Notice the loss in this example is the NLLLoss
        loss += criterion(decoder_output, target_variable[di])
        if ni == EOS_token:
            break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



'''
Example of input_variable, target_variable from the loop for iter in range(1, n_iters + 1)

-> input_variable
---> Variable containing:
      118
      246
      215
      247
      975
     1358
        5
        1
    [torch.LongTensor of size 8x1]

-> target_variable
---> Variable containing:
      130
       78
      148
      219
      598
        4
        1
    [torch.LongTensor of size 7x1]


#### About the nn.NLLLoss()
    criterion = nn.NLLLoss()

#### Example 1

-> target_variable[di]
    Variable containing:
     1
    [torch.LongTensor of size 1]

-> decoder_output
   -7.9144 -8.1044 -8.0803  ...  -7.8474 -8.1396 -8.0870
   [torch.FloatTensor of size 1x2925]

-> criterion(decoder_output, target_variable[di])    
    Variable containing:
    8.1044
    [torch.FloatTensor of size 1]

-> decoder_output[0,1]
    Variable containing:
    -8.1044
    [torch.FloatTensor of size 1]

#### Example 2

-> target_variable[1]
    Variable containing:
     16
    [torch.LongTensor of size 1]

-> decoder_output[0,16]
    Variable containing:
    -7.9383
    [torch.FloatTensor of size 1]

-> criterion(decoder_output, target_variable[1])
    Variable containing:
     7.9383
    [torch.FloatTensor of size 1]

'''
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
    
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=max_length):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder.forward(input_variable[ei],
                                                         encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder.forward(
            decoder_input, decoder_hidden, encoder_output, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, pairs, n=10):
    for i in range(n):
        pair = random.Random(i).choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


if train_flag == True:

    print("\nPreparing data...")
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    print("\nTraining...")
    hidden_size = hidden_size
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, 1, dropout_p=0.1)

    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every, learning_rate=learning_rate)

    print("\n\n")
    evaluateRandomly(encoder1, attn_decoder1, pairs )

