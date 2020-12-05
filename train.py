import torch
from torch import nn
import random
from data_processor import *
import time
import math
from torch import optim
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_lang, output_lang, pairs=prepareData("eng","deu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH=10


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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH):
    # エンコーダの最初の隠れ状態

    input_tensor=input_tensor.to(device)
    target_tensor=target_tensor.to(device)
    encoder_hidden = encoder.initHidden().to(device)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    ### エンコーダ

    # 入力文の単語の数だけ繰り返しエンコーダに入力します
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    ### デコーダ

    # デコーダの最初の入力: <SOS>
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # デコーダの最初の隠れ状態: context vector
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)  # デコーダ出力の中で一番大きい値とそのインデックス
        decoder_input = topi.squeeze().detach()  # 勾配計算されないようにdetachします

        loss += criterion(decoder_output, target_tensor[di])  # 出力とターゲットの誤差を追加していきます
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang)  # 1.前処理で作成したpairからn_iter分だけ抽出
                      for i in range(n_iters)]
    criterion = nn.NLLLoss() # 誤差関数の定義

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]  # 仏
        target_tensor = training_pair[1]  #英

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # lossを計算してprint
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        # lossを計算してplotの準備
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

hidden_size = 150
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = AttnDecoderRNN_default(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)


n_iters=int(input("n_iters:"))
trainIters(encoder, decoder, n_iters, print_every=5000)


encoder_path = 'encoder.pth'
decoder_path = 'decoder.pth'

torch.save(encoder.to('cpu').state_dict(), encoder_path)
torch.save(decoder.to('cpu').state_dict(), decoder_path)