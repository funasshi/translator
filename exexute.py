import torch
from torch import nn
import random
from data_processor import *
import time
import math
from torch import optim
from model import *

input_lang, output_lang, pairs=prepareData("eng","deu")

hidden_size=150
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN_default(hidden_size, output_lang.n_words, dropout_p=0.1)

encoder_path = 'encoder.pth'
decoder_path = 'decoder.pth'

encoder.load_state_dict(torch.load(encoder_path, map_location=torch.device('cpu')))
decoder.load_state_dict(torch.load(decoder_path, map_location=torch.device('cpu')))
SOS_token=0
EOS_token=1


def translate(input):
    input_tensor=tensorFromSentence(input_lang,input)
    output_list=[]
    input_length = input_tensor.size(0)  # 入力単語の数
    encoder_outputs = torch.zeros(10, encoder.hidden_size)

    encoder_hidden = encoder.initHidden()

    ### エンコーダ

    # 入力文の単語の数だけ繰り返しエンコーダに入力します
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    ### デコーダ

    # デコーダの最初の入力: <SOS>
    decoder_input = torch.tensor([[SOS_token]])

    # デコーダの最初の隠れ状態: context vector
    decoder_hidden = encoder_hidden

    while True:
        decoder_output, decoder_hidden ,attn_weights= decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)  # デコーダ出力の中で一番大きい値とそのインデックス
        decoder_input = topi.squeeze().detach()  # 勾配計算されないようにdetachします
        if decoder_input.item() == EOS_token:
            break
        else:
            output_list.append(output_lang.index2word[decoder_input.item()])

    return " ".join(output_list)


print(translate(input()))
