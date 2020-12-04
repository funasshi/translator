import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        hidden_size:    隠れ状態ベクトルの大きさ
        input_size:     入力ベクトルの大きさ = 全単語数
        """

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    # 最初の単語に対する prev_hiddenを定義します
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)  # <SOS>


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size,output_size, dropout_p=0.1):
        """
        hidden_size:    隠れ状態ベクトルの大きさ
        output_size:    出力ベクトルの大きさ = 全単語数
        dropout_p:    nn.Dropoutでdropoutさせる割合
        """

        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        input:    (1, 1) = (batch_size, 入力単語数)
        hidden:    (1, 1, hidden_size)
        encoder_outputs:    (max_length, hidden_size)
        """

        embedded = self.embedding(input).view(1, 1, -1)  # (1, 1, hidden_size)
        # embedded = self.dropout(embedded)  # (1, 1, hidden_size)
        gru_output, hidden = self.gru(embedded, hidden) #gru_output,hidden (1,1,hidden_size)
        s=torch.mv(encoder_outputs,gru_output.view(-1))
        a=F.softmax(s)
        c=torch.t(torch.mv(torch.t(encoder_outputs), a))
        output = F.log_softmax(self.out(c)).view(1,-1)
        return output, hidden

    def initHidden(self):
        # <SOS>と同じ
        return torch.zeros(1, 1, self.hidden_size)