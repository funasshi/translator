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
        # input       (入力単語数)
        # embedded    (入力単語数,　batch=1 ,hidden_size)

        # output      (入力単語数,　batch=1 ,hidden_size)
        # hidden      (1,　batch=1 ,hidden_size)
        print(input.device)
        embedded = self.embedding(input).view(input.shape[0], 1, -1)
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


class AttnDecoderRNN_default(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        """
        hidden_size:    隠れ状態ベクトルの大きさ
        output_size:    出力ベクトルの大きさ = 全単語数
        dropout_p:    nn.Dropoutでdropoutさせる割合
        max_length:     最長の文の長さ = 今回は10
        """

        super(AttnDecoderRNN_default, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        input:    (1, 1) = (batch_size, 入力単語数)
        hidden:    (1, 1, hidden_size)
        encoder_outputs:    (max_length, hidden_size)
        """

        embedded = self.embedding(input).view(1, 1, -1)  # (1, 1, hidden_size)
        embedded = self.dropout(embedded)  # (1, 1, hidden_size)

        concat = torch.cat((embedded[0], hidden[0]), dim=1)  # (1, hidden*2)

        ##### TODO
        attn = self.attn(concat)  # (1, hidden*2)  ->  (1, max_length)
        attn_weights = torch.nn.functional.softmax(attn)  # (1, max_length)

        # (1, 1, hidden_size) <= (1, 1, max_length) × (1, max_length, hidden_size)
        attn_applied = torch.bmm(attn_weights.view(1, *attn_weights.shape),
                                 encoder_outputs.view(1, *encoder_outputs.shape))
        #####

        concat = torch.cat((embedded[0], attn_applied[0]), 1)  # (1, hidden_size*2)

        # self.attn_combine: 全結合層
        attn_combine = self.attn_combine(concat).unsqueeze(0)  # (1, hidden_size*2) -> (1, 1, hidden_size)

        relu = F.relu(attn_combine)
        gru_output, hidden = self.gru(relu, hidden)

        output = F.log_softmax(self.out(gru_output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        # <SOS>と同じ
        return torch.zeros(1, 1, self.hidden_size)