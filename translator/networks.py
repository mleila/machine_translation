import random

import torch
from torch import nn
from torch.nn.modules import dropout


class Encoder(nn.Module):
    def __init__(self, sequenec_size, embedding_dim, hidden_size, num_layers, dropout_prob=0.5):
        super(Encoder, self).__init__()

        self.Embedding = nn.Embedding(
            num_embeddings=sequenec_size,
            embedding_dim=embedding_dim
            )

        self.RNN = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob
        )

    def forward(self, x):
        # x -> (sequenec_size, batch_size) every token represents a word and it is given by an integer specifying
        # the words index in the vocab

        embedding = self.Embedding(x)
        # embedding -> (sequenec_size, batch_size, embedding_dim) this will embed the word into a higher dimension

        output, (h_0, c_0) = self.RNN(embedding)
        # output -> (sequenec_size, batch_size, hidden_size)

        return h_0, c_0


class Decoder(nn.Module):

    def __init__(self, sequenec_size, embedding_dim, output_size, hidden_size, num_layers, dropout_prob=0.5):
        super(Decoder, self).__init__()
        self.Embedding = nn.Embedding(
            num_embeddings=sequenec_size,
            embedding_dim=embedding_dim
            )

        self.RNN = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob
        )

        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, h_0, c_0):
        # x shape: (batch_size) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.Embedding(x)
        # embedding -> (1, batch_size, embedding_size)

        outputs, (h_0, c_0) = self.RNN(embedding, (h_0, c_0))
        # outputs shape -> (1, batch_size, hidden_size)

        predictions = self.fc(outputs)
        # predictions -> (1, batch_size, sequenec_size)

        return predictions.squeeze(0), h_0, c_0


class Encode_Decoder_Model(nn.Module):

    def __init__(self, encoder, decoder, device):
        super(Encode_Decoder_Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, target_vocab_size, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs
