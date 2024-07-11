import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

SOS_TOKEN = 0
EOS_TOKEN = 1
MAX_LENGTH = 20

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, embedding=None, dropout_p=0.1):

        super().__init__()
        
        self.hidden_size = hidden_size
        
        if embedding is not None: # pre-trained embeddings
            self.embedding = nn.Embedding.from_pretrained(embedding.embedding_matrix, freeze=True)

            for index in embedding.missing_indices:
                nn.init.uniform_(self.embedding.weight[index], -0.1, 0.1)
                self.embedding.weight[index].requires_grad = True
        else:
            self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):

        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        
        return output, hidden

class Attention(nn.Module):

    def __init__(self, hidden_size):
        
        super().__init__()
        
        self.Wa = nn.Linear(hidden_size, hidden_size) # applied to query (decoder previous hidden)
        self.Ua = nn.Linear(hidden_size, hidden_size) # applied to keys (encoder outputs)
        self.Va = nn.Linear(hidden_size, 1) # applied to above combination (query + keys)

    def forward(self, query, keys):

        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, n_layers, embedding=None, dropout_p=0.1):

        super().__init__()

        self.hidden_size = hidden_size
        
        if embedding is not None: # pre-trained embeddings
            self.embedding = nn.Embedding.from_pretrained(embedding.embedding_matrix, freeze=True)

            for index in embedding.missing_indices:
                nn.init.uniform_(self.embedding.weight[index], -0.1, 0.1)
                self.embedding.weight[index].requires_grad = True
        else:
            self.embedding = nn.Embedding(output_size, hidden_size)

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, n_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_TOKEN)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, weights  = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(weights)

            if target_tensor is not None:
                # teacher forcing
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                # no teacher forcing
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach() # detach required since target value derived from previous computations

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):

        embedded = self.dropout(self.embedding(input))
        
        query = hidden.permute(1, 0, 2)
        query = query[:, -1, :].unsqueeze(1) # extract last layer (for multiple layers)

        context, weights = self.attention(query, encoder_outputs)

        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, weights

def train_epoch(dataloader, encoder, decoder, encoder_optimiser, decoder_optimiser, criterion):

    total_loss = 0

    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimiser.zero_grad()
        decoder_optimiser.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1))
        loss.backward()

        encoder_optimiser.step()
        decoder_optimiser.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train(train_dataloader, encoder, decoder, epochs, learning_rate=0.001, print_every=100, plot_every=100):

    def asMinutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def timeSince(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
    def showPlot(points):
        fig, ax = plt.subplots()
        loc = ticker.MultipleLocator(base=0.2)
        ax.yaxis.set_major_locator(loc)
        ax.plot(points)
        plt.show()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # reset every print_every
    plot_loss_total = 0  # reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / epochs),
                                        epoch, epoch / epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def decode(encoder, decoder, sentence, input_lang, output_lang):

    def indexesFromSentence(lang, sentence):
        return [lang.wordToIndex[word] for word in sentence.split(' ')]

    def tensorFromSentence(lang, sentence):
        indexes = indexesFromSentence(lang, sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

    with torch.no_grad():

        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.indexToWord[idx.item()])
    
    return decoded_words

def evaluate(encoder, decoder, input_lang, output_lang, pairs, n=10):

    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = decode(encoder, decoder, pair[0], input_lang, output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
