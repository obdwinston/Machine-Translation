import torch
from model import Decoder, Encoder, evaluate, train
from utils import getData, getDataLoader, getEmbeddings

device = (
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

input_lang, output_lang, pairs = getData('eng-fra', reverse=True)
train_dataloader = getDataLoader(64, input_lang, output_lang, pairs)
X, y = next(iter(train_dataloader))

# hidden_size = 200
batch_size = 32
n_layers = 2

embedding_fr, embedding_en = getEmbeddings(input_lang.wordToIndex, output_lang.wordToIndex, embedding_dim=200, download=True)
hidden_size = embedding_fr.embedding_matrix.size(1)

encoder = Encoder(input_lang.n_words, hidden_size, n_layers, embedding_fr).to(device)
decoder = Decoder(hidden_size, output_lang.n_words, n_layers, embedding_en).to(device)

train(train_dataloader, encoder, decoder, epochs=100, print_every=5, plot_every=5)

encoder.eval() # turns off dropout
decoder.eval() # turns off dropout
evaluate(encoder, decoder, input_lang, output_lang, pairs)
