import re
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import fasttext
import fasttext.util

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

class Language:
    
    def __init__(self, name):
        self.name = name
        self.wordToIndex = {}
        self.wordToCount = {}
        self.indexToWord = {0: 'SOS', 1: 'EOS'}
        self.n_words = 2  # count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.wordToIndex:
            self.wordToIndex[word] = self.n_words
            self.wordToCount[word] = 1
            self.indexToWord[self.n_words] = word
            self.n_words += 1
        else:
            self.wordToCount[word] += 1

class Embedding:

    def __init__(self, vocab_size, embedding_dim):
        self.embedding_matrix = torch.empty(vocab_size, embedding_dim, dtype=torch.float32)
        self.missing_indices = [0, 1] # include SOS and EOS

def getData(file_name, reverse=False):

    # read file

    def normaliseString(s):
        s = s.lower().strip()
        s = re.sub(r'([.!?])', r' \1', s)
        s = re.sub(r'[^a-zA-ZÀ-ÿ.!?]+', r' ', s)
        return s.strip()

    print('Reading lines...')

    lang1, lang2 = file_name.split('-')[0], file_name.split('-')[1]

    lines = open(f'data/{file_name}.txt', encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normaliseString(s) for s in l.split('\t')] for l in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Language(lang2)
        output_lang = Language(lang1)
    else:
        input_lang = Language(lang1)
        output_lang = Language(lang2)

    # process data

    def isValidPair(p):
        eng_prefixes = (
            'i am ', 'i m ',
            'he is', 'he s ',
            'she is', 'she s ',
            'you are', 'you re ',
            'we are', 'we re ',
            'they are', 'they re ')
        
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)

    def filterPairs(pairs):
        return [pair for pair in pairs if isValidPair(pair)]
    
    print('Read %s sentence pairs' % len(pairs))
    
    pairs = filterPairs(pairs)    
    print('Trimmed to %s sentence pairs' % len(pairs))
    
    print('Counting words...')
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    print('Counted words:')
    print(f'> Input language: {input_lang.name} ({input_lang.n_words} words)')
    print(f'> Output language: {output_lang.name} ({output_lang.n_words} words)')
    
    return input_lang, output_lang, pairs

def getDataLoader(batch_size, input_lang, output_lang, pairs):

    def indexesFromSentence(lang, sentence):
        return [lang.wordToIndex[word] for word in sentence.split(' ')]

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp) # input sentence indices
        tgt_ids = indexesFromSentence(output_lang, tgt) # target sentence indices
        
        inp_ids.append(EOS_TOKEN)
        tgt_ids.append(EOS_TOKEN)
        
        input_ids[idx, :len(inp_ids)] = inp_ids # input_ids (n, MAX_LENGTH)
        target_ids[idx, :len(tgt_ids)] = tgt_ids # target_ids (n, MAX_LENGTH)

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    return train_dataloader

def getEmbeddings(wordToIndex_fr, wordToIndex_en, embedding_dim=300, download=True):
    
    if download:
        fasttext.util.download_model('fr', if_exists='ignore')
        fasttext.util.download_model('en', if_exists='ignore')

    ft_fr_300 = fasttext.load_model(f'cc.fr.300.bin')
    ft_en_300 = fasttext.load_model(f'cc.en.300.bin')

    ft_fr = fasttext.util.reduce_model(ft_fr_300, embedding_dim)
    ft_en = fasttext.util.reduce_model(ft_en_300, embedding_dim)

    embedding_fr = Embedding(len(wordToIndex_fr) + 2, embedding_dim) # include SOS and EOS
    embedding_en = Embedding(len(wordToIndex_en) + 2, embedding_dim) # include SOS and EOS

    words_fr = ['SOS', 'EOS']
    for word, index in wordToIndex_fr.items():
        if word in ft_fr:
            embedding_vector = torch.from_numpy(ft_fr.get_word_vector(word)).to(torch.float32)
            embedding_fr.embedding_matrix[index] = embedding_vector
        else:
            embedding_fr.missing_indices.append(index)
            words_fr.append(word)

    words_en = ['SOS', 'EOS']
    for word, index in wordToIndex_en.items():
        if word in ft_en:
            embedding_vector = torch.from_numpy(ft_en.get_word_vector(word)).to(torch.float32)
            embedding_en.embedding_matrix[index] = embedding_vector
        else:
            embedding_en.missing_indices.append(index)
            words_en.append(word)
    
    print('Missing words:')
    print('French:')
    print(f'> Count: {len(embedding_fr.missing_indices)} / {len(wordToIndex_fr) + 2} ({len(embedding_fr.missing_indices) / (len(wordToIndex_fr) + 2) * 100:.3f}%)')
    print(f'> Indices: {embedding_fr.missing_indices}')
    print(f'> Words: {words_fr}')
    print('English:')
    print(f'> Count: {len(embedding_en.missing_indices)} / {len(wordToIndex_en) + 2} ({len(embedding_en.missing_indices) / (len(wordToIndex_en) + 2) * 100:.3f}%)')
    print(f'> Indices: {embedding_en.missing_indices}')
    print(f'> Words: {words_en}')
    
    return embedding_fr, embedding_en
