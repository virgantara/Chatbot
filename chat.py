#

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
from utils import *
from vocab import Voc
from model import *

USE_CUDA = False
#torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


save_dir = os.path.join("data", "save")

# Set checkpoint to load from; set to None if starting from scratch
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#``attn_model = 'general'``
#``attn_model = 'concat'``
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
corpus_name = "movie-corpus"
corpus = os.path.join("dataset", "movie-corpus")

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
datafile = os.path.join(corpus, "formatted_movie_lines.txt")

# voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
# Print some pairs to validate



loadFilename = os.path.join(save_dir, model_name, corpus_name, "2-2_500/4000_checkpoint.tar")
voc = Voc(corpus_name)  # Replace with your vocabulary object

checkpoint = torch.load(loadFilename)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']


# Initialize word embeddings
# embedding = nn.Embedding(voc.num_words, hidden_size)

embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

max_length = 10

# checkpoint = torch.load('data/save/cb_model/movie-corpus/2-2_500/4000_checkpoint.tar', map_location=device)
# encoder.load_state_dict(checkpoint['en'])
# decoder.load_state_dict(checkpoint['de'])
encoder.eval()
decoder.eval()

# Set dropout layers to ``eval`` mode
# encoder.eval()
# decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder, device)

# Begin chatting (uncomment and run the following line to begin)
# evaluateInput(encoder, decoder, searcher, voc)
# Evaluation function
def evaluate(searcher, voc, sentence, max_length=max_length):
    # Preprocess sentence
    sentence = normalizeString(sentence)
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)

    # Generate response
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]

    # Filter out "EOS" token and any padding
    decoded_words = [word for word in decoded_words if word != 'EOS']
    return ' '.join(decoded_words)

# Function to interact with the chatbot
def chatbot_response(input_sentence):
    response = evaluate(searcher, voc, input_sentence)
    return response

# Testing the chatbot
if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot_response(user_input)
        print("Bot:", response)