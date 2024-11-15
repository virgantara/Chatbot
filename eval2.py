import torch
import torch.nn.functional as F
from model import *  # Import your encoder and decoder models
from utils import *  # Import utility functions from tutorial
from vocab import *
# Parameters (replace with the parameters used during training)
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
max_length = 10  # Maximum sentence length to consider
attn_model = 'dot'  # Attention model type, e.g., 'dot', 'general', or 'concat'

# Load vocabulary and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
voc = Voc("chatbot_vocabulary")  # Replace with your vocabulary object
embedding = torch.nn.Embedding(voc.num_words, hidden_size)

# Load the encoder and decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout).to(device)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout).to(device)

# Load trained model parameters
checkpoint = torch.load('data/save/cb_model/movie-corpus/2-2_500/4000_checkpoint.tar', map_location=device)
encoder.load_state_dict(checkpoint['en'])
decoder.load_state_dict(checkpoint['de'])
encoder.eval()
decoder.eval()

# Greedy Search Decoder (inference)
class GreedySearchDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

# Instantiate searcher with trained models
searcher = GreedySearchDecoder(encoder, decoder)

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