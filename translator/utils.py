import random
from pathlib import Path
from numpy.core.fromnumeric import sort

import torch
import spacy


class Directory:

    def __init__(self, root_path: str):
        self.root_path = Path(root_path)

        self.data_path = self.root_path / Path('data')
        self.models_path = self.data_path / Path('models')
        self.languages_path = self.data_path / Path('languages')


    def get_latest_checkpint(self, model_name):
        model_path = self.models_path / Path(model_name)
        checkpoints = [f for f in model_path.glob('**/*') if f.is_file()]
        if not checkpoints:
            return
        checkpoints = sort(checkpoints)
        return checkpoints[-1]


def tokenize_sent(sentence, src_language, language_field, device):

    # load language model
    spacy_lang = spacy.load(src_language)

    # tokenize
    tokens = [token.text.lower() for token in spacy_lang(sentence)]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, language_field.init_token)
    tokens.append(language_field.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [language_field.vocab.stoi[token] for token in tokens]

    # Go through each german token and convert to an index
    text_to_indices = [language_field.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    return sentence_tensor


def translate_sentence(model, sentence_tensor, dest_language, language_field, device, max_length=50):
    # Build encoder hidden, cell state
    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [language_field.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == language_field.vocab.stoi["<eos>"]:
            break

    translated_sentence = [language_field.vocab.itos[idx] for idx in outputs]

    # remove start token
    return ' '.join(translated_sentence[1:])


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
