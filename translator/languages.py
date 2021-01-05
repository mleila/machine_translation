import pickle

from torchtext.data import Dataset, Example, Field
from translator.data import make_spacy_tokenizer


class Language:

    def __init__(self, name:str, init_token: str="<sos>", eos_token: str="<eos>"):
        """
        name: 'en'
        """
        self.name = name.lower()
        self.tokenizer = make_spacy_tokenizer(name.lower())
        self.field = Field(
            tokenize=self.tokenizer,
            init_token=init_token,
            eos_token=eos_token,
            unk_token='<unk>',
            lower=True,
            use_vocab=True)
        self.init_token = init_token
        self.eos_token = eos_token

    @property
    def vocab(self):
        return self.field.vocab

    def build_vocab(self, data: list, max_size=10_000, min_freq=2):
        """
        data: [sentence1, sentence2, ...]
        """
        # build vocab
        FIELDS = [('text', self.field)]
        all_data = ' '.join(data)
        examples = [Example.fromlist([all_data], fields=FIELDS)]
        datatset = Dataset(examples, fields=FIELDS)
        self.field.build_vocab(datatset, max_size=max_size, min_freq=min_freq)

    def save_language(self, file_path):
        pickle.dump(self.field.vocab, open(file_path, "wb"))

    def load_language(self, file_path):
        self.field.vocab = pickle.load(open(file_path, "rb"))


