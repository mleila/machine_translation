import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import spacy
from torchtext.data import Field, TabularDataset


def make_spacy_tokenizer(lang='en'):
    lang = spacy.load('en')
    def tokenize_lang(sentence):
        return [token.text for token in lang.tokenizer(sentence)]
    return tokenize_lang


class EuroParl:

    def __init__(self, data_dir, lang1_name='A', lang2_name='B', max_sent_len=80, sample_size=None):
        self.data_dir = data_dir
        file_1 = [file for file in os.listdir(data_dir) if file.startswith('europarl') and file.endswith(f'.{lang1_name}')][0]
        file_2 = [file for file in os.listdir(data_dir) if file.startswith('europarl') and file.endswith(f'.{lang2_name}')][0]
        language_1 = open(f'{data_dir}/{file_1}', encoding='utf-8').read().split('\n')
        language_2 = open(f'{data_dir}/{file_2}', encoding='utf-8').read().split('\n')
        raw_data = {lang1_name : [line for line in language_1], lang2_name: [line for line in language_2]}
        self.df = pd.DataFrame(raw_data, columns=[lang1_name, lang2_name])

        if sample_size:
            print(f'sampling {sample_size} out of {len(self.df)} samples')
            self.df = self.df.sample(n=sample_size)

        # sentences and sentences where translations are not of roughly equal length
        self.df[f'{lang1_name}_len'] = self.df[lang1_name].str.count(' ')
        self.df[f'{lang2_name}_len'] = self.df[lang2_name].str.count(' ')
        self.df = self.df.query(f'{lang1_name}_len < {max_sent_len} & {lang2_name}_len < {max_sent_len}')
        self.df = self.df.query(f'{lang2_name}_len < {lang1_name}_len * 1.5 & {lang2_name}_len * 1.5 > {lang1_name}_len')

        self.train = None
        self.valid = None
        self.test = None


    def train_test_split(self, test_size):
        self.train, self.test = train_test_split(self.df, test_size=test_size)


    def train_valid_test_split(self, valid_size, test_size):
        self.train, self.test = train_test_split(self.df, test_size=test_size)
        self.train, self.valid = train_test_split(self.train, test_size=valid_size/(1-test_size))


    def to_csv(self, output_path):
        self.train_path = Path(output_path) / Path("train.csv")
        self.valid_path = Path(output_path) / Path("valid.csv")
        self.test_path = Path(output_path) / Path("test.csv")
        if self.train is not None:
            self.train.to_csv(self.train_path, index=False)
        if self.valid is not None:
            self.valid.to_csv(self.valid_path, index=False)
        if self.test is not None:
            self.test.to_csv(self.test_path, index=False)


def make_fields(lang1_name, lang2_name):
    first_tokenizer = make_spacy_tokenizer(lang1_name.lower())
    second_tokenizer = make_spacy_tokenizer(lang2_name.lower())
    lang1_field = Field(tokenize=first_tokenizer)
    lang2_field = Field(tokenize=second_tokenizer, init_token = "<sos>", eos_token = "<eos>")
    return lang1_field, lang2_field


def make_dataset(
    first_data_field,
    second_data_field,
    train_path,
    valid_path=None,
    test_path=None,
    max_size=10_000,
    min_freq=2
    ):
    data_fields = [first_data_field, second_data_field]
    train, val, test = TabularDataset.splits(
        path='./',
        train=train_path,
        validation=valid_path,
        test=test_path,
        format='csv',
        fields=data_fields
        )
    return train, val, test
