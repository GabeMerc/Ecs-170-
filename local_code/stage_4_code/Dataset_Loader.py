"""Stage 4.1: inspect both corpora and prepare sentiment sequences."""
import csv
import html
import re
from collections import Counter
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from local_code.base_class.dataset import dataset


def tokenize(text):
    # Keep negations and contractions: removing them changes sentiment.
    text = re.sub(r'<[^>]+>', ' ', html.unescape(text)).lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)


class Vocabulary:
    PAD, UNK = 0, 1

    def __init__(self, documents, max_vocab=20000, min_freq=2):
        if max_vocab < 2 or min_freq < 1:
            raise ValueError('max_vocab >= 2 and min_freq >= 1 required')
        counts = Counter(token for doc in documents for token in tokenize(doc))
        words = sorted(counts, key=lambda word: (-counts[word], word))
        self.tokens = ['<pad>', '<unk>'] + [
            word for word in words if counts[word] >= min_freq
        ][:max_vocab - 2]
        self.word_to_id = {word: i for i, word in enumerate(self.tokens)}

    def __len__(self):
        return len(self.tokens)

    def encode(self, text, max_length=300):
        if max_length < 1:
            raise ValueError('max_length must be positive')
        return [self.word_to_id.get(w, self.UNK)
                for w in tokenize(text)[:max_length]] or [self.UNK]


class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, vocabulary, max_length):
        self.sequences = [torch.tensor(vocabulary.encode(t, max_length),
                                      dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


def collate_reviews(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    return (pad_sequence(sequences, batch_first=True, padding_value=0),
            lengths, torch.stack(labels))


class Dataset_Loader(dataset):
    def __init__(self, dName='IMDb', dDescription='Stage 4 text datasets',
                 data_root=None, max_vocab=20000, min_freq=2, max_length=300):
        super().__init__(dName, dDescription)
        self.dataset_source_folder_path = data_root
        self.max_vocab, self.min_freq = max_vocab, min_freq
        self.max_length = max_length

    def _root(self):
        if self.dataset_source_folder_path is None:
            raise ValueError('Set data_root or dataset_source_folder_path to stage_4_data')
        return Path(self.dataset_source_folder_path).expanduser()

    def _reviews(self, split):
        texts, labels = [], []
        for category, label in [('neg', 0), ('pos', 1)]:
            folder = self._root() / 'text_classification' / split / category
            paths = sorted(folder.glob('*.txt'))
            if not paths:
                raise FileNotFoundError(f'No reviews found in {folder}')
            for path in paths:
                texts.append(path.read_text(encoding='utf-8'))
                labels.append(label)
        return texts, labels

    def load_generation(self):
        path = self._root() / 'text_generation' / 'data'
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['ID', 'Joke']:
                raise ValueError(f'Expected ID,Joke CSV columns in {path}')
            return [row['Joke'] for row in reader]

    def inspect(self):
        result = {}
        for split in ['train', 'test']:
            texts, labels = self._reviews(split)
            result[split] = {'count': len(texts), 'negative': labels.count(0),
                             'positive': labels.count(1), 'example': texts[0][:300]}
        jokes = self.load_generation()
        result['generation'] = {'count': len(jokes), 'example': jokes[0]}
        return result

    def load(self):
        train_texts, train_labels = self._reviews('train')
        test_texts, test_labels = self._reviews('test')
        # Never learn vocabulary/frequencies from the held-out test reviews.
        vocabulary = Vocabulary(train_texts, self.max_vocab, self.min_freq)
        self.data = {
            'train': ReviewDataset(train_texts, train_labels, vocabulary, self.max_length),
            'test': ReviewDataset(test_texts, test_labels, vocabulary, self.max_length),
            'vocabulary': vocabulary,
        }
        return self.data
