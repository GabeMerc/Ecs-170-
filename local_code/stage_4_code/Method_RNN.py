"""Stage 4.2 sentiment classifier. Outputs two unnormalized class logits."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from local_code.base_class.method import method


class Method_RNN(method, nn.Module):
    def __init__(self, mName='RNN', mDescription='Recurrent sentiment classifier',
                 vocab_size=None, embedding_dim=50, hidden_dim=128,
                 num_layers=1, dropout=0.2, cell='rnn', bidirectional=False,
                 pretrained_embeddings=None, freeze_embeddings=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if vocab_size is None or vocab_size < 2:
            raise ValueError('vocab_size must include PAD and UNK')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        cells = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        if cell.lower() not in cells:
            raise ValueError('cell must be rnn, lstm, or gru')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            weights = torch.as_tensor(pretrained_embeddings, dtype=torch.float32)
            if weights.shape != (vocab_size, embedding_dim):
                raise ValueError('Embedding matrix shape does not match vocabulary/dimension')
            with torch.no_grad():
                self.embedding.weight.copy_(weights)
                self.embedding.weight[0].zero_()
        self.embedding.weight.requires_grad_(not freeze_embeddings)
        self.recurrent = cells[cell.lower()](
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), 2)
        self.bidirectional = bidirectional

    def forward(self, tokens, lengths):
        if tokens.ndim != 2 or lengths.ndim != 1 or len(lengths) != len(tokens):
            raise ValueError('Expected tokens [batch,time] and lengths [batch]')
        if torch.any(lengths < 1) or torch.any(lengths > tokens.shape[1]):
            raise ValueError('Sequence lengths must be between 1 and padded width')
        embedded = self.embedding(tokens)
        # Padding must not affect the final recurrent state.
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, hidden = self.recurrent(packed)
        if isinstance(hidden, tuple):  # LSTM returns (h_n, c_n).
            hidden = hidden[0]
        final = (torch.cat((hidden[-2], hidden[-1]), dim=1)
                 if self.bidirectional else hidden[-1])
        return self.classifier(self.dropout(final))

    @torch.no_grad()
    def predict(self, tokens, lengths):
        was_training = self.training
        self.eval()
        try:
            return self(tokens, lengths).argmax(dim=1)
        finally:
            self.train(was_training)
