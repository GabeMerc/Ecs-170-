"""Stage 4.4 word-level recurrent language model for joke generation."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from local_code.base_class.method import method


class Method_RNN_Generator(method, nn.Module):
    def __init__(self, mName='RNN-Generator', mDescription='Recurrent joke language model',
                 vocab_size=None, embedding_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3, cell='lstm'):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        if vocab_size is None or vocab_size < 3:
            raise ValueError('vocab_size must include PAD, UNK and <eos>')
        if not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        cells = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        if cell.lower() not in cells:
            raise ValueError('cell must be rnn, lstm, or gru')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.recurrent = cells[cell.lower()](
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tokens, lengths=None, state=None):
        """Returns next-token logits [batch,time,vocab] and the final recurrent state."""
        embedded = self.dropout(self.embedding(tokens))
        if lengths is None:
            outputs, state = self.recurrent(embedded, state)
        else:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            outputs, state = self.recurrent(packed, state)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True,
                                             total_length=tokens.shape[1])
        return self.output(self.dropout(outputs)), state

    @torch.no_grad()
    def generate(self, prefix_ids, eos_id, max_new_tokens=60, temperature=1.0,
                 top_k=None, generator=None):
        """Extends prefix_ids until <eos>; temperature=0 means greedy decoding."""
        if not prefix_ids:
            raise ValueError('prefix_ids must contain at least one token')
        was_training = self.training
        self.eval()
        try:
            device = self.output.weight.device
            logits, state = self(torch.tensor([prefix_ids], device=device))
            next_logits = logits[0, -1]
            generated = []
            for _ in range(max_new_tokens):
                # Never emit PAD/UNK: they are not words.
                next_logits[:2] = float('-inf')
                if temperature == 0:
                    token = int(next_logits.argmax())
                else:
                    scaled = next_logits / temperature
                    if top_k is not None:
                        cutoff = torch.topk(scaled, top_k).values[-1]
                        scaled = scaled.masked_fill(scaled < cutoff, float('-inf'))
                    probs = torch.softmax(scaled, dim=0).cpu()
                    token = int(torch.multinomial(probs, 1, generator=generator))
                if token == eos_id:
                    break
                generated.append(token)
                logits, state = self(torch.tensor([[token]], device=device), state=state)
                next_logits = logits[0, -1]
            return generated
        finally:
            self.train(was_training)
