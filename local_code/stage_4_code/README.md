# Stage 4.1 and 4.2

## Scope

4.1 inspects both instructor datasets. 4.2 implements a recurrent binary sentiment classifier using the existing `dataset` and `method` base classes. Full training/evaluation/learning curves (4.3), text generation (4.4), architecture experiments (4.5), and the report (4.6) remain separate team tasks. No trained accuracy is claimed here.

## Dataset inspection

Verified against the supplied Desktop ECS170/stage_4_data directory:

| Dataset | Records | Format |
| --- | ---: | --- |
| Sentiment training | 25,000 | 12,500 negative and 12,500 positive UTF-8 files |
| Sentiment testing | 25,000 | 12,500 negative and 12,500 positive UTF-8 files |
| Generation | 1,622 | CSV in extensionless text_generation/data; ID,Joke columns |

Labels are negative=0, positive=1; ratings in filenames are not prediction targets. The joke CSV requires a CSV parser because jokes can contain quoted commas. Example opening: “What did the bartender say”. Raw datasets and GloVe files are external inputs and are not committed.

## Setup and inspection

Requires Python 3.10+ and PyTorch (validated with PyTorch 2.9.1). Run commands from the repository root:

```bash
python -m script.stage_4_script.script_inspect_text --data-root "$HOME/Desktop/ECS170/stage_4_data"
python -m unittest discover -s tests -p 'test_stage4.py' -v
```

## Classification integration for 4.3

```python
from torch.utils.data import DataLoader
from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, collate_reviews
from local_code.stage_4_code.Method_RNN import Method_RNN

loader = Dataset_Loader(data_root='/path/to/stage_4_data')
data = loader.load()
train_loader = DataLoader(data['train'], batch_size=64, shuffle=True,
                          collate_fn=collate_reviews)
model = Method_RNN(vocab_size=len(data['vocabulary']))
for tokens, lengths, labels in train_loader:
    logits = model(tokens, lengths)
    # Pass logits and labels to torch.nn.CrossEntropyLoss, then optimize.
    break
```

Defaults: 20,000 vocabulary entries including PAD=0 and UNK=1, minimum frequency 2, maximum 300 tokens per review, 50-dimensional learned embeddings, one vanilla RNN layer with 128 hidden units, dropout 0.2, and two output logits. Text cleaning removes HTML, decodes entities, lowercases, and tokenizes while retaining negation/contractions. The vocabulary is fitted only on training reviews; empty reviews map to UNK. Batches use dynamic right padding and packed sequences so padding does not alter the final state. No softmax is applied before cross entropy.

The forward interface is `model(tokens, lengths)` with tokens `[batch,time]` and lengths `[batch]`. Move tokens/model/labels to the same device; lengths are moved to CPU internally for packing. `predict` returns class IDs and restores the prior training mode. Save vocabulary tokens alongside model weights so later inference uses the same IDs.

For 4.5, `cell='lstm'` or `cell='gru'`, `num_layers`, and `bidirectional` are supported but no comparative results have been generated. Optional `pretrained_embeddings` accepts an aligned float matrix `[vocab_size, embedding_dim]`, with `freeze_embeddings` controlling updates; the PAD row is zeroed. GloVe is not required for 4.1/4.2 and pretrained vectors were not present in the supplied folder.

For later training, take validation data from the training split and fit vocabulary on that training subset when tuning. Keep the provided test split held out until final evaluation. Use gradient clipping for vanilla RNN training, record seeds/configuration, and report actual measured metrics.
