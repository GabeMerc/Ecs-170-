# Stage 4.1 to 4.4

## Scope

4.1 inspects both instructor datasets. 4.2 implements a recurrent binary sentiment classifier using the existing `dataset` and `method` base classes. 4.3 trains it, plots learning curves and evaluates on the test set. 4.4 trains a recurrent language model on the jokes and checks generated stories against the training data (see the sections at the end). Architecture experiments (4.5) and the report (4.6) remain separate team tasks.

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

## 4.3 Training and evaluation

```bash
python -u -m script.stage_4_script.script_rnn_classification --data-root PATH --runs rnn lstm
```

10% of the training reviews (stratified, seed 21) are held out as validation and the vocabulary is fitted on the remaining 22,500 only. Adam (lr 1e-3), batch 64 with length-bucketed batches, gradient clipping 5.0, embedding 100, hidden 128, dropout 0.3, up to 15 epochs with early stopping on validation loss (patience 3). The checkpoint with the lowest validation loss is evaluated once on the 25,000 test reviews. Learning curves (train vs validation loss/accuracy per epoch), per-epoch logs, metrics and confusion matrices are written to `result/stage_4_result/classification/`. Use CPU: on Apple MPS (PyTorch 2.3) packed LSTM sequences crash.

| Run | Params | Best epoch | Test accuracy | Precision | Recall | F1 | Train time (CPU) |
|---|---|---|---|---|---|---|---|
| Vanilla RNN | 2,029,698 | 7 / 10 | 0.7552 | 0.7657 | 0.7354 | 0.7503 | 448s |
| LSTM | 2,118,018 | 5 / 8 | 0.8239 | 0.8288 | 0.8166 | 0.8226 | 1453s |

Both models overfit after a few epochs (training accuracy keeps rising, validation loss turns up), which is why early stopping matters. The LSTM's gating handles 300-token reviews much better than the vanilla RNN, whose validation accuracy is noisy and peaks near 0.78.

## 4.4 Text generation

```bash
python -u -m script.stage_4_script.script_rnn_generation --data-root PATH --start "what did the" "my wife asked"
```

Jokes are tokenized with punctuation kept (an ellipsis is one token) and each joke ends with `<eos>`. A 2-layer LSTM language model (embedding 128, hidden 256, dropout 0.3, 2.65M parameters) is trained for 40 epochs on 90% of the jokes; the other 10% give a validation perplexity curve. Each three-word start is decoded greedily and with 3 samples (temperature 0.8, top-k 10). Every story is compared with the training jokes: exact copy, most similar training joke, longest span copied verbatim, and the share of its 2/3/4-grams that occur in training. Outputs are in `result/stage_4_result/generation/`.

Results: training perplexity falls from 426 to 2.9, while validation perplexity bottoms at 219 (epoch 10) and then rises to 554, so the model memorises the corpus. For starts seen in training, greedy decoding reproduces training jokes verbatim (e.g. "what did the green grape say to the purple grape? ..."), and samples recombine training fragments into new, mostly grammatical jokes (70-90% of 3-grams seen in training). For a start that never occurs in the jokes ("my wife asked"), only 29-40% of 3-grams are seen in training and the stories lose coherence after a few words.
