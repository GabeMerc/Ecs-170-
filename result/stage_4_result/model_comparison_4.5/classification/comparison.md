# Stage 4.5 sentiment classification: model comparison (test set, 25,000 reviews)

Train 22500 / validation 2500 split of the training set, vocabulary 20000. Unlisted settings: embedding 100, hidden 128, 1 layer, dropout 0.3, Adam lr 0.001, batch 64, early stopping patience 3. Test metrics use the epoch with the lowest validation loss.

| Run | Settings | Params | Best epoch | Val accuracy | Test accuracy | Precision | Recall | F1 | Train time | Device |
|---|---|---|---|---|---|---|---|---|---|---|
| rnn | cell=rnn | 2,029,698 | 2 / 5 | 0.7424 | 0.7236 | 0.7349 | 0.6995 | 0.7168 | 32s | cuda |
| lstm | cell=lstm | 2,118,018 | 6 / 9 | 0.8620 | 0.8418 | 0.8613 | 0.8148 | 0.8374 | 55s | cuda |
| gru | cell=gru | 2,088,578 | 3 / 6 | 0.8680 | 0.8636 | 0.8682 | 0.8572 | 0.8627 | 43s | cuda |
| bilstm | cell=lstm, bidirectional=True | 2,236,034 | 4 / 7 | 0.8548 | 0.8386 | 0.8662 | 0.8009 | 0.8323 | 64s | cuda |
| lstm_2layer | cell=lstm, num_layers=2 | 2,250,114 | 4 / 7 | 0.8516 | 0.8420 | 0.8531 | 0.8262 | 0.8394 | 64s | cuda |
| gru_hidden256 | cell=gru, hidden_dim=256 | 2,275,458 | 3 / 6 | 0.8812 | 0.8666 | 0.8532 | 0.8855 | 0.8691 | 40s | cuda |
