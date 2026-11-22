# Stage 4.3 sentiment classification (test set, 25,000 reviews)

Train 22500 / validation 2500 split of the training set, vocabulary 20000, seed 21.

| Run | Params | Best epoch | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|---|---|
| rnn | 2,029,698 | 7 / 10 | 0.7552 | 0.7657 | 0.7354 | 0.7503 | 448s |
| lstm | 2,118,018 | 5 / 8 | 0.8239 | 0.8288 | 0.8166 | 0.8226 | 1453s |
