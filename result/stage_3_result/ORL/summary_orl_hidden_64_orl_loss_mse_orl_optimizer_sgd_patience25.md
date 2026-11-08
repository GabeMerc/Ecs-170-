# ORL CNN results (mean +- std over seeds [21, 42, 7])

| Run | Change vs baseline | Params | Epochs | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|---|---|---|---|---|---|---|---|
| orl_hidden_64 | hidden_sizes=(64,) | 1,357,352 | 150 | 0.7000 +- 0.1080 | 0.6010 +- 0.1164 | 0.7000 +- 0.1080 | 0.6311 +- 0.1175 |
| orl_loss_mse | loss_name=mse | 5,149,928 | 136 | 0.8583 +- 0.0773 | 0.8153 +- 0.0934 | 0.8583 +- 0.0773 | 0.8292 +- 0.0886 |
| orl_optimizer_sgd | optimizer_name=sgd | 5,149,928 | 32 | 0.2750 +- 0.0540 | 0.1925 +- 0.0612 | 0.2750 +- 0.0540 | 0.2080 +- 0.0580 |
