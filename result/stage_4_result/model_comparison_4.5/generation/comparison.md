# Stage 4.5 joke generation: model comparison

Unlisted settings: embedding 128, hidden 256, 2 layers, dropout 0.3, Adam lr 0.002, batch 32, 40 epochs, stories generated from the final epoch. Story columns cover every generated story (greedy plus 3 samples at temperature 0.8, top-k 10, for each start).

| Run | Settings | Params | Final train ppl | Best val ppl (epoch) | Final val ppl | Generated from epoch | Exact copies | Mean nearest sim. | Mean 3-grams in train | Train time | Device |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rnn | cell=rnn | 1,962,900 | 3.40 | 258.81 (5) | 706.11 | 40 | 3 / 20 | 0.60 | 0.71 | 31s | cuda |
| lstm | cell=lstm | 2,654,100 | 3.25 | 227.60 (9) | 591.67 | 40 | 1 / 20 | 0.65 | 0.68 | 36s | cuda |
| gru | cell=gru | 2,423,700 | 2.42 | 251.49 (6) | 649.67 | 40 | 2 / 20 | 0.70 | 0.70 | 30s | cuda |
| gru_1layer | cell=gru, num_layers=1 | 2,028,948 | 1.73 | 207.20 (5) | 670.85 | 40 | 5 / 20 | 0.74 | 0.78 | 24s | cuda |
| lstm_dropout50 | cell=lstm, dropout=0.5 | 2,654,100 | 6.91 | 225.17 (10) | 381.95 | 40 | 1 / 20 | 0.65 | 0.63 | 32s | cuda |
| lstm_best_ckpt | cell=lstm, checkpoint=best | 2,654,100 | 3.25 | 227.60 (9) | 591.67 | 9 | 0 / 20 | 0.50 | 0.41 | 31s | cuda |

## Greedy story per start

| Start | Run | Story | Exact copy | 3-grams in train |
|---|---|---|---|---|
| what did the | rnn | what did the 0 say to the 8? nice belt. | yes | 1.00 |
| what did the | lstm | what did the fish say when it hit the wall? dam | no | 1.00 |
| what did the | gru | what did the fish say when it hit the wall? dam | no | 1.00 |
| what did the | gru_1layer | what did the tin man say when he got run over by a steamroller? curses! foil again! | yes | 1.00 |
| what did the | lstm_dropout50 | what did the fish say to the other? i ` ll meet you! | no | 0.77 |
| what did the | lstm_best_ckpt | what did the difference say to the other? he was a little dressing. | no | 0.54 |
| why did the | rnn | why did the chicken cross the road? to get to the other slide | no | 1.00 |
| why did the | lstm | why did the chicken cross the road? to get to the other slide | no | 1.00 |
| why did the | gru | why did the chicken cross the road? to get to the other slide | no | 1.00 |
| why did the | gru_1layer | why did the chicken cross the road? to get to the other side. | yes | 1.00 |
| why did the | lstm_dropout50 | why did the chicken cross the road? to get to the other slide | no | 1.00 |
| why did the | lstm_best_ckpt | why did the chicken cross the road? because he was a little dressing. | no | 0.85 |
| did you hear | rnn | did you hear about the kidnapping recently? he was impeached! | no | 0.90 |
| did you hear | lstm | did you hear about the kidnapping that fell? he had a dickens of problems. | no | 0.64 |
| did you hear | gru | did you hear about the stallion that died? he had a dickens. | no | 0.75 |
| did you hear | gru_1layer | did you hear about the kidnapping in delaware? the goatherd woke him up. | no | 0.92 |
| did you hear | lstm_dropout50 | did you hear about the kidnapping who was accidentally? he was outstanding. | no | 0.58 |
| did you hear | lstm_best_ckpt | did you hear about the chicken? he was a little dressing. | no | 0.64 |
| i went to | rnn | i went to a shredded and drank, " i don't know what he laced it with a little joke. | no | 0.53 |
| i went to | lstm | i went to a seafood disco last night......... but i don't know it. | no | 0.67 |
| i went to | gru | i went to a seafood disco last week. i don't know what it was a mirage. | no | 0.69 |
| i went to | gru_1layer | i went to the dermatologist about something on my neck - - - - - - - - - - - - - - - - - line. | no | 0.41 |
| i went to | lstm_dropout50 | i went to the hokey pokey but i turned myself around. | no | 1.00 |
| i went to | lstm_best_ckpt | i went to the bar... i was a joke. | no | 0.44 |
| my wife asked | rnn | my wife asked she's doing so full of themselves. | no | 0.38 |
| my wife asked | lstm | my wife asked to put a joke about science but i was watching a plain planed plane it out of the shoulders. | no | 0.48 |
| my wife asked | gru | my wife asked to keep me to help me a dirty joke. but i haven't tell it was warranted. | no | 0.42 |
| my wife asked | gru_1layer | my wife asked me i don't give her anything. i think she was such a little bit of mildew - - - so i was told this joke but i didn't know where i found it. | no | 0.44 |
| my wife asked | lstm_dropout50 | my wife asked to chase a skin country... it was a tad polish. | no | 0.38 |
| my wife asked | lstm_best_ckpt | my wife asked to the other joke... i was a joke. | no | 0.18 |
