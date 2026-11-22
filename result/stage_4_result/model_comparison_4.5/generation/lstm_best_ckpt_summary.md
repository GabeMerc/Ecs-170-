# Stage 4.4 joke generation vs training data

LSTM language model, 2,654,100 parameters, 40 epochs. Final train perplexity 3.25, validation perplexity 591.67. Stories generated from the best validation checkpoint (epoch 9).

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the difference say to the other? he was a little dressing. | no | 0.59 | 5/15 | 0.54 |
| what did the | sample 1 | what did the difference say to the other? because it's he wanted to the a little joke. | no | 0.52 | 5/19 | 0.35 |
| what did the | sample 2 | what did the difference say to the other? he was his -! | no | 0.64 | 5/14 | 0.42 |
| what did the | sample 3 | what did the chicken say to the other? it was a other. | no | 0.62 | 5/14 | 0.58 |
| why did the | greedy | why did the chicken cross the road? because he was a little dressing. | no | 0.65 | 8/15 | 0.85 |
| why did the | sample 1 | why did the man between the road? he was a dog of one in the head. | no | 0.53 | 4/18 | 0.38 |
| why did the | sample 2 | why did the skeleton say to get the other? because the little of a first of a head. | no | 0.47 | 4/20 | 0.22 |
| why did the | sample 3 | why did the skeleton cross the other? a little angle | no | 0.48 | 4/11 | 0.56 |
| did you hear | greedy | did you hear about the chicken? he was a little dressing. | no | 0.62 | 5/13 | 0.64 |
| did you hear | sample 1 | did you hear about a best favorite? | no | 0.53 | 4/8 | 0.33 |
| did you hear | sample 2 | did you hear about the best about the chicken? " | no | 0.54 | 5/11 | 0.44 |
| did you hear | sample 3 | did you hear about the most who was a lightbulb? because he was a other. | no | 0.67 | 5/17 | 0.53 |
| i went to | greedy | i went to the bar... i was a joke. | no | 0.48 | 4/11 | 0.44 |
| i went to | sample 1 | i went to the bar... i think " but i was have a head. | no | 0.41 | 4/16 | 0.21 |
| i went to | sample 2 | i went to the bar... he was the little joke... i was a joke... | no | 0.39 | 4/17 | 0.40 |
| i went to | sample 3 | i went to make a little...... but you don't know. | no | 0.52 | 3/13 | 0.55 |
| my wife asked | greedy | my wife asked to the other joke... i was a joke. | no | 0.37 | 3/13 | 0.18 |
| my wife asked | sample 1 | my wife asked and a bar of the other. the other day to the bar. | no | 0.33 | 3/17 | 0.33 |
| my wife asked | sample 2 | my wife asked to the joke......, he was me. | no | 0.36 | 2/13 | 0.00 |
| my wife asked | sample 3 | my wife asked of the joke of a bar. i am up to be a bar... but but " i had a bartender. | no | 0.30 | 3/26 | 0.25 |
