# Stage 4.4 joke generation vs training data

LSTM language model, 2,654,100 parameters, 40 epochs. Final train perplexity 6.91, validation perplexity 381.95.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the fish say to the other? i ` ll meet you! | no | 0.71 | 6/15 | 0.77 |
| what did the | sample 1 | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 2 | what did the german say to the pear? nice belt. | no | 0.83 | 5/12 | 0.70 |
| what did the | sample 3 | what did the buffalo say to the 8? nice belt. | no | 0.92 | 8/12 | 1.00 |
| why did the | greedy | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| why did the | sample 1 | why did the chicken cross the road? it had a chicken complex. | no | 0.65 | 8/14 | 0.75 |
| why did the | sample 2 | why did the buddhist lay novocaine to get? because it had a million. | no | 0.56 | 4/15 | 0.38 |
| why did the | sample 3 | why did the chicken cross the road? to get to the nearest bride. | no | 0.87 | 12/15 | 0.85 |
| did you hear | greedy | did you hear about the kidnapping who was accidentally? he was outstanding. | no | 0.73 | 6/14 | 0.58 |
| did you hear | sample 1 | did you hear about the kidnapping who hated suicide? he was a little horse! | no | 0.56 | 6/16 | 0.64 |
| did you hear | sample 2 | did you hear a one about a crappy restaurant? he was held. | no | 0.60 | 5/14 | 0.58 |
| did you hear | sample 3 | did you hear about the schizophrenic accounts ivy to sleep? he was a million. | no | 0.57 | 7/16 | 0.57 |
| i went to | greedy | i went to the hokey pokey but i turned myself around. | no | 0.88 | 10/12 | 1.00 |
| i went to | sample 1 | i went to the dermatologist pokey but it said to see it was fake. | no | 0.44 | 5/15 | 0.38 |
| i went to | sample 2 | i went to a seafood disco production week. | no | 0.67 | 6/9 | 0.57 |
| i went to | sample 3 | i went to the dermatologist pokey but i guess it would be a joke. | no | 0.50 | 5/15 | 0.62 |
| my wife asked | greedy | my wife asked to chase a skin country... it was a tad polish. | no | 0.42 | 5/15 | 0.38 |
| my wife asked | sample 1 | my wife asked to help to be ambidextrous to the hokey pokey currently. | no | 0.37 | 4/14 | 0.25 |
| my wife asked | sample 2 | my wife asked to help on a pbs with a neck. | no | 0.37 | 3/12 | 0.10 |
| my wife asked | sample 3 | my wife asked to cheer a skin disorder... it was riveting. | no | 0.44 | 5/13 | 0.45 |
