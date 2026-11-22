# Stage 4.4 joke generation vs training data

GRU language model, 2,028,948 parameters, 40 epochs. Final train perplexity 1.73, validation perplexity 670.85.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the tin man say when he got run over by a steamroller? curses! foil again! | yes | 1.00 | 20/20 | 1.00 |
| what did the | sample 1 | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 2 | what did the buffalo say to the circle? nice belt. | no | 0.83 | 6/12 | 0.90 |
| what did the | sample 3 | what did the buffalo say to the number eight? nice belt. | no | 0.85 | 8/13 | 1.00 |
| why did the | greedy | why did the chicken cross the road? to get to the other side. | yes | 1.00 | 15/15 | 1.00 |
| why did the | sample 1 | why did the chicken cross the road? to get to the other side. | yes | 1.00 | 15/15 | 1.00 |
| why did the | sample 2 | why did the chicken cross the road? to get to the other side. | yes | 1.00 | 15/15 | 1.00 |
| why did the | sample 3 | why did the snail draw an art pot of water? they were part of a stewicide pact. | no | 0.77 | 12/19 | 0.82 |
| did you hear | greedy | did you hear about the kidnapping in delaware? the goatherd woke him up. | no | 0.90 | 9/15 | 0.92 |
| did you hear | sample 1 | did you hear about the kidnapping in delaware? the goatherd woke him up. | no | 0.90 | 9/15 | 0.92 |
| did you hear | sample 2 | did you hear about the kidnapping in delaware? the goatherd woke him up. | no | 0.90 | 9/15 | 0.92 |
| did you hear | sample 3 | did you hear about the stallion and the mare? they ended into a stable relationship. | no | 0.91 | 11/17 | 0.73 |
| i went to | greedy | i went to the dermatologist about something on my neck - - - - - - - - - - - - - - - - - line. | no | 0.45 | 12/29 | 0.41 |
| i went to | sample 1 | i went to the dermatologist about something my neck - hand. | no | 0.56 | 7/12 | 0.70 |
| i went to | sample 2 | i went to a shredded cheese convention the other day... it said to me. | no | 0.80 | 12/16 | 0.93 |
| i went to | sample 3 | i went to a seafood disco last week...... its shift - - knick knack. | no | 0.69 | 10/17 | 0.80 |
| my wife asked | greedy | my wife asked me i don't give her anything. i think she was such a little bit of mildew - - - so i was told this joke but i didn't know where i found it. | no | 0.27 | 5/38 | 0.44 |
| my wife asked | sample 1 | my wife asked me i give the bike a dude who's abroad. i think, it was warranted. | no | 0.38 | 5/20 | 0.44 |
| my wife asked | sample 2 | my wife asked i do like to eat breakfast. who's there? no, not your parents, why do lawyers have such on boring. elephino | no | 0.32 | 5/28 | 0.31 |
| my wife asked | sample 3 | my wife asked me i was watching a transformer...... i love i love chocolate. " | no | 0.36 | 4/18 | 0.31 |
