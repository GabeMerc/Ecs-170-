# Stage 4.4 joke generation vs training data

LSTM language model, 2,654,100 parameters, 40 epochs. Final train perplexity 3.25, validation perplexity 591.67.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the fish say when it hit the wall? dam | no | 0.96 | 12/12 | 1.00 |
| what did the | sample 1 | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 2 | what did the german physicist force eat for breakfast during ww2? ohm. | no | 0.81 | 7/14 | 0.75 |
| what did the | sample 3 | what did the buddhist say to the other planets? " is a mile is a big giveaway! | no | 0.55 | 7/19 | 0.65 |
| why did the | greedy | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| why did the | sample 1 | why did the strawberry cross the road? to get to the other slide | no | 0.86 | 9/14 | 0.83 |
| why did the | sample 2 | why did the chicken cross the playground? it was chassing to get loki. | no | 0.73 | 8/15 | 0.77 |
| why did the | sample 3 | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| did you hear | greedy | did you hear about the kidnapping that fell? he had a dickens of problems. | no | 0.65 | 6/16 | 0.64 |
| did you hear | sample 1 | did you hear about the kidnapping that got married by the tree? i was pro from my life because i couldn't concentrate! | no | 0.49 | 6/24 | 0.41 |
| did you hear | sample 2 | did you hear about the scarecrow who didn't go to prom? he worked it out. | no | 0.70 | 7/17 | 0.87 |
| did you hear | sample 3 | did you hear about the guy who was arrested? he was held in a bouqet. | no | 0.79 | 7/17 | 0.73 |
| i went to | greedy | i went to a seafood disco last night......... but i don't know it. | no | 0.62 | 7/17 | 0.67 |
| i went to | sample 1 | i went to a seafood disco last night... it's called a mussel. | no | 0.76 | 7/14 | 0.67 |
| i went to | sample 2 | i went to a shredded about my door. all it takes, it was cancelled in his head. | no | 0.41 | 5/20 | 0.44 |
| i went to | sample 3 | i went to the dermatologist about the desert. i can eat an obstacle illusion. | no | 0.35 | 6/16 | 0.43 |
| my wife asked | greedy | my wife asked to put a joke about science but i was watching a plain planed plane it out of the shoulders. | no | 0.39 | 6/23 | 0.48 |
| my wife asked | sample 1 | my wife asked to put an facebook... but i couldn't find a glass of god. | no | 0.39 | 4/17 | 0.40 |
| my wife asked | sample 2 | my wife asked to give a joke about science but i couldn't concentrate. | no | 0.44 | 6/14 | 0.58 |
| my wife asked | sample 3 | my wife asked to me a soap and a balloon... i don't want to be obsessed with this day. | no | 0.30 | 4/21 | 0.26 |
