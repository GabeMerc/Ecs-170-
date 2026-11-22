# Stage 4.4 joke generation vs training data

LSTM language model, 2,654,100 parameters, 40 epochs. Final train perplexity 2.93, validation perplexity 553.82.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the green grape say to the purple grape? " breathe you idiot! breathe! " | yes | 1.00 | 19/19 | 1.00 |
| what did the | sample 1 | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 2 | what did the german physicist give for his legs? he uses a cacti! | no | 0.52 | 5/15 | 0.46 |
| what did the | sample 3 | what did the fish say when his mail caught fire? he was outstanding in his field. | no | 0.59 | 8/18 | 0.88 |
| why did the | greedy | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| why did the | sample 1 | why did the skeleton cross the road naked? to get to the other slide | no | 0.83 | 7/15 | 0.92 |
| why did the | sample 2 | why did the strawberry go out of business? because they lacked funny scents is the wurst. | no | 0.69 | 8/18 | 0.69 |
| why did the | sample 3 | why did the dog go to the party? he was pro - lighted | no | 0.57 | 6/14 | 0.92 |
| did you hear | greedy | did you hear about the kidnapping recently? the goatherd woke him up. | yes | 1.00 | 14/14 | 1.00 |
| did you hear | sample 1 | did you hear about the homemade accounts ivy? i was framed from my 9 year old. | no | 0.57 | 6/18 | 0.62 |
| did you hear | sample 2 | did you hear about the skeleton who fell into the nobel prize? he was outstanding in his field. | no | 0.72 | 11/20 | 0.89 |
| did you hear | sample 3 | did you hear about the neutron who couldn't scare? he was pro - bono | no | 0.76 | 7/15 | 0.92 |
| i went to | greedy | i went to a seafood disco...... but i don't know what it was too weak. | no | 0.55 | 6/18 | 0.75 |
| i went to | sample 1 | i went to an emotional wedding... but i didn't see that he was two - sided. | no | 0.45 | 5/18 | 0.69 |
| i went to | sample 2 | i went to a clairvoyants disco... but i don't know why. | no | 0.50 | 5/13 | 0.73 |
| i went to | sample 3 | i went to an emotional wedding but it was too saucy. | no | 0.59 | 6/12 | 0.80 |
| my wife asked | greedy | my wife asked to me her her password. i responded, i am mildly disappointed. | no | 0.38 | 5/17 | 0.40 |
| my wife asked | sample 1 | my wife asked me i jammed a duckdoo back. i guess it was a little forced. | no | 0.49 | 6/18 | 0.38 |
| my wife asked | sample 2 | my wife asked her her way to monopoly to monopoly. as the format of my essay. i think it must have a brain. | no | 0.42 | 6/26 | 0.29 |
| my wife asked | sample 3 | my wife asked me she was leaving her eyebrows when i said to me it. | no | 0.32 | 4/16 | 0.36 |
