# Stage 4.4 joke generation vs training data

RNN language model, 1,962,900 parameters, 40 epochs. Final train perplexity 3.40, validation perplexity 706.11.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 1 | what did the 0 say to the 8? nice belt. | yes | 1.00 | 12/12 | 1.00 |
| what did the | sample 2 | what did the german policeman say to the barber? it was chassing. | no | 0.62 | 7/14 | 0.83 |
| what did the | sample 3 | what did the buddhist say to the other plate? dinners it. | no | 0.72 | 7/13 | 0.82 |
| why did the | greedy | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| why did the | sample 1 | why did the chicken cross the road? to be to be topical | no | 0.71 | 9/13 | 0.82 |
| why did the | sample 2 | why did the chicken cross the playground? to get to the other slide | yes | 1.00 | 14/14 | 1.00 |
| why did the | sample 3 | why did the buddhist refuse novocaine when it was covered out of the alphabet? the post office. | no | 0.45 | 7/19 | 0.82 |
| did you hear | greedy | did you hear about the kidnapping recently? he was impeached! | no | 0.62 | 8/12 | 0.90 |
| did you hear | sample 1 | did you hear about the kidnapping recently? it kept crashing. | no | 0.69 | 8/12 | 0.80 |
| did you hear | sample 2 | did you hear about the two monocles are you of a tree? a pool table. | no | 0.54 | 8/17 | 0.73 |
| did you hear | sample 3 | did you hear about the native american who was a windshield sausage. | no | 0.55 | 8/13 | 0.64 |
| i went to | greedy | i went to a shredded and drank, " i don't know what he laced it with a little joke. | no | 0.40 | 6/21 | 0.53 |
| i went to | sample 1 | i went to a seafood disco last week and it goes to waste the perseid day, but i was framed to make a van succinct buildup. | no | 0.51 | 8/28 | 0.46 |
| i went to | sample 2 | i went to a shredded and drank 37 cups of a tree? ' cause you smell carrots? | no | 0.39 | 5/19 | 0.65 |
| i went to | sample 3 | i went to a shredded... well are the right day. the photon says, " well, chew, chew! * | no | 0.39 | 6/24 | 0.45 |
| my wife asked | greedy | my wife asked she's doing so full of themselves. | no | 0.35 | 4/10 | 0.38 |
| my wife asked | sample 1 | my wife asked she's doing right joke... but i blew it friday. | no | 0.35 | 4/14 | 0.42 |
| my wife asked | sample 2 | my wife asked to have a heart attack towards the local swimming pool. " | no | 0.43 | 6/15 | 0.54 |
| my wife asked | sample 3 | my wife asked to have a dirty joke? it was a soft. | no | 0.43 | 4/14 | 0.50 |
