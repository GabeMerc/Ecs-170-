# Stage 4.4 joke generation vs training data

GRU language model, 2,423,700 parameters, 40 epochs. Final train perplexity 2.42, validation perplexity 649.67.

| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |
|---|---|---|---|---|---|---|
| what did the | greedy | what did the fish say when it hit the wall? dam | no | 0.96 | 12/12 | 1.00 |
| what did the | sample 1 | what did the number zero say to the number eight? you're being uncanny! | no | 0.79 | 11/15 | 0.92 |
| what did the | sample 2 | what did the 0 say when he lost his tractor? curses! foil again! | no | 0.67 | 7/16 | 0.86 |
| what did the | sample 3 | what did the pilot say to the 8? nice belt. | no | 0.92 | 8/12 | 0.90 |
| why did the | greedy | why did the chicken cross the road? to get to the other slide | no | 0.93 | 13/14 | 1.00 |
| why did the | sample 1 | why did the chicken cross the road? to get to the other side. | yes | 1.00 | 15/15 | 1.00 |
| why did the | sample 2 | why did the strawberry go when he couldn't cross the road? because it couldn't get a date! | no | 0.72 | 5/19 | 0.59 |
| why did the | sample 3 | why did the chicken cross the road? to get to the other side. | yes | 1.00 | 15/15 | 1.00 |
| did you hear | greedy | did you hear about the stallion that died? he had a dickens. | no | 0.71 | 6/14 | 0.75 |
| did you hear | sample 1 | did you hear about the stallion and poison accidentally? he lost the huile d'olive | no | 0.71 | 7/15 | 0.69 |
| did you hear | sample 2 | did you hear about the stallion and the mare? they couldn't stop it. | no | 0.77 | 11/15 | 0.77 |
| did you hear | sample 3 | did you hear about the two monocles that got married? he was pro - bono | no | 0.73 | 7/16 | 0.79 |
| i went to | greedy | i went to a seafood disco last week. i don't know what it was a mirage. | no | 0.55 | 8/18 | 0.69 |
| i went to | sample 1 | i went to a seafood disco disco week. i said it was a mirage. | no | 0.52 | 6/16 | 0.43 |
| i went to | sample 2 | i went to a seafood disco last week. and a real said. | no | 0.62 | 8/14 | 0.58 |
| i went to | sample 3 | i went to a seafood disco last week. and pulled a mussel. | no | 0.90 | 8/14 | 0.75 |
| my wife asked | greedy | my wife asked to keep me to help me a dirty joke. but i haven't tell it was warranted. | no | 0.44 | 4/21 | 0.42 |
| my wife asked | sample 1 | my wife asked to keep his dinner to my son and i want to change his car to get his sex change operation. | no | 0.40 | 7/24 | 0.32 |
| my wife asked | sample 2 | my wife asked out the bathroom, so he didn't give a little bit... and it was warranted. | no | 0.31 | 4/20 | 0.28 |
| my wife asked | sample 3 | my wife asked his laptop... he was 60 in nearly " every nation. i guess he had to get a sex nation. | no | 0.30 | 3/25 | 0.26 |
