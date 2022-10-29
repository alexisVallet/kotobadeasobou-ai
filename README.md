# Kotoba de asobou solver
Simple, greedy algorithm that solves every word in [kotoba de asobou](https://taximanli.github.io/kotobade-asobou/),
without hints and without hard mode.

## Running the code
This code depends on `numpy`, `tqdm` and `jaconv`. It was tested with Python 3.10. You can install dependencies by 
running:
```bash
pip3 install -r requirements.txt
```

You can then run `play.py` to run the solver against every word in the game to show the algorithm wins every time:
```bash
$ python3 play.py 
Precomputing the first word...
100%|████████████████████████████████████████████████████████████████████████████████████| 2504/2504 [00:03<00:00, 629.18it/s]
len(won_words)=2504
len(lost_words)=0
win rate = 1.0
```
You can also run it on specific target words using the `--words` flag.

## Algorithm
For any given step. Let $W$ be the set of all possible words in the game, $R$ 
be the set of remaining words given the hints given so far, and $G$ be the set of words
tried so far.
- If there are fewer words in $R$ than remaining steps, try all the words one by one.
- Otherwise, pick the word $w$ in $W - G$ that maximizes the following score $s(w)$: for each unique character $c$
at index $i$ in $w$, such that $c$ is in words in $R$, and which has not given us a yellow or green hint
at index $i$ yet. Add to $s$ the number of words in $R$ that contain $c$.

Note that this algorithm aims to win with the highest probability in 12 steps, not find the word in the 
fewest number of steps. 

Although this algorithm does seem to win 100% of the time with the 2504 words of the game. It's 
definitely not  optimal in any other sense. For instance, I expect that planning algorithms like 
MCTS should find more clever solutions that can win faster, or win with higher probability in more
difficult variants of the game.
