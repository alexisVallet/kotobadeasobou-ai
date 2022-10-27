import multiprocessing as mp
from functools import partial

import tqdm
import numpy as np

from words import WORDS


_ALL_CHARS = np.array(WORDS).view('U1').reshape((len(WORDS), -1))


def compute_best_word(remaining_words, blacklist):
    remaining_chars = np.array(remaining_words).view('U1').reshape((len(remaining_words), -1))
    a, b = np.unique(remaining_chars, return_counts=True)
    char_to_count = {ch: co for (ch, co) in zip(a, b)}
    word_score = {
        _ALL_CHARS[i].view('U4')[0]: sum([char_to_count[c] for c in np.unique(_ALL_CHARS[i]) if c in char_to_count]) for i
        in range(_ALL_CHARS.shape[0])
    }
    best_word, score = max([(w, s) for w, s in word_score.items() if w not in blacklist], key=lambda t: t[1])

    return best_word, score


def play_word(first_word, word):
    remaining_words = WORDS
    best_word, score = first_word
    j = 0
    guesses = [best_word]

    while best_word != word and j < 11:
        grays = []
        yellows = []
        greens = []
        for i, c in enumerate(best_word):
            if word[i] == c:
                greens.append((i, c))
            elif c in word:
                yellows.append(c)
            else:
                grays.append(c)
        remaining_words = [
            w for w in remaining_words if
            all([w[i] == c for i, c in greens])
            and all([c in w for c in yellows])
            and not any([c in w for c in grays])
        ]
        best_word, score = compute_best_word(remaining_words, blacklist=guesses)
        guesses.append(best_word)
        j += 1
    return word, (word == best_word), guesses


def main():
    won_words = []
    lost_words = []

    # No need to compute the same first word every time.
    first_word = compute_best_word(WORDS, blacklist=[])

    pool = mp.Pool()

    out_it = pool.imap(partial(play_word, first_word), WORDS)

    for word, won, guesses in tqdm.tqdm(out_it, total=len(WORDS)):
        if won:
            won_words.append(word)
        else:
            lost_words.append((word, guesses))
    print(f"{len(won_words)=}")
    print(f"{len(lost_words)=}")
    print(f"win rate = {len(won_words) / len(WORDS)}")
    print(lost_words)


if __name__ == '__main__':
    main()
