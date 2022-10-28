import argparse
import math

import copy
import multiprocessing as mp
import random
from functools import partial

import tqdm
import numpy as np
from scipy.special import softmax

from words import WORDS


def is_hiragana_word(w):
    return all(chr(0x3041) <= c <= chr(0x3096) for c in w)


_WORDS_HIRAGANA = [
    w for w in WORDS if is_hiragana_word(w)
]
_WORDS_KATAKANA = [
    w for w in WORDS if not is_hiragana_word(w)
]


def compute_best_word(all_words, remaining_words, blacklist, greens_and_yellows):
    remaining_chars = np.array(remaining_words).view('U1').reshape((len(remaining_words), -1))
    a, b = np.unique(remaining_chars, return_counts=True)
    char_to_count = {ch: co for (ch, co) in zip(a, b)}
    word_scores = {}
    for word_chars in all_words:
        word_score = 0
        scored_chars = set()

        for j, c in enumerate(word_chars):
            if c not in scored_chars and c in char_to_count and (j, c) not in greens_and_yellows:
                word_score += char_to_count[c]
                scored_chars.add(c)
        word_scores[word_chars] = word_score
    best_word, _ = max(((w, c) for w, c in word_scores.items() if w not in blacklist), key=lambda t: t[1])

    return best_word


def step(remaining_words, j, best_word, target_word=None):
    if target_word is None:
        # Simulate by picking a target word at random.
        target_word = random.choice(remaining_words)
    if best_word != target_word and j < 11:
        grays = []
        yellows = []
        greens = []
        for i, c in enumerate(best_word):
            if target_word[i] == c:
                greens.append((i, c))
            elif c in target_word:
                yellows.append((i, c))
            else:
                grays.append(c)
        new_remaining_words = [
            w for w in remaining_words if
            all([w[i] == c for i, c in greens])
            and all([c in w for _, c in yellows])
            and not any([c in w for c in grays])
        ]
        return "running", (new_remaining_words, greens, yellows)
    else:
        return "terminated", (best_word == target_word)


def play_word(first_word_hiragana, first_word_katakana, word):
    if is_hiragana_word(word):
        all_words = _WORDS_HIRAGANA
        best_word = first_word_hiragana
    else:
        all_words = _WORDS_KATAKANA
        best_word = first_word_katakana
    remaining_words = all_words
    j = 0
    guesses = [best_word]
    all_greens_and_yellows = set()

    while True:
        status, outs = step(remaining_words, j, best_word, target_word=word)
        if status == "terminated":
            break
        remaining_words, greens, yellows = outs
        all_greens_and_yellows |= set(greens) | set(yellows)

        num_steps_remaining = 11 - j
        if num_steps_remaining >= len(remaining_words):
            # Special case when we have more steps than remaining words.
            # Then we can just brute force.
            best_word = remaining_words.pop(0)
        else:
            # Otherwise, we reduce the search space until we can.
            best_word = compute_best_word(all_words, remaining_words, blacklist=guesses,
                                          greens_and_yellows=all_greens_and_yellows)
        guesses.append(best_word)
        j += 1
    return word, (word == best_word), guesses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test_words", type=int, default=len(WORDS))
    parser.add_argument("--words", type=str, default=None, nargs='*')
    args = parser.parse_args()
    if args.words is not None:
        words_to_test = args.words
    else:
        words_to_test = copy.deepcopy(WORDS)
        random.shuffle(words_to_test)
        words_to_test = words_to_test[:args.n_test_words]
    won_words = []
    lost_words = []

    # No need to compute the same first word every time. We compute it with 10x the resources because
    # why not.
    print(f"Precomputing the first word...")
    first_word_hiragana = compute_best_word(_WORDS_HIRAGANA, _WORDS_HIRAGANA, blacklist=[], greens_and_yellows=frozenset())
    first_word_katakana = compute_best_word(_WORDS_KATAKANA, _WORDS_KATAKANA, blacklist=[], greens_and_yellows=frozenset())

    pool = mp.Pool()

    out_it = pool.imap(partial(play_word, first_word_hiragana, first_word_katakana), words_to_test)

    for word, won, guesses in tqdm.tqdm(out_it, total=len(words_to_test)):
        if won:
            won_words.append(word)
        else:
            lost_words.append((word, guesses))
    print(f"{len(won_words)=}")
    print(f"{len(lost_words)=}")
    print(f"win rate = {len(won_words) / len(words_to_test)}")
    print(lost_words)


if __name__ == '__main__':
    main()
