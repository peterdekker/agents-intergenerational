import numpy as np
from constants import N_FEATURES, LEARNING_RATE, NOISE_RATE

def is_prefixing_verb(prefixes):
    return '' not in prefixes # TODO: only minimal check, what with more affixes?

def is_suffixing_verb(suffixes):
    return '' not in suffixes # TODO: only minimal check, what with more affixes?


# def update_language(language, signal_own, signal_received, feedback):
#     replace_positions = np.random.choice([True, False], size=N_FEATURES, p=[LEARNING_RATE, 1-LEARNING_RATE])
#     if feedback:
#         # If positive feedback: replace positions by received signal
#         signal_own.put(replace_positions, signal_received)

#         #zeros_vector = np.zeros(N_FEATURES)
#         #signal_own.put(replace_positions, zeros_vector)
#     else:
#         a = signal_received
#         signal_received_inv = np.where((a == 0) | (a == 1), a ^ 1, a)
#         signal_own.put(replace_positions, signal_received_inv)
#         #random_vector = np.random.randint(0,2, N_FEATURES)
#         #signal_own.put(replace_positions, random_vector)


# def apply_noise(signal):
#     # Apply noise by replacing some bits
#     replace_positions = np.random.choice([True, False], size=N_FEATURES, p=[NOISE_RATE, 1-NOISE_RATE])
#     random_vector = np.random.randint(0, 2, N_FEATURES)
#     signal.put(replace_positions, random_vector)
