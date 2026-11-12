# Three example sequences
sequences = [
    [2, 5, 8, 3],
    [2, 5, 9, 10, 11, 3],
    [2, 7, 3]
]

# Find the longest sequence length
max_length = max(len(sequence) for sequence in sequences)

print("Maximum length:", max_length)

# Pad shorter sequences
padded_sequences = []

for sequence in sequences:
    while len(sequence) < max_length:
        sequence.append(0)

    padded_sequences.append(sequence)

print("\nPadded sequences:")

for sequence in padded_sequences:
    print(sequence)
# Example of truncation while preserving EOS
sequence = [2, 5, 8, 9, 10, 11, 12, 3]

max_length = 6

truncated_sequence = sequence[:max_length - 1] + [sequence[-1]]

print("\nOriginal:", sequence)
print("Truncated:", truncated_sequence)
def pad_or_truncate(sequence, max_length):
    if len(sequence) > max_length:
        sequence = sequence[:max_length - 1] + [sequence[-1]]

    while len(sequence) < max_length:
        sequence.append(0)

    return sequence

print(pad_or_truncate([2, 5, 8, 3], 6))

print(pad_or_truncate([2, 5, 8, 9, 10, 11, 12, 3], 6))