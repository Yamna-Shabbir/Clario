
import json
from collections import Counter

DATASET_PATH = "models/custom_summarizer/dataset/processed_train.jsonl"

# Special tokens
special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

word_counts = Counter()

# Read our processed dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:

    for line in file:
        example = json.loads(line)

        # Count words from inputs and targets
        text = example["input"] + " " + example["target"]

        tokens = text.lower().split()

        word_counts.update(tokens)

# Start vocabulary with special tokens
vocabulary = {
    token: index
    for index, token in enumerate(special_tokens)
}

MAX_VOCAB_SIZE = 20000

for word, count in word_counts.most_common():

    if len(vocabulary) >= MAX_VOCAB_SIZE:
        break

    if word not in vocabulary:
        vocabulary[word] = len(vocabulary)
# Save vocabulary
with open(
    "models/custom_summarizer/dataset/vocabulary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(vocabulary, file, indent=2)

print("Vocabulary created!")
print("Vocabulary size:", len(vocabulary))
print("\nFirst 20 entries:")

for word, index in list(vocabulary.items())[:20]:
    print(word, ":", index)
print("\nVocabulary statistics:")

print("Total unique words:", len(word_counts))

print("Words appearing only once:",
      sum(1 for count in word_counts.values() if count == 1))

print("Words appearing 5 or more times:",
      sum(1 for count in word_counts.values() if count >= 5))