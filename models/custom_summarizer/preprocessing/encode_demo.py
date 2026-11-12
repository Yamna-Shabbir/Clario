import json

# Load our vocabulary
with open(
    "models/custom_summarizer/dataset/vocabulary.json",
    "r",
    encoding="utf-8"
) as file:
    vocabulary = json.load(file)

# A sample sentence
sentence = "The team discussed the project deadline."

# Convert sentence into lowercase tokens
tokens = sentence.lower().split()

# Convert each token into its vocabulary ID
token_ids = []

for word in tokens:
    if word in vocabulary:
        token_ids.append(vocabulary[word])
    else:
        token_ids.append(vocabulary["<UNK>"])

token_ids.insert(0, vocabulary["<BOS>"])
token_ids.append(vocabulary["<EOS>"])

print("Original sentence:", sentence)
print("Tokens:", tokens)
print("Numerical IDs:", token_ids)