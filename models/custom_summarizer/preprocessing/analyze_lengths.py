import json

DATASET_PATH = "models/custom_summarizer/dataset/processed_train.jsonl"

input_lengths = []
target_lengths = []

# Read the processed dataset
with open(DATASET_PATH, "r", encoding="utf-8") as file:

    for line in file:
        example = json.loads(line)

        # Count transcript words
        input_tokens = example["input"].lower().split()

        # Count summary words
        target_tokens = example["target"].lower().split()

        input_lengths.append(len(input_tokens))
        target_lengths.append(len(target_tokens))


# Calculate statistics
print("Total examples:", len(input_lengths))

print("\nINPUT (Meeting Transcripts)")
print("Average length:", sum(input_lengths) / len(input_lengths))
print("Maximum length:", max(input_lengths))
print("Minimum length:", min(input_lengths))

print("\nTARGET (Summaries)")
print("Average length:", sum(target_lengths) / len(target_lengths))
print("Maximum length:", max(target_lengths))
print("Minimum length:", min(target_lengths))