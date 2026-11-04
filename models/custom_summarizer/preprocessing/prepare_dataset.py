
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR.parent / "dataset" / "train.jsonl"
OUTPUT_FILE = BASE_DIR.parent / "dataset" / "processed_train.jsonl"


def prepare_dataset():

    processed_data = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            meeting = json.loads(line)

            # Combine all meeting speakers and their dialogue
            transcript_parts = []

            for segment in meeting["meeting_transcripts"]:
                speaker = segment["speaker"]
                content = segment["content"]

                transcript_parts.append(f"{speaker}: {content}")

            full_transcript = "\n".join(transcript_parts)

            # Extract general and specific queries
            all_queries = (
                meeting["general_query_list"]
                + meeting["specific_query_list"]
            )

            for item in all_queries:

                processed_data.append({
                    "input": full_transcript,
                    "query": item["query"],
                    "target": item["answer"]
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for item in processed_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Dataset preparation completed!")
    print("Total examples:", len(processed_data))


if __name__ == "__main__":
    prepare_dataset()
