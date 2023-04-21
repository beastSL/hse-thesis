from argparse import ArgumentParser
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=Path, help="Path to the sentence-tokenized dataset"
    )
    parser.add_argument(
        "--markup-path", type=Path, help="Path to the file containing markup"
    )
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    dataset = pd.read_csv(dataset_path)

    markup_path = args.markup_path

    try:
        marked_dataset = pd.read_csv(markup_path)    
    except:
        marked_dataset = pd.DataFrame(columns=['sentence_id', 'score'])
    last_sentence_id = marked_dataset['sentence_id'].max() if len(marked_dataset) else -1
    
    unmarked_dataset = dataset[dataset['sentence_id'] > last_sentence_id]
    for index, row in unmarked_dataset.iterrows():
        paper_id = row['paper_id']
        sentence_id = row['sentence_id']

        prev_context = dataset.iloc[max(0, index - 2):index]
        next_context = dataset.iloc[index + 1:min(len(dataset), index + 4)]

        prev_context = ' '.join(prev_context[prev_context['paper_id'] == paper_id]['text'])
        next_context = ' '.join(next_context[next_context['paper_id'] == paper_id]['text'])

        print()
        print(prev_context)
        print("=========================")
        print(row['text'])
        print("=========================")
        print(next_context)
        print()
        print("Write a number from 0 to 5 (0 for Non-Applicable)")

        score = int(input())

        marked_dataset = pd.concat([marked_dataset, pd.DataFrame([{'sentence_id': sentence_id, 'score': score}])], ignore_index=True)
        marked_dataset.to_csv(markup_path, index=False)
