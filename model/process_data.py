from argparse import ArgumentParser
from pathlib import Path

from data import train_tokenizers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "base_dir", type=Path, help="Path to the directory containing original data"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="The path that the trained tokenizer will be saved into",
    )
    args = parser.parse_args()
    
    train_tokenizers(args.base_dir, args.tokenizer_path)
