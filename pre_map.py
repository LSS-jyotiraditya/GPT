# slms/pre_map.py
import os
import argparse
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer

def build_argparser():
	parser = argparse.ArgumentParser(description="Pre-tokenize TinyStories and save to disk.")
	parser.add_argument("--tokenizer-file", type=str, default="data/TinyStories-tokenizer.json", help="Path to tokenizer json.")
	parser.add_argument("--output-dir", type=str, required=True, help="Directory to save tokenized dataset.")
	parser.add_argument("--block-size", type=int, default=1080, help="Sequence length for pad/truncate.")
	parser.add_argument("--pad-token", type=str, default="<|im_end|>", help="Pad token string.")
	parser.add_argument("--pad-id", type=int, default=2, help="Pad token id.")
	parser.add_argument("--train-count", type=int, default=30000, help="Number of train examples to keep (-1 for all).")
	parser.add_argument("--val-count", type=int, default=3000, help="Number of val examples to keep (-1 for all).")
	parser.add_argument("--num-proc", type=int, default=None, help="Number of processes for map().")
	parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
	return parser

def main():
	args = build_argparser().parse_args()

	print(f"Loading dataset: roneneldan/TinyStories")
	raw_ds = load_dataset("roneneldan/TinyStories")

	print(f"Loading tokenizer from: {args.tokenizer_file}")
	tokenizer = Tokenizer.from_file(args.tokenizer_file)
	tokenizer.enable_padding(pad_id=args.pad_id, pad_token=args.pad_token, length=args.block_size)
	tokenizer.enable_truncation(max_length=args.block_size)

	def encode_batch(batch):
		encs = tokenizer.encode_batch(batch["text"])
		return {"input_ids": [e.ids for e in encs]}

	print("Tokenizing...")
	tokenized = raw_ds.map(
		encode_batch,
		batched=True,
		remove_columns=["text"],
		num_proc=args.num_proc,
		desc="Tokenizing TinyStories",
	)

	# Optional downsampling (to match your training script defaults)
	def maybe_select(split, keep_count):
		if keep_count is None or keep_count < 0:
			return split
		if keep_count > len(split):
			return split
		return split.shuffle(seed=args.seed).select(range(keep_count))

	print("Downsampling (if requested)...")
	train_tok = maybe_select(tokenized["train"], args.train_count)
	val_tok = maybe_select(tokenized["validation"], args.val_count)

	ds = DatasetDict(train=train_tok, validation=val_tok)

	# Save to disk
	out_dir = args.output_dir
	os.makedirs(out_dir, exist_ok=True)
	print(f"Saving tokenized dataset to: {out_dir}")
	ds.save_to_disk(out_dir)
	print("Done.")
	print(f"Train examples: {len(ds['train'])}, Val examples: {len(ds['validation'])}")
	print(f"Block size: {args.block_size}, Pad id: {args.pad_id}, Pad token: {args.pad_token}")

if __name__ == "__main__":
	main()
