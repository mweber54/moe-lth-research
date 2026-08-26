from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories")

# Optional: save it locally
dataset.save_to_disk("./TinyStories")