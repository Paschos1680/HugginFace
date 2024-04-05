# load_multilingual_minilm.py
from transformers import AutoTokenizer, AutoModel

save_directory = r"C:\Users\Michalis\Desktop\ceid\HugginFace\finetuned_ready\multilingual-MiniLM-L12-v2"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Multilingual MiniLM L12 V2 Model Loaded and Saved Successfully.")
