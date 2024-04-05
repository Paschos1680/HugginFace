# load_mpnet_base_v2.py
from transformers import AutoTokenizer, AutoModel

save_directory = r"C:\Users\Michalis\Desktop\ceid\HugginFace\finetuned_ready\mpnet-base-v2"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("MPNet Base V2 Model Loaded and Saved Successfully.")
