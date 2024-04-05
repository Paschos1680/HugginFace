# load_xlm_r_multilingual.py
from transformers import AutoTokenizer, AutoModel

save_directory = r"C:\Users\Michalis\Desktop\ceid\HugginFace\finetuned_ready\xlm-r-multilingual-v1"

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-xlm-r-multilingual-v1")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("XLM-R Multilingual Model Loaded and Saved Successfully.")
