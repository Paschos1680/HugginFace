# load_eugenesiow_bart.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

save_directory = r"C:\Users\Michalis\Desktop\ceid\HugginFace\finetuned_ready\eugenesiow-bart"

tokenizer = AutoTokenizer.from_pretrained("eugenesiow/bart-paraphrase")
model = AutoModelForSeq2SeqLM.from_pretrained("eugenesiow/bart-paraphrase")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Eugenesiow BART Model Loaded and Saved Successfully.")
