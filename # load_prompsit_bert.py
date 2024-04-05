# load_prompsit_bert.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

save_directory = r"C:\Users\Michalis\Desktop\ceid\HugginFace\finetuned_ready\prompsit-bert-en"

tokenizer = AutoTokenizer.from_pretrained("Prompsit/paraphrase-bert-en")
model = AutoModelForSequenceClassification.from_pretrained("Prompsit/paraphrase-bert-en")

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print("Prompsit BERT Model Loaded and Saved Successfully.")
