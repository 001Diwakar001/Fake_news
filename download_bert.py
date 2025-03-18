from transformers import BertForSequenceClassification, BertTokenizer

# Choose the appropriate BERT model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # Change if needed
save_path = "backend/bert_model"  # Folder where model will be saved

# Download and save locally
model = BertForSequenceClassification.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"BERT model saved in '{save_path}'")
