from transformers import AutoTokenizer, AutoModelForTokenClassification

model_name = "GroNLP/bert-base-dutch-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=5
)

text = "Sloop betonvloer 120 m2 à 35,00 per m2"
inputs = tokenizer(text, return_tensors="pt", truncation=True)

outputs = model(**inputs)
pred_ids = outputs.logits.argmax(dim=-1)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for token, pred in zip(tokens, pred_ids[0]):
    print(token, pred.item())