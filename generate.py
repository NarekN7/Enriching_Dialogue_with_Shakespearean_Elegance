import torch


# Defining function to generate text using the provided model, tokenizer, and prompt
def output_result(model, tokenizer, prompt):
    # Encoding the prompt using the tokenizer and converting it to PyTorch tensor
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Create attention mask and generating text using our model
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    # Decoding the generated output into text, skipping special tokens and returning the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
