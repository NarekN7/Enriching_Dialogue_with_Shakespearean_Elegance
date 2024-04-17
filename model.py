from transformers import GPT2LMHeadModel, GPT2Tokenizer


# Defining function to load model and tokenizer from a specified path
def load_model_and_tokenizer(model_path):
    # Loading pre-trained GPT-2 model from the specified path
    model = GPT2LMHeadModel.from_pretrained(model_path)
    # Loading pre-trained GPT-2 tokenizer from the specified path
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # Returning the loaded model and tokenizer
    return model, tokenizer
