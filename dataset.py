from transformers import TextDataset, DataCollatorForLanguageModeling


# Defining function to create dataset from the provided file using already defined tokenizer
def create_dataset(tokenizer, file_path, block_size):
    # Returning TextDataset created from the file_path and tokenizer with specified block_size
    return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)


# Defining function to create data collator for language modeling
def create_data_collator(tokenizer):
    # Returning DataCollatorForLanguageModeling with MLM (Masked Language Modeling) disabled
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
