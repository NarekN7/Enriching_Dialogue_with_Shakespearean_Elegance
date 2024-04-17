from transformers import Trainer, TrainingArguments, GPT2Tokenizer
from dataset import create_dataset, create_data_collator
from model import load_model_and_tokenizer
from generate import output_result
import argparse
import re


# Defining function to train the model
def train_model(model, args, train_dataset, data_collator):
    # Loading tokenizer and setting pad token
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Defining training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    # Initializing Trainer with model, training arguments, and datasets
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Training the model
    trainer.train()

    # Saving trained model and tokenizer
    output_path = args.output_dir
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


# Defining main function
def main(args):
    # If training flag is provided
    if args.train:
        model, tokenizer = load_model_and_tokenizer(
            "gpt2"
        )  # Loading model and tokenizer
        train_dataset = create_dataset(
            tokenizer, args.file_path, args.block_size
        )  # Creating training dataset
        data_collator = create_data_collator(tokenizer)  # Creating data collator
        train_model(model, args, train_dataset, data_collator)  # Train the model
    else:
        # If prompt flag is provided
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        if args.prompt:
            import logging

            logging.getLogger("transformers").setLevel(logging.ERROR)
            # Generate text using model and tokenizer and clean up extra newlines
            generated_text = output_result(model, tokenizer, args.prompt)
            generated_text = re.sub(r"\s{3,}", "\n\n", generated_text)
            print(generated_text)
        else:
            # If prompt is not provided, prompting the user
            print(
                "Please provide a prompt using the '--prompt' argument for text generation."
            )


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Shakespeare Generation")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--file_path", type=str, default="data.txt", help="Path to the training data"
    )
    parser.add_argument(
        "--block_size", type=int, default=128, help="Block size for training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for saving trained model",
    )
    parser.add_argument(
        "--model_path", type=str, default="output", help="Path to the trained model"
    )
    parser.add_argument("--prompt", type=str, help="Prompt for generating text")
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device during training",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=10,
        help="Save checkpoint every X updates steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints.",
    )
    args = parser.parse_args()
    main(args)
