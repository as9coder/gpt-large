import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
# !!! IMPORTANT: Update this path if your fine-tuned model is saved elsewhere !!!
MODEL_PATH = "./results_gpt2_large_A40"  # Path to your fine-tuned model directory
BASE_TOKENIZER_NAME = "gpt2-large"   # The base model tokenizer used during fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_text(prompt, model, tokenizer, max_length=150, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95):
    """
    Generates text from a given prompt using the fine-tuned model.
    """
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode

    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)

    # Generate text
    with torch.no_grad():  # Disable gradient calculations for inference
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,  # Important for open-ended generation
            eos_token_id=tokenizer.eos_token_id
        )

    print("\n--- Generated Text ---")
    for i, output in enumerate(outputs):
        generated_sequence = tokenizer.decode(output, skip_special_tokens=True)
        # Remove the prompt from the beginning of the generated sequence if present
        if generated_sequence.startswith(prompt):
            generated_text_only = generated_sequence[len(prompt):].lstrip()
        else:
            # Fallback if prompt is not exactly at the beginning (e.g., due to tokenization nuances)
            # This tries to find the prompt within a reasonable start window
            prompt_tokens_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            output_tokens = output.tolist()
            # Try to slice off based on input token length (more robust)
            start_index = 0
            if inputs[0].tolist() == output_tokens[:len(inputs[0])]:
                 start_index = len(inputs[0])
            
            generated_text_only = tokenizer.decode(output_tokens[start_index:], skip_special_tokens=True).lstrip()


        print(f"Sequence {i + 1}:")
        print(generated_text_only)
        print("----------------------")
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def main():
    print(f"Loading fine-tuned model from: {MODEL_PATH}")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    except OSError:
        print(f"ERROR: Could not find a fine-tuned model at {MODEL_PATH}.")
        print("Please ensure the path is correct and the model was saved successfully after training.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        return

    print(f"Loading tokenizer: {BASE_TOKENIZER_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token as it was None.")
    except Exception as e:
        print(f"An unexpected error occurred while loading the tokenizer: {e}")
        return
        
    # --- Example Usage ---
    while True:
        custom_prompt = input("\nEnter your prompt (or type 'quit' to exit):\n")
        if custom_prompt.lower() == 'quit':
            break
        if not custom_prompt.strip():
            print("Prompt cannot be empty.")
            continue
            
        print(f"Prompt: \"{custom_prompt}\"")
        generate_text(custom_prompt, model, tokenizer)

if __name__ == "__main__":
    main() 