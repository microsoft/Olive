# app.py
# ruff: noqa: T201
import onnxruntime_genai as og

model_folder = "models/model"

# Load the base model and tokenizer
model = og.Model(model_folder)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options["max_length"] = 1000

# chat_template = "<|user|>\n{input} <|end|>\n<|assistant|>"
chat_template = "<|im_start|>system\nYou are Qwen, a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n"

# Keep asking for input prompts in a loop
while True:
    text = input("Prompt (Use quit() to exit): ")
    if not text:
        print("Error, input cannot be empty")
        continue

    if text == "quit()":
        break

    # Generate prompt (prompt template + input)
    prompt = f"{chat_template.format(input=text)}"

    # Encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)

    # Create params and generator
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    # Append input tokens to the generator
    generator.append_tokens(input_tokens)

    print("")
    print("Output: ", end="", flush=True)
    # Stream the output
    try:
        while not generator.is_done():
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")
    print()
    print()

    del generator
