# app.py
# ruff: noqa: T201
import onnxruntime_genai as og

model_folder = "models/model"
max_length = 2048

# Load the base model and tokenizer
config = og.Config(model_folder)
config.clear_providers()
config.append_provider('cuda')

model = og.Model(config)
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options['batch_size'] = 1
search_options["max_length"] = max_length

# Keep asking for input prompts in a loop
while True:
    text = input("Prompt (Use quit() to exit): ")
    if not text:
        print("Error, input cannot be empty")
        continue

    if text == "quit()":
        break

    # Create params and generator
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)

    # Generate system prompt (prompt template + input)
    messages = """[{"role": "system", "content": "You are a helpful AI assistant."}]"""
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=False)
    # Encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)
    # Append input tokens to the generator
    generator.append_tokens(input_tokens)

    # Generate user prompt (prompt template + input)
    messages = f"""[{{"role": "user", "content": "{text}"}}]"""
    # Apply the chat template
    prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
    # Encode the prompt using the tokenizer
    input_tokens = tokenizer.encode(prompt)
    # Append input tokens to the generator
    generator.append_tokens(input_tokens)

    print("")
    print("Output: ", end="", flush=True)
    # Stream the output
    count = 0
    try:
        while not generator.is_done() and count < max_length:
            generator.generate_next_token()
            count += 1

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")
    print()
    print()

    del generator
