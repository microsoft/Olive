# app.py
# ruff: noqa: T201
from argparse import ArgumentParser

import onnxruntime_genai as og

parser = ArgumentParser(description="Run a simple chat application with the Phi-3.5 model.")
parser.add_argument(
    "-m",
    "--model_folder",
    type=str,
    default="models/phi3_5-qdq",
    help="Path to the folder containing the outputs of Olive run",
)
args = parser.parse_args()

# Load the base model and tokenizer
model = og.Model(f"{args.model_folder}/model")
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()

# Set the max length to something sensible by default,
# since otherwise it will be set to the entire context length
search_options = {}
search_options["max_length"] = 200

# Keep asking for input prompts in a loop
while True:
    text = input("Prompt (Use quit() to exit): ")
    if not text:
        print("Error, input cannot be empty")
        continue

    if text == "quit()":
        break

    # Generate prompt (prompt template + input)
    prompt = tokenizer.apply_chat_template(
        messages=f"""[{{"role": "user", "content": "{text}"}}]""", add_generation_prompt=True
    )

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
