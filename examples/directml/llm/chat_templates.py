# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

LLAMA_2_CHAT_UNCENSORED_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
### HUMAN:
{{message['content']}}
### RESPONSE:
{% endif %}
{% if message['role'] == 'assistant' %}
### RESPONSE:
{{message['content']}}
{% endif %}
{% endfor %}
"""

ORCA_MINI_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
### User:
{{message['content']}}
### Response:
{% endif %}
{% if message['role'] == 'assistant' %}
### Response:
{{message['content']}}
{% endif %}
{% endfor %}
"""

VICUNA_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
USER:
{{message['content']}}
ASSISTANT:
{% endif %}
{% if message['role'] == 'assistant' %}
ASSISTANT:
{{message['content']}}
{% endif %}
{% endfor %}
"""

CHATML_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
<|im_start|>user
{{message['content']}}<|im_end|>
<|im_start|>assistant
{% endif %}
{% if message['role'] == 'assistant' %}
<|im_start|>assistant:
{{message['content']}}<|im_end|>
{% endif %}
{% endfor %}
"""

ZEPHYR_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
<|user|>
{{message['content']}}</s>
<|assistant|>
{% endif %}
{% if message['role'] == 'assistant' %}
<|assistant|>
{{message['content']}}</s>
{% endif %}
{% endfor %}
"""

ALPACA_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
### Instruction:
{{message['content']}}
### Response:
{% endif %}
{% if message['role'] == 'assistant' %}
### Response:
{{message['content']}}
{% endif %}
{% endfor %}
"""

OPENCHAT_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
GPT4 Correct User: {{message['content']}}<|end_of_turn|>GPT4 Correct Assistant:
{% endif %}
{% if message['role'] == 'assistant' %}
GPT4 Correct Assistant:
{{message['content']}}
{% endif %}
{% endfor %}
"""

NEURALCHAT_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
### User:
{{message['content']}}
### Assistant:
{% endif %}
{% if message['role'] == 'assistant' %}
### Assistant:
{{message['content']}}
{% endif %}
{% endfor %}
"""

FALCON_CHAT_TEMPLATE = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ ' ' }}
{% endif %}
{{ message['content'] }}
{% if not loop.last %}
{{ '  ' }}
{% endif %}
{% endfor %}
{{ eos_token }}
"""

PHI_2_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "Human: {{ message['content'] }}\nAI:"
    "{% endif %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
)

# TODO (pavignol): Change name of template for actual type
LOCAL_CHAT_TEMPLATE = """{% for message in messages %}
{% if message['role'] == 'user' %}<|user|>
{{message['content']}}<|end|>
<|assistant|>{% endif %}
{% if message['role'] == 'assistant' %}
{{message['content']}}{% endif %}
{% endfor %}"""


def get_chat_template(model_type: str):
    return {
        "llama-2-7b-chat-uncensored": LLAMA_2_CHAT_UNCENSORED_CHAT_TEMPLATE,
        "orca-mini-7b": ORCA_MINI_CHAT_TEMPLATE,
        "vicuna-7b-v1.5": VICUNA_CHAT_TEMPLATE,
        "dolphin-2.6-mistral-7b": CHATML_CHAT_TEMPLATE,
        "zephyr-7b-beta": ZEPHYR_CHAT_TEMPLATE,
        "openhermes-2.5-mistral-7b": CHATML_CHAT_TEMPLATE,
        "noushermes-llama-2-7b": ALPACA_CHAT_TEMPLATE,
        "openchat-7b-3.5": OPENCHAT_CHAT_TEMPLATE,
        "neural-chat-7b-v3.1": NEURALCHAT_CHAT_TEMPLATE,
        "tinyllama-1.1b-chat-v0.6": ZEPHYR_CHAT_TEMPLATE,
        "falcon-7b-chat": FALCON_CHAT_TEMPLATE,
        "phi-2": PHI_2_CHAT_TEMPLATE,
        "local": LOCAL_CHAT_TEMPLATE,
    }.get(model_type)
