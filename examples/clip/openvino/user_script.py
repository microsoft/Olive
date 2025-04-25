from io import BytesIO

import requests
import torch
from datasets import load_dataset
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from olive.data.registry import Registry

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------

seed = 0
# seed everything to 0 for reproducibility, https://pytorch.org/docs/stable/notes/randomness.html
# do not set random seed and np.random.seed for aml test, since it will cause aml job name conflict
torch.manual_seed(seed)
# the following are needed only for GPU
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def check_text_data(data):
    """Check if the given data is text-based."""
    if isinstance(data, str):
        return True
    if isinstance(data, list):
        return all(isinstance(x, str) for x in data)
    return False


def get_pil_from_url(url):
    """Download and convert an image from a URL to a PIL Image object."""
    response = requests.get(url, verify=True, timeout=20)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


def wrap_collate_fn(processor, max_length):
    def collate_fn(example, image_column="image_url", text_column="caption"):
        """Preprocess an example by loading and transforming image and text data.

        Check if the text data in the example is valid by calling the `check_text_data` function.
        Download the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
        If there is any error during the download process, return None.
        Return the preprocessed inputs with transformed image and text data.
        """
        if len(example) != 1:
            raise ValueError(f"Expected 'example' to have exactly one element, but got {len(example)}.")
        example = example[0]

        if not check_text_data(example[text_column]):
            raise ValueError("Text data is not valid")

        url = example[image_column]
        try:
            image = get_pil_from_url(url)
            w, h = image.size
            if h == 1 or w == 1:
                return None
        except Exception:
            return None

        inputs = processor(text=example[text_column], images=[image], return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[1] > max_length:
            return None
        return inputs

    return collate_fn


def prepare_calibration_data(dataloader, init_steps):
    """Prepare calibration data from a dataloader for a specified number of initialization steps.

    Iterate over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    with tqdm(total=init_steps) as pbar:
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch:
                pbar.update(1)
                with torch.no_grad():
                    data.append(
                        {
                            "input_ids": batch["input_ids"].to("cpu"),
                            "pixel_values": batch["pixel_values"].to("cpu"),
                            "attention_mask": batch["attention_mask"].to("cpu"),
                        }
                    )
    return data


@Registry.register_dataset()
def conceptual_captions_dataset(opt_init_steps=200, max_train_samples=1000, **kwargs):
    """Prepare a vision-text dataset for quantization."""
    dataset = load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True)
    model_path = kwargs.get("model_path")
    if not model_path:
        raise ValueError(
            "The 'model_path' parameter is required in data_configs.load_dataset_config but was not provided."
        )
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    max_length = model.config.text_config.max_position_embeddings
    train_dataset = dataset["train"].shuffle(seed=seed)
    collate_fn = wrap_collate_fn(processor, max_length)
    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
    return prepare_calibration_data(dataloader, opt_init_steps)


def custom_transform_func(data_item):
    np_inputs = {}
    for inp in data_item:
        # Drop the first dimension using slicing
        np_inputs[inp] = data_item[inp].numpy()[0, ...]
    return np_inputs
