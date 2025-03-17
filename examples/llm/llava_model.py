# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from decoder_model import DecoderModel
from transformers import AutoModel, CLIPVisionModel, LlavaConfig


class LlavaMultiModalProjector(torch.nn.Module):
    def __init__(self, config: LlavaConfig, vision_config):
        super().__init__()

        text_hidden_size = config.text_config.hidden_size if hasattr(config, "text_config") else config.n_embd
        self.linear_1 = torch.nn.Linear(vision_config.hidden_size, text_hidden_size, True)
        self.linear_2 = torch.nn.Linear(text_hidden_size, text_hidden_size, True)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = torch.nn.functional.gelu(hidden_states)  # pylint: disable=not-callable
        return self.linear_2(hidden_states)


class LlavaModel(torch.nn.Module):
    def __init__(self, llava_config: LlavaConfig):
        super().__init__()
        self.llava_config = llava_config

        if hasattr(llava_config, "vision_config"):
            self.vision_tower = AutoModel.from_config(llava_config.vision_config)
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(llava_config.img_processor["model_name"])

        self.multi_modal_projector = LlavaMultiModalProjector(llava_config, self.vision_tower.config)
        self.vocab_size = llava_config.vocab_size
        self.language_model = DecoderModel(use_embeddings=True)

        self.pad_token_id = self.llava_config.pad_token_id if self.llava_config.pad_token_id is not None else -1

    # This function has mostly been borrowed from the following transformers file, but modified to suit our needs
    # https://github.com/huggingface/transformers/blob/03cc17775b961d16cc4d0d7ab0c8487120d0b708/src/transformers/models/llava/modeling_llava.py#L279
    def _merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids, attention_mask):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        image_token_index = getattr(self.llava_config, "image_token_index", 32000)
        special_image_token_mask = input_ids == image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by
        # `nb_text_tokens_per_images - 1` text tokens. `torch.cumsum` computes how each image token shifts subsequent
        # text token positions. - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite = torch.logical_and(
            image_to_overwrite, image_to_overwrite.long().cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)
        )

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is "
                f"{torch.sum(special_image_token_mask)} while the number of image given to the model is {num_images}. "
                f"This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask = torch.logical_or(final_attention_mask, image_to_overwrite).long()

        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, position_ids

    def forward(self, input_ids, attention_mask, pixel_values, past_key_values):
        # 1. Extract the input embeddings
        inputs_embeds = self.language_model.get_embeddings()(input_ids)

        # 2. Merge text and images
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        # layer_idx = getattr(self.llava_config, "vision_feature_layer", -2)
        layer_idx = getattr(self.llava_config, "vision_feature_layer", -2)
        selected_image_feature = image_outputs.hidden_states[layer_idx]

        if getattr(self.llava_config, "vision_feature_select_strategy", "default") == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif self.llava_config.vision_feature_select_strategy != "full":
            raise ValueError(f"Unexpected select feature strategy: {self.llava_config.vision_feature_select_strategy}")

        image_features = self.multi_modal_projector(selected_image_feature)
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask
        )

        return self.language_model(inputs_embeds, position_ids, attention_mask, past_key_values)

    def forward_use_cache(self, input_ids_increment, attention_mask, past_key_values):
        # 1. Extract the input embeddings
        inputs_embeds = self.language_model.get_embeddings()(input_ids_increment)
        position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
        return self.language_model(inputs_embeds, position_ids, attention_mask, past_key_values)
