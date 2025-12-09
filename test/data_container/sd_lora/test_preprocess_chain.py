# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from olive.data.component.sd_lora.dataset import ImageFolderDataset
from olive.data.component.sd_lora.preprocess_chain import image_lora_preprocess


class TestImageLoraPreprocess:
    @pytest.fixture
    def mock_dataset(self, tmp_path):
        img = Image.new("RGB", (512, 512), color="red")
        img.save(tmp_path / "test.jpg")
        (tmp_path / "test.txt").write_text("test caption")
        return ImageFolderDataset(data_dir=str(tmp_path))

    def test_default_runs_bucketing(self, mock_dataset, tmp_path):
        """Without steps, default (aspect_ratio_bucketing) is used."""
        output_dir = tmp_path / "output"
        result = image_lora_preprocess(mock_dataset, base_resolution=512, output_dir=str(output_dir))
        assert hasattr(result, "bucket_assignments")

    def test_custom_steps_replaces_default(self, mock_dataset, tmp_path):
        """When steps is provided, only those steps run (no default bucketing)."""
        output_dir = tmp_path / "output"
        with patch("olive.data.component.sd_lora.preprocess_chain.Registry.get_pre_process_component") as mock_registry:
            mock_fn = MagicMock(return_value=mock_dataset)
            mock_registry.return_value = mock_fn

            image_lora_preprocess(
                mock_dataset,
                base_resolution=512,
                steps={"image_filtering": {"min_size": 256}},
                output_dir=str(output_dir),
            )

            calls = [str(c) for c in mock_registry.call_args_list]
            assert any("image_filtering" in c for c in calls)
            assert not any("aspect_ratio_bucketing" in c for c in calls)
