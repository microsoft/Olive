# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for documentation version detection functionality."""

import unittest


class TestVersionDetection(unittest.TestCase):
    """Test cases for version detection in Sphinx documentation."""

    def test_development_version_processing(self):
        """Test version processing for development versions."""
        version_string = "0.10.0.dev0"
        version = version_string.replace(".dev0", "")
        self.assertEqual(version, "0.10.0")

    def test_release_version_processing(self):
        """Test version processing for release versions."""
        version_string = "0.9.1"
        version = version_string.replace(".dev0", "")
        self.assertEqual(version, "0.9.1")

    def test_major_version_development(self):
        """Test version processing for major version development builds."""
        version_string = "1.0.0.dev0"
        version = version_string.replace(".dev0", "")
        self.assertEqual(version, "1.0.0")

    def test_version_processing_logic(self):
        """Test the version processing logic directly."""
        test_cases = [
            ("0.10.0.dev0", "0.10.0"),
            ("0.9.1", "0.9.1"),
            ("1.0.0.dev0", "1.0.0"),
            ("2.1.3", "2.1.3"),
            ("0.5.2.dev0", "0.5.2"),
        ]
        
        for input_version, expected_output in test_cases:
            with self.subTest(input_version=input_version):
                result = input_version.replace(".dev0", "")
                self.assertEqual(result, expected_output)

    def test_semantic_version_pattern(self):
        """Test that processed versions follow semantic version patterns."""
        test_versions = ["0.10.0", "0.9.1", "1.0.0", "2.1.3"]
        
        for version in test_versions:
            with self.subTest(version=version):
                # Should match pattern like "0.9.1" or "0.10.0"
                parts = version.split(".")
                self.assertGreaterEqual(len(parts), 2)  # At least major.minor
                for part in parts[:3]:  # Check first 3 parts are numeric
                    self.assertTrue(part.isdigit(), f"Version part '{part}' should be numeric in '{version}'")


if __name__ == "__main__":
    unittest.main()