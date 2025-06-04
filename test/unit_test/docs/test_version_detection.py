# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for documentation version detection functionality."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add docs source to path to import conf module
docs_source_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "docs", "source")
sys.path.insert(0, docs_source_path)

from conf import get_git_version


class TestVersionDetection(unittest.TestCase):
    """Test cases for Git version detection in Sphinx documentation."""

    @patch('subprocess.run')
    def test_exact_tag_match(self, mock_run):
        """Test version detection when HEAD is exactly on a tag."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.9.1\n"
        mock_run.return_value = mock_result
        
        version = get_git_version()
        self.assertEqual(version, "0.9.1")

    @patch('subprocess.run')
    def test_multiple_tags_at_head(self, mock_run):
        """Test version detection when HEAD has multiple tags."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "v0.9.1\nsome-other-tag\n"
        mock_run.return_value = mock_result
        
        version = get_git_version()
        self.assertEqual(version, "0.9.1")

    @patch('subprocess.run')
    def test_development_version(self, mock_run):
        """Test version detection for development builds ahead of latest tag."""
        def side_effect(*args, **kwargs):
            command = args[0]
            mock_result = MagicMock()
            mock_result.returncode = 0
            
            if "--points-at" in command:
                # No tags at current HEAD
                mock_result.stdout = ""
            elif "--list" in command:
                # Available tags
                mock_result.stdout = "v0.9.1\nv0.9.0\nv0.8.0\ndev-tag\n"
            elif "--exact-match" in command:
                # Not on exact tag
                mock_result.returncode = 1
                mock_result.stdout = ""
            
            return mock_result
        
        mock_run.side_effect = side_effect
        
        version = get_git_version()
        self.assertEqual(version, "0.9.1.dev")

    @patch('subprocess.run')
    def test_fallback_to_latest(self, mock_run):
        """Test fallback to 'latest' when no valid tags are found."""
        mock_result = MagicMock()
        mock_result.returncode = 1  # Command failed
        mock_run.return_value = mock_result
        
        version = get_git_version()
        self.assertEqual(version, "latest")

    @patch('subprocess.run')
    def test_no_semantic_version_tags(self, mock_run):
        """Test behavior when no semantic version tags are available."""
        def side_effect(*args, **kwargs):
            command = args[0]
            mock_result = MagicMock()
            mock_result.returncode = 0
            
            if "--points-at" in command:
                mock_result.stdout = ""
            elif "--list" in command:
                # Only non-semantic version tags
                mock_result.stdout = "dev-tag\nsome-branch\nrelease-candidate\n"
            
            return mock_result
        
        mock_run.side_effect = side_effect
        
        version = get_git_version()
        self.assertEqual(version, "latest")

    def test_version_import_in_conf(self):
        """Test that version is properly set in Sphinx configuration."""
        import conf
        
        # Version should be a string and not empty
        self.assertIsInstance(conf.version, str)
        self.assertNotEqual(conf.version, "")
        
        # Release should match version
        self.assertEqual(conf.version, conf.release)
        
        # Should be either a semantic version, semantic version with .dev, or "latest"
        version = conf.version
        if version != "latest":
            # Should match pattern like "0.9.1" or "0.9.1.dev"
            parts = version.replace(".dev", "").split(".")
            self.assertGreaterEqual(len(parts), 2)  # At least major.minor
            for part in parts[:3]:  # Check first 3 parts are numeric
                self.assertTrue(part.isdigit(), f"Version part '{part}' should be numeric in '{version}'")


if __name__ == "__main__":
    unittest.main()