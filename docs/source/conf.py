# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import subprocess
import sys

# ruff: noqa
# pylint: skip-file
sys.path.append(os.path.abspath("exts"))
def get_git_version():
    """Get the current version from Git tags or fallback to 'latest'."""
    try:
        # Check if current HEAD matches any tag (exact release)
        result = subprocess.run(
            ["git", "tag", "--points-at", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        if result.returncode == 0 and result.stdout.strip():
            tags = result.stdout.strip().split('\n')
            # Prefer semantic version tags
            for tag in tags:
                if tag.startswith('v') and len(tag.split('.')) >= 3:
                    return tag[1:]  # Remove 'v' prefix
            # Fallback to first tag
            return tags[0][1:] if tags[0].startswith('v') else tags[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    try:
        # Get the latest semantic version tag for development builds
        result = subprocess.run(
            ["git", "tag", "--list"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        if result.returncode == 0 and result.stdout.strip():
            # Filter for semantic version tags and get the latest
            tags = result.stdout.strip().split('\n')
            version_tags = []
            for tag in tags:
                if tag.startswith('v') and len(tag.split('.')) >= 3:
                    try:
                        # Check if it's a proper semantic version
                        parts = tag[1:].split('.')
                        if len(parts) >= 3 and all(part.isdigit() for part in parts[:3]):
                            version_tags.append(tag)
                    except (ValueError, IndexError):
                        continue
            
            if version_tags:
                # Sort by version number and get the latest
                version_tags.sort(key=lambda x: [int(part) for part in x[1:].split('.')[:3]])
                latest_tag = version_tags[-1]
                
                # Check if we're on the exact tag or ahead of it
                try:
                    result = subprocess.run(
                        ["git", "describe", "--tags", "--exact-match", "HEAD"],
                        capture_output=True,
                        text=True,
                        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    )
                    if result.returncode == 0:
                        # We're on an exact tag
                        return latest_tag[1:]
                except (subprocess.SubprocessError, FileNotFoundError):
                    pass
                
                # We're ahead of the latest tag, add dev suffix
                return f"{latest_tag[1:]}.dev"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Default fallback
    return "latest"


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Olive"
copyright = "2023-2025, Olive Dev team"
version = get_git_version()
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "auto_config_doc",
    "sphinxarg.ext",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.jquery",
    "gallery_directive",
]

myst_enable_extensions = [
    "html_image",
    "colon_fence",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

suppress_warnings = ["myst.xref_missing"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = [
    # better contrast between h3 and h4, high priority so that it overrides the theme
    ("css/header.css", {"priority": 1000}),
]
html_js_files = [
    "js/custom_version.js",
]

html_theme_options = {
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/microsoft/Olive",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/olive-ai",
            "icon": "fa-custom fa-pypi",
        },
    ],
    "show_toc_level": 1,
    "navbar_align": "left",
    # "announcement": "Announcement: This is an example announcement.",
    "show_version_warning_banner": True,
    "navbar_center": ["navbar-nav"],
    "navbar_start": ["navbar-logo"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    },
}

html_sidebars = {
    "**": [],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
todo_include_todos = False

# pydantic defaults
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_config_member = False
autodoc_pydantic_settings_show_validator_summary = False
autodoc_pydantic_settings_show_validator_members = False
autodoc_pydantic_settings_show_field_summary = False
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_settings_member_order = "bysource"  # is groupwise and alphabetical otherwise

# disable the anchor check since https://github.com/sphinx-doc/sphinx/issues/9016
# we could enable it when the issue is fixed
linkcheck_anchors = False
linkcheck_ignore = [
    # TODO(trajep): remove this when the issue is fixed
    r"https://developer.qualcomm.com/*",
    r"https://docs.qualcomm.com/*",
    # TODO(jambayk): remove this when the issue is fixed
    r"https://www.intel.com/*",
    # TODO(team): html files are generated after doc build. Linkcheck doesn't work for them.
    # Remove this when linkcheck works for html files.
    r"^(?!https).*\.html$",
]
