# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys

# ruff: noqa
# pylint: skip-file
sys.path.append(os.path.abspath("exts"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Olive"
copyright = "2023-2025, Olive Dev team"
version = "0.8.0"

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
