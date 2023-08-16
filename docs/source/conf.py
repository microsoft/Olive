import os
import sys

import sphinx_rtd_theme

sys.path.append(os.path.abspath("exts"))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Olive"
copyright = "2023, olivedevteam@microsoft.com"
author = "olivedevteam@microsoft.com"
version = "latest"

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
    "sphinx_tabs.tabs",
    # "nbsphinx",
    "auto_config_doc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.jquery",
]

myst_enable_extensions = [
    "html_image",
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
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]
html_css_files = [
    "css/width.css",
]
html_js_files = [
    "js/custom_version.js",
]

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

# disable the anchor check since https://github.com/sphinx-doc/sphinx/issues/9016
# we could enable it when the issue is fixed
linkcheck_anchors = False
