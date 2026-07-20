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
copyright = "2023-2026, Olive Dev team"
# The docs version is provided by CI. Default to main for local/dev builds.
version = os.getenv("OLIVE_DOCS_VERSION", "main")
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


def setup(app):
    # sphinx-argparse >= 0.6.0 registers a "commands" domain that does not implement
    # resolve_any_xref. MyST then falls back to calling its resolve_xref for every
    # markdown link, which logs "Error, no command xref target ..." for each unmatched
    # target. Providing resolve_any_xref lets MyST resolve real command targets and
    # skips the noisy fallback (and the "myst.domains" legacy-domain warning).
    try:
        from sphinxarg.ext import ArgParseDomain
        from sphinxarg.utils import target_to_anchor_id
    except ImportError:
        # Older sphinx-argparse (< 0.6.0) has no "commands" domain, nothing to patch.
        return

    # Don't override a native implementation a future release might provide.
    if "resolve_any_xref" in ArgParseDomain.__dict__:
        return

    from sphinx.util.nodes import make_refnode

    def resolve_any_xref(self, env, fromdocname, builder, target, node, contnode):
        anchor_id = target_to_anchor_id(target)
        results = []
        for _cmd, _sig, _type, docname, anchor, _prio in self.get_objects():
            if anchor_id == anchor:
                refnode = make_refnode(builder, fromdocname, docname, anchor, contnode, anchor)
                results.append(("commands:command", refnode))
        return results

    ArgParseDomain.resolve_any_xref = resolve_any_xref


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = [
    # better contrast between h3 and h4, high priority so that it overrides the theme
    ("css/header.css", {"priority": 1000}),
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
    "navbar_end": ["version-switcher"],
    "footer_start": ["copyright"],
    "secondary_sidebar_items": {
        "**": ["page-toc"],
    },
    "switcher": {
        "json_url": "./_static/versions.json",
        "version_match": version,
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
