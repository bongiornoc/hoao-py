"""Sphinx configuration for the public API and numerical conventions."""

project = "HOAO"
author = "Christian Bongiorno and Paul Ruelloux"
copyright = "2026, Christian Bongiorno and Paul Ruelloux"
release = "0.1.0"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.doctest", "sphinx.ext.mathjax", "numpydoc"]
numpydoc_show_class_members = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
html_theme = "alabaster"
exclude_patterns = ["_build"]
nitpicky = False
