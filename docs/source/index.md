---
myst:
  html_meta:
    "description lang=en": |
      Top-level documentation for pydata-sphinx theme, with links to the rest
      of the site..
html_theme.sidebar_secondary.remove: true
---
  
# OLIVE: AI Model Optimization Toolkit for the ONNX Runtime


```{gallery-grid}
:grid-columns: 1 2 2 3

- header: "{octicon}`codescan-checkmark` Overview"
  content: "Learn the benefits of using OLIVE to optimize your models."
- header: "{octicon}`zap` Get Started"
  content: "Install `olive-ai` with `pip` and get up and running with OLIVE in minutes."
- header: "{octicon}`rocket` How To"
  content: "Find more details on specific OLIVE capabilities, such as quantization, running workflows on remote compute, model packaging, conversions, and more!"
- header: "{fas}`code`  API Reference"
  content: "Get more details on specific OLIVE capabilities, such as running workflows on remote compute (for example, Azure AI), model packaging, conversions, and more!"
- header: "{octicon}`diff-added`  Extending OLIVE"
  content: "Learn about the design of OLIVE and how to extend OLIVE with your own optimization methods."
```


```{toctree}
:maxdepth: 2
:hidden:

why-olive.md
getting-started/getting-started.md
how-to/index
examples.md
reference/index
extending/index
```