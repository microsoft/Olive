# Generating the documentation

To generate the documentation, you first have to build it.

## Pre-requisites

Install Olive. At the root of the code repository:

```bash
pip install -e .
```

Install pip requirements. At `docs`:

```bash
pip install -r requirements.txt
```

## Building the documentation

At `docs`:

```bash
make html
make linkcheck
```

## Previewing the documentation

At `docs/build/html`:

```bash
python -m http.server {port-number}
```

The documentation site will be running at `http://localhost:<port-number>`
