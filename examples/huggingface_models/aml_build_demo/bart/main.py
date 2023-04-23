from olive.workflows import run as run_workflow

run_workflow("./cpu_config.json", clean_cache=True, system="aml_system")
