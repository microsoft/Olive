from olive.workflows import run as olive_run

# config = "./cpu_config.json"
config = "./gpu_config.json"
# config = "./aml_cpu_config.json"
rls = olive_run(config)
print(rls)
