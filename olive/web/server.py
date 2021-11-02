import sys
import subprocess
import os
from ..constants import PYTHON_PATH, INSTALLED_PACKAGES_DICT


def install_server_dependencies():
    install_cmd = "{} -m pip install pandas netron flask flask-cors redis celery[redis] flower".format(PYTHON_PATH)
    subprocess.run(install_cmd, stdout=subprocess.PIPE, shell=True, check=True)


def server_dependencies_installed():
    required_packages = ["pandas", "netron", "flask", "flask-cors", "redis", "celery", "flower"]
    for package in required_packages:
        if package not in INSTALLED_PACKAGES_DICT.keys():
            return False
    return True


def start_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.platform.startswith('win'):
        server_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'start_windows.bat')
    else:
        server_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'start_linux.sh')
    subprocess.call([server_file])


def stop_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.platform.startswith('win'):
        server_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_windows.bat')
    else:
        server_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stop_linux.sh')
    subprocess.call([server_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
