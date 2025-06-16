"""
Utilities for detecting and managing Python environments
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def get_conda_envs() -> List[Dict[str, str]]:
    """Get list of available conda environments"""
    envs = []
    try:
        # Try to find conda in various locations
        conda_exe = None
        conda_paths = []
        
        # Common conda installation paths on Windows
        if os.name == "nt":
            user_home = os.path.expanduser("~")
            conda_paths.extend([
                "conda", "conda.bat", "conda.exe",
                os.path.join(user_home, "Anaconda3", "Scripts", "conda.exe"),
                os.path.join(user_home, "anaconda3", "Scripts", "conda.exe"),
                os.path.join(user_home, "Miniconda3", "Scripts", "conda.exe"),
                os.path.join(user_home, "miniconda3", "Scripts", "conda.exe"),
                "C:\\Anaconda3\\Scripts\\conda.exe",
                "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe",
                "C:\\ProgramData\\Miniconda3\\Scripts\\conda.exe",
                "C:\\tools\\Anaconda3\\Scripts\\conda.exe",
                "C:\\tools\\Miniconda3\\Scripts\\conda.exe"
            ])
        else:
            conda_paths.extend(["conda", "conda.sh"])
        
        # Also check PATH environment variable
        path_dirs = os.environ.get("PATH", "").split(os.pathsep)
        for path_dir in path_dirs:
            if "conda" in path_dir.lower():
                conda_paths.append(os.path.join(path_dir, "conda"))
                if os.name == "nt":
                    conda_paths.append(os.path.join(path_dir, "conda.exe"))
                    conda_paths.append(os.path.join(path_dir, "conda.bat"))
        
        for cmd in conda_paths:
            try:
                if os.path.exists(cmd) or not os.path.isabs(cmd):
                    result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        conda_exe = cmd
                        logger.info(f"Found conda at: {cmd}")
                        break
            except:
                continue
        
        if not conda_exe:
            logger.warning("Could not find conda executable")
            return envs
        
        # Get conda environments
        result = subprocess.run([conda_exe, "env", "list", "--json"], capture_output=True, text=True)
        if result.returncode == 0:
            env_data = json.loads(result.stdout)
            for env_path in env_data.get("envs", []):
                env_name = Path(env_path).name
                if env_name == "base":
                    env_name = "conda (base)"
                else:
                    env_name = f"conda ({env_name})"
                
                # Check if olive is installed
                python_exe = Path(env_path) / ("Scripts" if os.name == "nt" else "bin") / "python"
                if python_exe.exists():
                    has_olive = check_olive_installed(str(python_exe))
                    envs.append({
                        "name": env_name,
                        "path": env_path,
                        "python": str(python_exe),
                        "type": "conda",
                        "has_olive": has_olive
                    })
    except Exception as e:
        logger.warning(f"Failed to get conda environments: {e}")
    
    return envs

def get_virtualenvs() -> List[Dict[str, str]]:
    """Get list of virtual environments in common locations"""
    envs = []
    
    # Common virtual environment locations
    search_paths = [
        Path.home() / "envs",
        Path.home() / ".virtualenvs",
        Path.cwd() / "venv",
        Path.cwd() / "env",
        Path.cwd() / ".venv"
    ]
    
    # Also check parent directories for virtual environments
    current = Path.cwd()
    for _ in range(3):  # Check up to 3 parent levels
        parent = current.parent
        if parent != current:
            search_paths.extend([
                parent / "venv",
                parent / "env",
                parent / ".venv"
            ])
            current = parent
    
    seen_paths = set()
    for search_path in search_paths:
        if search_path.exists() and search_path.is_dir() and str(search_path) not in seen_paths:
            seen_paths.add(str(search_path))
            
            # Check if this directory itself is a virtual environment
            python_exe = search_path / ("Scripts" if os.name == "nt" else "bin") / "python"
            if python_exe.exists():
                has_olive = check_olive_installed(str(python_exe))
                version = get_python_version(str(python_exe))
                envs.append({
                    "name": f"venv ({search_path.name}) - Python {version}",
                    "path": str(search_path),
                    "python": str(python_exe),
                    "type": "virtualenv",
                    "has_olive": has_olive
                })
            else:
                # Check subdirectories
                for env_dir in search_path.iterdir():
                    if env_dir.is_dir():
                        python_exe = env_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
                        if python_exe.exists():
                            has_olive = check_olive_installed(str(python_exe))
                            version = get_python_version(str(python_exe))
                            envs.append({
                                "name": f"venv ({env_dir.name}) - Python {version}",
                                "path": str(env_dir),
                                "python": str(python_exe),
                                "type": "virtualenv",
                                "has_olive": has_olive
                            })
    
    return envs

def get_system_python() -> List[Dict[str, str]]:
    """Get system Python installation"""
    envs = []
    
    # Current Python
    has_olive = check_olive_installed(sys.executable)
    envs.append({
        "name": f"System Python {sys.version.split()[0]}",
        "path": os.path.dirname(sys.executable),
        "python": sys.executable,
        "type": "system",
        "has_olive": has_olive,
        "current": True
    })
    
    # Try to find other Python installations
    if os.name == "nt":
        # Windows: Check common locations
        common_paths = [
            Path("C:/Python39"),
            Path("C:/Python310"),
            Path("C:/Python311"),
            Path("C:/Python312"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Python"
        ]
        
        for base_path in common_paths:
            if base_path.exists():
                for python_dir in base_path.glob("Python*"):
                    python_exe = python_dir / "python.exe"
                    if python_exe.exists() and str(python_exe) != sys.executable:
                        has_olive = check_olive_installed(str(python_exe))
                        version = get_python_version(str(python_exe))
                        envs.append({
                            "name": f"System Python {version}",
                            "path": str(python_dir),
                            "python": str(python_exe),
                            "type": "system",
                            "has_olive": has_olive
                        })
    
    return envs

def check_olive_installed(python_exe: str) -> bool:
    """Check if olive is installed in the given Python environment"""
    try:
        result = subprocess.run(
            [python_exe, "-c", "import olive; print('yes')"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0 and "yes" in result.stdout
    except:
        return False

def get_python_version(python_exe: str) -> str:
    """Get Python version for the given executable"""
    try:
        result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Extract version from "Python 3.9.7"
            return result.stdout.strip().split()[1]
    except:
        pass
    return "Unknown"

def get_all_environments() -> List[Dict[str, str]]:
    """Get all available Python environments"""
    all_envs = []
    
    # Get conda environments
    all_envs.extend(get_conda_envs())
    
    # Get virtual environments
    all_envs.extend(get_virtualenvs())
    
    # Get system Python
    all_envs.extend(get_system_python())
    
    # Sort by whether they have olive installed
    all_envs.sort(key=lambda x: (not x.get("has_olive", False), x["name"]))
    
    return all_envs

def run_in_environment(python_exe: str, code: str, env_vars: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    """Run Python code in a specific environment"""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    try:
        result = subprocess.run(
            [python_exe, "-c", code],
            capture_output=True,
            text=True,
            env=env
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)