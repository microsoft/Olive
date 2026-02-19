# -------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT
# --------------------------------------------------------------------------

import json
import logging
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Union

from olive.common.config_utils import ParamCategory
from olive.common.utils import hardlink_copy_file
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, QairtPreparedModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtPreparation(Pass):
    """Prepare a HuggingFace model for QAIRT by running an external preparation script.

    This pass executes a Python script that performs quantization and other preparation
    steps to convert a HuggingFace model into a QAIRT-compatible format. The script
    receives configuration via a JSON file and produces a QairtPreparedModelHandler.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "script_path": PassConfigParam(
                type_=str,
                required=True,
                category=ParamCategory.PATH,
                description="Path to the Python script that performs QAIRT preparation. "
                "The script should accept a --config argument pointing to a JSON configuration file.",
            ),
            "script_config": PassConfigParam(
                type_=dict,
                required=False,
                default_value={},
                description="Configuration dictionary to pass to the preparation script. "
                "This will be merged with input_model_path and output_model_path in the JSON config file. "
                "Example: {'precision': 'int8', 'calibration_samples': 100, 'backend': 'HTP'}",
            ),
            "cache_dir": PassConfigParam(
                type_=str,
                required=False,
                default_value="./cache/qairt/preparation",
                description="Directory to be used as the cache directory for subsequent QairtPreparation invocations."
                "By default, saves to a similar location to the Olive cache.",
            )
        }

    def _run_for_config(
        self,
        model: HfModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> QairtPreparedModelHandler:
        """Execute the preparation script to convert HfModelHandler to QairtPreparedModelHandler.

        Args:
            model: Input HfModelHandler
            config: Pass configuration
            output_model_path: Path where the prepared model should be saved

        Returns:
            QairtPreparedModelHandler pointing to the prepared model

        Raises:
            ValueError: If input model is not HfModelHandler or script path is invalid
            RuntimeError: If script execution fails
        """
        # Validate input model type
        if not isinstance(model, HfModelHandler):
            raise ValueError(
                f"QairtPreparation requires HfModelHandler as input, got {type(model).__name__}"
            )

        # Resolve and validate script path
        script_path = Path(config.script_path).resolve()
        if not script_path.exists():
            raise ValueError(f"Preparation script not found at: {script_path}")
        if not script_path.suffix == ".py":
            raise ValueError(f"Script must be a Python file (.py), got: {script_path}")

        # Prepare configuration for the script
        cache_dir_path = Path(config.cache_dir).resolve()
        script_config = {
            "CACHE_DIR": str(cache_dir_path),
            "OUTPUT_DIR": str(output_model_path)
        }
        
        # Merge user-provided config
        if config.script_config:
            script_config.update(config.script_config)

        # Create temporary JSON config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="olive_qairt_prep_"
        ) as config_file:
            logging.error(script_config)
            json.dump(script_config, config_file, indent=2)
            config_file_path = config_file.name

        try:
            logger.info("Executing script %s", script_path)
            logger.debug("Script configuration: %s", script_config)

            # Helper function to read from pipe in a separate thread
            def enqueue_output(pipe, queue):
                """Read lines from pipe and put them in queue."""
                for line in iter(pipe.readline, ''):
                    queue.put(line)
                pipe.close()

            # Execute the preparation script with streaming output
            process = subprocess.Popen(
                ["python", str(script_path), "--config", config_file_path],
                cwd=str(script_path.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
            )

            # Create queues for stdout and stderr
            stdout_queue = Queue()
            stderr_queue = Queue()

            # Start threads to read from stdout and stderr concurrently
            stdout_thread = threading.Thread(target=enqueue_output, args=(process.stdout, stdout_queue))
            stderr_thread = threading.Thread(target=enqueue_output, args=(process.stderr, stderr_queue))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # Collect output for error reporting
            stdout_lines = []
            stderr_lines = []

            # Stream stdout and stderr in real-time from queues
            while process.poll() is None or not stdout_queue.empty() or not stderr_queue.empty():
                # Try to read from stdout queue
                try:
                    stdout_line = stdout_queue.get_nowait()
                    line = stdout_line.rstrip()
                    logger.info(line)
                    stdout_lines.append(line)
                except Empty:
                    pass
                
                # Try to read from stderr queue
                try:
                    stderr_line = stderr_queue.get_nowait()
                    line = stderr_line.rstrip()
                    logger.debug(line)
                    stderr_lines.append(line)
                except Empty:
                    pass

            # Wait for process to complete and get return code
            returncode = process.wait()

            # Check for errors
            if returncode != 0:
                stdout_text = "\n".join(stdout_lines)
                stderr_text = "\n".join(stderr_lines)
                error_msg = (
                    f"QAIRT preparation script failed with exit code {returncode}.\n"
                    f"Script: {script_path}\n"
                    f"Working directory: {script_path.parent}\n"
                    f"Stdout: {stdout_text}\n"
                    f"Stderr: {stderr_text}"
                )
                raise RuntimeError(error_msg)

            logger.info("QAIRT preparation script completed successfully")

        finally:
            # Clean up temporary config file
            try:
                Path(config_file_path).unlink()
            except Exception as e:
                logger.warning("Failed to delete temporary config file %s: %s", config_file_path, e)

        return QairtPreparedModelHandler(model_path=output_model_path)
