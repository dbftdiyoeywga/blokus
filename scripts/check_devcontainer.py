#!/usr/bin/env python3
"""
Script to check if the current environment is running inside the devcontainer.
This helps ensure that development is happening in the correct environment.
"""

import os
import platform
import subprocess
import sys


def is_in_devcontainer():
    """Check if we're running inside a devcontainer."""
    # Check for environment variables that are typically set in devcontainers
    if os.environ.get("REMOTE_CONTAINERS") == "true":
        return True

    if os.environ.get("REMOTE_CONTAINERS_IPC"):
        return True

    if os.environ.get("VSCODE_REMOTE_CONTAINERS_SESSION"):
        return True

    # Check for .devcontainer directory in parent directories
    current_dir = os.path.abspath(os.getcwd())
    while current_dir != os.path.dirname(current_dir):  # Stop at root
        if os.path.exists(os.path.join(current_dir, ".devcontainer")):
            # We found a .devcontainer directory, but we're still not sure
            # if we're actually running inside it
            break
        current_dir = os.path.dirname(current_dir)

    # Check Python version
    if sys.version_info.major != 3 or sys.version_info.minor != 12:
        return False

    # Try to check for uv
    try:
        result = subprocess.run(["uv", "--version"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False

    # Try to check for ruff
    try:
        result = subprocess.run(["ruff", "--version"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False

    # If we've made it this far, we're probably in the devcontainer
    return True


def main():
    """Main function."""
    if is_in_devcontainer():
        print("✅ Running in devcontainer environment!")
        print(f"Python version: {platform.python_version()}")
        return 0
    else:
        print("❌ NOT running in devcontainer environment!")
        print("Please open this project in a devcontainer.")
        print(f"Current Python version: {platform.python_version()} (should be 3.12.x)")
        print("See README.md for instructions on setting up the devcontainer.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
