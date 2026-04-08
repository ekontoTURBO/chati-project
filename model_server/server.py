"""
Ollama Model Server Launcher for Gemma 4 E2B (Multimodal)
=========================================================
Run inside WSL2. Starts Ollama and ensures the Gemma 4 E2B
model is loaded and ready to serve.

Usage:
    python server.py
    # OR just run directly:
    ollama serve
"""

import subprocess
import sys
import yaml
import argparse
import time
import urllib.request
import json
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load server configuration from YAML file.

    Args:
        config_path: Absolute or relative path to config.yaml

    Returns:
        Dictionary of configuration values
    """
    path = Path(config_path)
    if not path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def wait_for_ollama(host: str, port: int, timeout: int = 30) -> bool:
    """Wait for Ollama server to become ready.

    Args:
        host: Server host
        port: Server port
        timeout: Max seconds to wait

    Returns:
        True if server is ready, False if timed out
    """
    url = f"http://{host}:{port}/api/tags"
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(1)
    return False


def ensure_model_loaded(host: str, port: int, model_id: str) -> None:
    """Check if the model is available and pull it if needed.

    Args:
        host: Ollama server host
        port: Ollama server port
        model_id: Model tag to check/pull
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if model_id in models:
                print(f"  Model '{model_id}' is available.")
                return
    except Exception:
        pass

    print(f"  Model '{model_id}' not found. Pulling...")
    subprocess.run(["ollama", "pull", model_id], check=True)


def main():
    """Entry point — parse args, load config, launch Ollama."""
    parser = argparse.ArgumentParser(
        description="Launch Ollama server for Gemma 4 E2B"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = config["model"]["id"]
    host = config["server"]["host"]
    port = config["server"]["port"]

    print("=" * 60)
    print(" VRChat AI Agent — Ollama Model Server")
    print("=" * 60)
    print(f"  Model:  {model_id}")
    print(f"  Host:   {host}")
    print(f"  Port:   {port}")
    print(f"  API:    http://{host}:{port}/v1/")
    print("=" * 60)
    print()

    # Set host binding via environment variable
    import os
    os.environ["OLLAMA_HOST"] = f"{host}:{port}"

    try:
        # Launch Ollama as a subprocess, streaming output
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        # Wait for server to be ready
        print("Waiting for Ollama to start...")
        if wait_for_ollama("127.0.0.1", port):
            print("Ollama is ready!")
            ensure_model_loaded("127.0.0.1", port, model_id)
            print()
            print(f"  Server running at http://{host}:{port}")
            print(f"  OpenAI API at    http://{host}:{port}/v1/")
            print("  Press Ctrl+C to stop.")
            print()
        else:
            print("WARNING: Ollama may not be fully ready yet.")

        process.wait()

    except FileNotFoundError:
        print("\nERROR: 'ollama' command not found.")
        print("Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        process.terminate()
    except subprocess.CalledProcessError as e:
        print(f"\nServer exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
