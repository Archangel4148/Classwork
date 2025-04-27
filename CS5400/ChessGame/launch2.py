import subprocess

subprocess.run([
    "python", "JoueurPython/main.py", "chess",
    "-s", "mst-ai.xyz",
    "-r", "simon",
    "--index", "1"
])