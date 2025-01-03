import json
from pathlib import Path
import socket
import requests

WEBHOOK_URL_KEY = "discord_webhook"

current_dir = Path(__file__).parent
with open(current_dir / 'config.json', 'r') as f:
    config = json.load(f)
if WEBHOOK_URL_KEY not in config or config[WEBHOOK_URL_KEY] == "":
    print("No discord_webhook key found in config.json. Please add a Discord webhook URL.")
    config[WEBHOOK_URL_KEY] = None

def send(message: str):
    # get the hostname of the server
    hostname = socket.gethostname()
    if config[WEBHOOK_URL_KEY] is None:
        print("No webhook URL provided. Skipping sending message to Discord.")
        return
    payload = {
        "content": f"*[gym_atari@{hostname}]* {message}"
    }
    response = requests.post(config[WEBHOOK_URL_KEY], json=payload)
    response.raise_for_status()
