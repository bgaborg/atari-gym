import json
from pathlib import Path
import requests

current_dir = Path(__file__).parent
with open(current_dir / 'config.json', 'r') as f:
    config = json.load(f)
if "discord_webhook" not in config:
    print("No discord_webhook key found in config.json. Please add a Discord webhook URL.")
    config["discord_webhook"] = None

def send(self, message: str):
    if self.webhook_url is None:
        print("No webhook URL provided. Skipping sending message to Discord.")
        return
    payload = {
        "content": f"*[gym_atari]* {message}"
    }
    response = requests.post(self.webhook_url, json=payload)
    response.raise_for_status()
