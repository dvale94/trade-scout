import requests
from . import config

def discord_notify(text: str):
    if not config.DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set; printing message instead:\n", text)
        return
    resp = requests.post(config.DISCORD_WEBHOOK_URL, json={"content": text}, timeout=10)
    resp.raise_for_status()