import argparse
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--key", default="change-this-secret")
args = parser.parse_args()

with open(args.image, "rb") as f:
    r = requests.post(
        args.url.rstrip("/") + "/recognize",
        headers={"x-api-key": args.key},
        files={"file": (args.image, f, "image/jpeg")},
        timeout=120,
    )

print(r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))
