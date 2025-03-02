import time
import requests
import subprocess


def StartNgrok(port=5000):
    # Automatically start the ngrok service and obtain the public URL
    ngrok_process_result = subprocess.Popen(["ngrok", "http", str(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(3)
    try:
        response = requests.get("http://127.0.0.1:4040/api/tunnels")
        public_url = response.json()["tunnels"][0]["public_url"]
        print(f">>> Ngrok public URL: {public_url}")
        return public_url, ngrok_process_result
    except Exception as e:
        print(f">>> Ngrok exception: {e}")
        return None, ngrok_process_result
