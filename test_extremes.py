import base64, json, urllib.request, urllib.error, sys

with open('data/videos_1hz/P01_s009-001_60.913_61.707.mp4', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
payload = {
    'model': 'Qwen/Qwen3-VL-2B-Instruct',
    'messages': [{'role': 'user', 'content': [
        {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + b64}},
        {'type': 'text', 'text': 'Describe this video.'},
    ]}],
    'max_tokens': 32,
    'temperature': 0,
}
req = urllib.request.Request('http://localhost:8000/v1/chat/completions',
    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
except urllib.error.HTTPError as e:
    body = json.loads(e.read().decode())
    msg = body['error']['message']
    # print the tail of the message where the actual exception text lives
    print(msg[-1200:])
