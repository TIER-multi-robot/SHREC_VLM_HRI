"""Local open-weight backend handler (vLLM OpenAI-compatible server).

Sends the identical SHREC prompt text to a vLLM server. For Condition B the
pre-rendered 1 Hz segment mp4 from data/videos_1hz/ is passed as a base64
data-URI video_url, so every model receives byte-identical video files.

No think/answer system-prompt format is used: the plain SHREC prompt is sent
as-is (human decision, CP0 review).

All sampling parameters are passed explicitly so the model's
generation_config defaults do not apply; identical for both Qwen-family
models: temperature=0 (greedy), top_p=1.0, top_k disabled (-1),
repetition_penalty=1.0, max_tokens per CLI arg (default 4096).
"""

import base64
import json
import os
import time
import urllib.request

from call_log import log_call

# Explicit sampling parameters shared by both Qwen-family models.
SAMPLING_DEFAULTS = {
    'temperature': 0,
    'top_p': 1.0,
    'top_k': -1,
    'repetition_penalty': 1.0,
}


class VLLMClient:
    def __init__(self, model_id, base_url='http://localhost:8000/v1',
                 precision='bf16', max_output_tokens=4096, call_logger=None,
                 log_path=None, run_name=''):
        self.model_id = model_id
        self.base_url = base_url.rstrip('/')
        self.precision = precision
        self.max_output_tokens = max_output_tokens
        self.call_logger = call_logger
        self.log_path = log_path
        self.run_name = run_name
        self.sampling_params = dict(SAMPLING_DEFAULTS)
        self.dump_dir = os.environ.get('VLM_DUMP_REQUESTS')

    def chat(self, message, img_frames=None, transcription=None, vid_path=None, examples=None):
        content = []
        truncated = False
        n_frames = 0

        if vid_path:
            with open(vid_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            content.append({
                'type': 'video_url',
                'video_url': {'url': 'data:video/mp4;base64,' + b64},
            })
            try:
                import cv2
                cap = cv2.VideoCapture(vid_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception:
                n_frames = None
        content.append({'type': 'text', 'text': message})

        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)
            _idx = getattr(self, '_dump_idx', 0)
            self._dump_idx = _idx + 1
            with open(os.path.join(self.dump_dir, 'vllm_request_{:03d}.json'.format(_idx)), 'w') as f:
                json.dump({'model': self.model_id, 'sampling_params': self.sampling_params,
                           'messages': [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': '<{} bytes of mp4 base64>'.format(len(b64) if vid_path else 0)}}, content[-1]]}]}, f, indent=2)

        payload = {
            'model': self.model_id,
            'messages': [{'role': 'user', 'content': content}],
            'max_tokens': self.max_output_tokens,
        }
        payload.update(self.sampling_params)
        start = time.time()
        error = None
        text = '[ERROR]'
        usage = {}
        try:
            req = urllib.request.Request(
                self.base_url + '/chat/completions',
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req) as resp:
                out = json.loads(resp.read())
            choice = out['choices'][0]
            text = choice['message']['content']
            truncated = choice.get('finish_reason') == 'length'
            usage = out.get('usage', {})
        except Exception as e:
            error = str(e)

        latency = time.time() - start
        if self.call_logger:
            self.call_logger(
                model_version=self.model_id,
                prompt_tokens=usage.get('prompt_tokens'),
                output_tokens=usage.get('completion_tokens'),
                latency_s=round(latency, 3),
                truncated=truncated,
                error=error,
                video_frames=n_frames,
                max_output_tokens=self.max_output_tokens,
                sampling_params=self.sampling_params,
            )
        elif self.log_path:
            log_call(self.log_path, {
                'run_name': self.run_name,
                'model': self.model_id,
                'model_version': self.model_id,
                'backend': 'vllm',
                'precision': self.precision,
                'prompt_tokens': usage.get('prompt_tokens'),
                'output_tokens': usage.get('completion_tokens'),
                'latency_s': round(latency, 3),
                'truncated': truncated,
                'error': error,
                'video_frames': n_frames,
                'max_output_tokens': self.max_output_tokens,
                'sampling_params': self.sampling_params,
            })
        return text
