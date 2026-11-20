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
import socket
import time
import urllib.request

from call_log import log_call

# Request timeout: connect+read. A hung request must fail the SAMPLE as
# [ERROR] rather than stall the run for hours (b3 incident: four stalled runs).
REQUEST_TIMEOUT_S = 600
# Consecutive timeouts before the run is killed (SystemExit bypasses the
# broad `except Exception` in main_vlm_exp, so the run exits non-zero and
# gets redone by the orchestrator's skip-complete logic).
MAX_CONSECUTIVE_TIMEOUTS = 3

# Explicit sampling parameters shared by both Qwen-family models.
SAMPLING_DEFAULTS = {
    'temperature': 0,
    'top_p': 1.0,
    'top_k': -1,
    'repetition_penalty': 1.0,
}

# Reasoning-elicitation suffix for Cosmos-Reason2 models, appended as a
# system message. Source: RAW README.md of nvidia/Cosmos-Reason2-32B
# (https://huggingface.co/nvidia/Cosmos-Reason2-32B/raw/main/README.md),
# Input section, "Other Properties Related to Input". Literal card text:
# Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n\n<answer>\nyour answer\n</answer>.
# (README escape sequences interpreted as newlines below.)
# Reasoning parity decision (CP2e): matches reasoning behavior within
# the Cosmos/Qwen model pair; the SHREC task prompt itself is unchanged.
COSMOS_REASONING_SUFFIX = 'Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>.'


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
        single_frame_duplicated = False

        if vid_path:
            import cv2
            cap = cv2.VideoCapture(vid_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # Qwen3VL processor requires >= 2 frames (temporal_factor=2).
            # Human decision (CP2 review): duplicate the single frame at request
            # time; flagged in the sidecar; sensitivity analysis excludes these.
            if n_frames == 1:
                cap = cv2.VideoCapture(vid_path)
                ret, frame = cap.read()
                cap.release()
                h, w = frame.shape[:2]
                tmp_path = vid_path + '.2frame.mp4'
                writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, (w, h))
                writer.write(frame); writer.write(frame)
                writer.release()
                vid_path = tmp_path
                n_frames = 2
                single_frame_duplicated = True
        if vid_path:
            with open(vid_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            content.append({
                'type': 'video_url',
                'video_url': {'url': 'data:video/mp4;base64,' + b64},
            })
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
            'messages': [],
            'max_tokens': self.max_output_tokens,
        }
        if 'Cosmos' in self.model_id:
            payload['messages'].append(
                {'role': 'system', 'content': COSMOS_REASONING_SUFFIX})
        payload['messages'].append({'role': 'user', 'content': content})
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
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
                out = json.loads(resp.read())
            choice = out['choices'][0]
            text = choice['message']['content']
            truncated = choice.get('finish_reason') == 'length'
            usage = out.get('usage', {})
        except Exception as e:
            error = str(e)
            if isinstance(e, (socket.timeout, TimeoutError)) or 'timed out' in str(e):
                self._consecutive_timeouts = getattr(self, '_consecutive_timeouts', 0) + 1
                if self._consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                    # Kill the run (non-zero exit) so it is redone cleanly
                    # instead of hanging forever on a dead server.
                    raise SystemExit(
                        'vLLM request timed out {} times in a row (timeout={}s); '
                        'failing run {}'.format(self._consecutive_timeouts, REQUEST_TIMEOUT_S, self.run_name))
            else:
                self._consecutive_timeouts = 0
        else:
            self._consecutive_timeouts = 0

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
                single_frame_duplicated=single_frame_duplicated,
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
                'single_frame_duplicated': single_frame_duplicated,
                'max_output_tokens': self.max_output_tokens,
                'sampling_params': self.sampling_params,
            })
        return text
