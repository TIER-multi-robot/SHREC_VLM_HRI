import sys
sys.path.insert(0, '.')
from utils_vllm import VLLMClient

c = VLLMClient('nvidia/Cosmos-Reason2-2B', base_url='http://localhost:8000/v1')
logged = []
c.call_logger = lambda **kw: logged.append(kw)
resp = c.chat('Describe this video in one sentence.',
              vid_path='data/videos_1hz/P01_s009-001_60.913_61.707.mp4')
print('response:', resp[:100])
print('log record:', {k: logged[0][k] for k in ('video_frames', 'single_frame_duplicated', 'prompt_tokens', 'error')})
