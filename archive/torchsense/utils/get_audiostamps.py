import numpy as np
import torch
import webrtcvad


def frame_generator(frame_duration_ms, audio, sample_rate):
    """将音频数据分割成指定时长的帧并转换为16-bit PCM格式"""
    if isinstance(audio, torch.Tensor):
        if audio.is_cuda:
            audio = audio.cpu()
        audio = audio.numpy()
    if audio.dtype != np.int16:
        audio = audio.astype(np.int16)
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    for offset in range(0, len(audio) - n, n):
        yield audio[offset:offset + n].tobytes()


def get_audio_timestamps(mic, sample_rate: int = 16000, frame_duration_ms: int = 30, padding_duration_ms: int = 40):
    vad = webrtcvad.Vad(3)
    mic = mic.squeeze()
    frames = frame_generator(frame_duration_ms, mic, sample_rate)
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False
    voiced_frames = []
    time_stamps = []
    all_time_stamps = []
    frame_duration = frame_duration_ms / 1000.0  # 转换为秒
    current_time = 0.0  # 初始化时间戳

    for frame in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                time_stamps.append(current_time)  # 记录开始时间
                voiced_frames.extend(f for f, s in ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            if not is_speech:
                triggered = False
                time_stamps.append(current_time)  # 记录结束时间
                if time_stamps[1] - time_stamps[0] > 0.1:
                    all_time_stamps.append(time_stamps)
                voiced_frames = []
                time_stamps = []

        current_time += frame_duration

    if voiced_frames:
        time_stamps.append(current_time)  # 如果最后一帧是语音，记录结束时间
        all_time_stamps.append(time_stamps)
    return all_time_stamps
