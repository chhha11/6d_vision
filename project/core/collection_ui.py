"""Chinese capture instructions in a separate panel; source pixels stay intact."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=1)
def chinese_font():
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        'C:/Windows/Fonts/msyh.ttc',
    ]
    for filename in candidates:
        if Path(filename).is_file():
            return ImageFont.truetype(filename, 20)
    raise RuntimeError('缺少中文字体，请安装 Noto Sans CJK 后重新启动采集。')


@lru_cache(maxsize=64)
def instruction_panel(width, title, lines):
    font = chinese_font()
    probe = ImageDraw.Draw(Image.new('RGB', (width, 1)))
    wrapped = []
    for line in (title, *lines):
        current = ''
        for char in line:
            if current and probe.textlength(current + char, font=font) > width - 24:
                wrapped.append(current)
                current = ''
            current += char
        wrapped.append(current)
    # Fixed minimum height avoids resizing the camera viewport at each step.
    panel = Image.new('RGB', (width, max(174, 24 + 29 * len(wrapped))), (24, 30, 34))
    draw = ImageDraw.Draw(panel)
    for row, line in enumerate(wrapped):
        draw.text((12, 10 + 29 * row), line, font=font,
                  fill=(255, 222, 100) if row == 0 else (238, 242, 246))
    return cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)


def instructions(frame, title, *lines):
    return np.vstack((frame, instruction_panel(frame.shape[1], title, tuple(lines))))


def progress_preview(frame, message):
    if message.startswith('Stopping'):
        return instructions(frame, '正在安全停止，请稍候', '后台任务结束后将保存退出，请勿拔掉相机。')
    if 'Capturing reference' in message:
        return instructions(frame, '第 3 步：正在采集深度',
                            '相机和目标都不要动；手不要遮挡目标。',
                            '此时显示的是采集前画面，短暂停住是正常现象。', '完成后自动进入检查；按 Q 可安全退出。')
    if 'Checking CAD' in message:
        return instructions(frame, '第 4 步：正在检查样本',
                            '正在检查深度拟合、前后静止程度和参考特征。',
                            '请继续保持静止；通过后会显示轮廓供你确认。', '检查失败会显示中文原因，失败数据也会保存。')
    return instructions(frame, '正在恢复相机画面，请继续保持静止',
                        '深度采集后需要重新打开彩色相机，请稍候。', '按 Q 可安全退出。')


def rejection_help(error):
    text = str(error).lower()
    if 'stillness' in text or 'scene changed' in text or 'moved' in text:
        return '前后画面未通过静止检查。移开手、固定相机和目标后重拍。'
    if 'feature' in text or '12 usable' in text:
        return '可用参考特征不足。改善照明、避开反光或稍转一个角度。'
    if 'depth pose rejected' in text or 'geometry' in text:
        return '深度与模型拟合未通过。重新框住目标，调整距离或朝向。'
    if 'timeout' in text or 'worker' in text or 'depth failed' in text:
        return '深度采集失败或超时。检查相机连接，再保持静止重试。'
    return '本次检查未通过，失败数据已保存。可调整后重拍或跳过。'
