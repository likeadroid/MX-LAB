import subprocess
import re

def get_keyframes(video_path):
    """Retourne la liste des timestamps des keyframes (I-frames) dans la vidéo via FFprobe."""
    print("[DEBUG] Détection des keyframes en cours...")
    cmd = [
        'ffprobe', '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=pict_type,pkt_pts_time', '-of', 'csv', video_path
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    timestamps = []
    for line in result.stdout.splitlines():
        if 'I' in line:
            m = re.search(r'(\d+\.\d+)', line)
            if m:
                timestamps.append(float(m.group(1)))
    print(f"[DEBUG] {len(timestamps)} keyframes détectées.")
    return timestamps

def adjust_to_keyframe(start, keyframes):
    """Ajuste le timestamp de début au keyframe précédent ou égal."""
    prev = 0.0
    for kf in keyframes:
        if kf > start:
            break
        prev = kf
    return prev
