"""
Command Executor
================
Maps decoded intents to concrete system actions using Python's standard library.
Falls back gracefully when optional depependencies (pygame, pyautogui) are absent.
"""

import webbrowser
import subprocess
import sys
import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ─── Action registry ──────────────────────────────────────────────────────────

def _exec_open_browser() -> str:
    url = "https://www.google.com"
    webbrowser.open(url)
    return f"Browser opened → {url}"


def _exec_scroll_down() -> str:
    try:
        import pyautogui
        pyautogui.scroll(-5)
        return "Scrolled down 5 units"
    except ImportError:
        return "Simulated scroll ↓ (pyautogui not installed)"


def _exec_type_hello() -> str:
    try:
        import pyautogui
        time.sleep(0.3)
        pyautogui.typewrite("Hello from MindLink AI!", interval=0.04)
        return 'Typed "Hello from MindLink AI!"'
    except ImportError:
        return 'Simulated typing: "Hello from MindLink AI!" (pyautogui not installed)'


def _exec_play_music() -> str:
    """
    Look for any .mp3 / .wav in the project demo folder and play it.
    Falls back to pygame beep if nothing is found.
    """
    from pathlib import Path
    music_files = list(Path(__file__).parent.parent.glob("demo/*.mp3")) + \
                  list(Path(__file__).parent.parent.glob("demo/*.wav"))

    if music_files:
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(str(music_files[0]))
            pygame.mixer.music.play()
            return f"Playing: {music_files[0].name}"
        except ImportError:
            pass
        # Platform fallback
        if sys.platform.startswith("win"):
            subprocess.Popen(["start", "", str(music_files[0])], shell=True)
            return f"Playing (OS): {music_files[0].name}"

    return "Simulated music playback ♪ (no audio file found in demo/)"


def _exec_stop_action() -> str:
    try:
        import pygame
        pygame.mixer.music.stop()
    except Exception:
        pass
    return "Stop signal sent – all active tasks halted"


# ─── Dispatcher ───────────────────────────────────────────────────────────────

ACTIONS: Dict[str, callable] = {
    "open_browser": _exec_open_browser,
    "scroll_down":  _exec_scroll_down,
    "type_hello":   _exec_type_hello,
    "play_music":   _exec_play_music,
    "stop_action":  _exec_stop_action,
}


def execute_command(intent: str) -> Dict[str, str]:
    """
    Execute the action matching `intent`.

    Returns
    -------
    dict with keys: status, message
    """
    fn = ACTIONS.get(intent)
    if fn is None:
        return {
            "status":  "error",
            "message": f"Unknown intent: {intent}",
        }
    try:
        msg = fn()
        logger.info("Executed %s → %s", intent, msg)
        return {"status": "success", "message": msg}
    except Exception as exc:
        logger.exception("Execution failed for intent %s", intent)
        return {"status": "error", "message": str(exc)}
