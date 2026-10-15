from __future__ import annotations
import sys, queue, threading, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exit_monitor import ExitMonitor
from keyboard_monitor import KeyboardMonitor, keyboard
from config import Config
from main import _window_closed
import cv2

class ExitTests(unittest.TestCase):
    def make_monitor(self):
        stop=threading.Event();kbd=KeyboardMonitor(queue.Queue(),stop,Config())
        return stop,kbd,ExitMonitor(stop,kbd)
    def test_escape_press_without_release_stops(self):
        stop,kbd,_=self.make_monitor();kbd._on_press(keyboard.Key.esc)
        self.assertTrue(stop.is_set())
    def test_poll_ignores_stale_hook_modifier(self):
        stop,kbd,monitor=self.make_monitor();kbd.pressed.add(keyboard.Key.ctrl_l)
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertTrue(stop.is_set())
    def test_poll_survives_hook_failure(self):
        stop,kbd,monitor=self.make_monitor();kbd.hook_failed=True
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertTrue(stop.is_set())
    def test_modified_escape_not_exit_after_modifiers_released_while_held(self):
        stop,kbd,monitor=self.make_monitor()
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk in [0x1B,0x11,0x10]):monitor.poll()
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertFalse(stop.is_set())
        with patch('exit_monitor.key_down',return_value=False):monitor.poll()
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertTrue(stop.is_set())
    def test_poll_preserves_hook_modified_escape(self):
        stop,kbd,monitor=self.make_monitor();kbd.escape_was_modified=True;kbd.escape_generation=1
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertFalse(stop.is_set())
    def test_no_key_does_not_exit(self):
        stop,_,monitor=self.make_monitor()
        with patch('exit_monitor.key_down',return_value=False):monitor.poll()
        self.assertFalse(stop.is_set())
    def test_missed_modified_release_recovers(self):
        stop,kbd,monitor=self.make_monitor();kbd.escape_was_modified=True;kbd.escape_generation=1
        with patch('exit_monitor.key_down',return_value=False):monitor.poll()
        with patch('exit_monitor.key_down',side_effect=lambda vk:vk==0x1B):monitor.poll()
        self.assertTrue(stop.is_set())
    def test_physical_release_before_callback_keeps_chord_guard(self):
        stop,kbd,monitor=self.make_monitor()
        for key in [keyboard.Key.ctrl_l,keyboard.Key.shift,keyboard.Key.esc]:kbd._on_press(key)
        with patch('exit_monitor.key_down',return_value=False):monitor.poll()
        for key in [keyboard.Key.ctrl_l,keyboard.Key.shift,keyboard.Key.esc]:kbd._on_release(key)
        self.assertFalse(stop.is_set())
    def test_window_x_detected(self):
        with patch('main.cv2.getWindowProperty',return_value=0):self.assertTrue(_window_closed())
        with patch('main.cv2.getWindowProperty',return_value=1):self.assertFalse(_window_closed())
    def test_destroyed_window_detected(self):
        with patch('main.cv2.getWindowProperty',side_effect=cv2.error('closed')):self.assertTrue(_window_closed())
    def test_hook_start_failure_is_logged(self):
        stop,kbd,_=self.make_monitor()
        listener=SimpleNamespace(start=lambda:(_ for _ in ()).throw(RuntimeError('start')),stop=lambda:None,ident=None)
        with patch('keyboard_monitor.keyboard.Listener',return_value=listener):kbd._run()
        self.assertTrue(kbd.hook_failed)
        self.assertEqual(kbd.output_queue.get_nowait().kind,'KEYBOARD_HOOK_FAILED')

if __name__=='__main__':unittest.main()
