"""Interactive CLI command interface for the real-time transcriber.

Provides keyboard-driven control while the application is running:
  r — toggle recording (pause/resume audio capture)
  s — show status (uptime, chunks, recording state)
  q — quit (graceful shutdown)
  h — show this help

Uses raw terminal mode on Unix for instant, single-character input
without requiring Enter. The terminal is restored on exit.
"""

import asyncio
import logging
import sys
import termios
import tty
from collections.abc import Callable

logger = logging.getLogger("realtime-transcriber.cli")


class CommandInterface:
    """Reads keyboard input in raw mode and dispatches commands.

    Must be used as an async context manager to ensure terminal
    state is properly restored.

    Usage:
        async with CommandInterface(handler) as cli:
            await cli.run()
    """

    def __init__(
        self,
        on_toggle_recording: Callable[[], bool | None],
        on_show_status: Callable[[], str],
        on_quit: Callable[[], None],
    ) -> None:
        """Initialize the command interface.

        Args:
            on_toggle_recording: Called when 'r' is pressed.
                May return the new recording state (True/False) or None
                if toggle failed.
            on_show_status: Called when 's' is pressed.
                Returns a status string to display.
            on_quit: Called when 'q' is pressed to trigger shutdown.
        """
        self._on_toggle = on_toggle_recording
        self._on_status = on_show_status
        self._on_quit = on_quit
        self._running: bool = False
        self._original_terminal: list | None = None
        self._stdin_fd: int = sys.stdin.fileno()

    async def run(self) -> None:
        """Start reading keyboard input in a background task.

        Blocks until _running is set to False (by quit command).
        """
        self._running = True
        self._enter_raw_mode()
        transport = None
        try:
            reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(reader)
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, sys.stdin
            )

            self._print_help()

            while self._running:
                try:
                    char = await asyncio.wait_for(reader.read(1), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                if not char:
                    break

                await self._dispatch(char)

        finally:
            if transport is not None:
                transport.close()
            self._exit_raw_mode()

    async def _dispatch(self, char: bytes) -> None:
        """Dispatch a single character to the appropriate handler.

        Args:
            char: Single byte read from stdin.
        """
        key = char.decode("utf-8", errors="replace").lower()

        if key == "r":
            self._handle_toggle()
        elif key == "s":
            self._handle_status()
        elif key == "q":
            self._handle_quit()
        elif key == "h":
            self._print_help()
        elif key == "\x03":  # Ctrl+C
            self._handle_quit()
        # Ignore other keys silently

    def _handle_toggle(self) -> None:
        """Handle the recording toggle command."""
        result = self._on_toggle()
        if result is True:
            print("\r\033[K🎤  Recording RESUMED", flush=True)
        elif result is False:
            print("\r\033[K⏸  Recording PAUSED", flush=True)
        else:
            print("\r\033[K⚠️  Toggle failed", flush=True)

    def _handle_status(self) -> None:
        """Handle the status display command."""
        status_text = self._on_status()
        print("\n" + status_text + "\n", flush=True)

    def _handle_quit(self) -> None:
        """Handle the quit command."""
        print("\r\033[K🛑  Shutting down...", flush=True)
        self._running = False
        self._on_quit()

    def _enter_raw_mode(self) -> None:
        """Put the terminal in raw mode for single-char input."""
        if not sys.stdin.isatty():
            return

        try:
            self._original_terminal = termios.tcgetattr(self._stdin_fd)
            tty.setraw(self._stdin_fd)
        except (termios.error, OSError) as exc:
            logger.debug("Cannot set raw terminal mode: %s", exc)
            self._original_terminal = None

    def _exit_raw_mode(self) -> None:
        """Restore the original terminal settings."""
        if self._original_terminal is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._original_terminal)
            except (termios.error, OSError) as exc:
                logger.debug("Failed to restore terminal: %s", exc)

    @staticmethod
    def _print_help() -> None:
        """Print the command help banner."""
        print(
            "\n┌──────────────────────────────────────────┐\n"
            "│  Commands:                               │\n"
            "│    r  —  Toggle recording (pause/resume) │\n"
            "│    s  —  Show status                     │\n"
            "│    q  —  Quit (graceful shutdown)        │\n"
            "│    h  —  Show this help                  │\n"
            "└──────────────────────────────────────────┘\n",
            flush=True,
        )
