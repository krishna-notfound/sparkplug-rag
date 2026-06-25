from rich.console import Console
from rich.panel import Panel


class Logger:
    def __init__(self):
        self.console = Console()

    def info(self, message):
        self.console.print(f"[bold cyan][INFO][/bold cyan] {message}")

    def success(self, message):
        self.console.print(f"[bold green][SUCCESS][/bold green] {message}")

    def warning(self, message):
        self.console.print(f"[bold yellow][WARNING][/bold yellow] {message}")

    def error(self, message):
        self.console.print(f"[bold red][ERROR][/bold red] {message}")

    def debug(self, message):
        self.console.print(f"[bold magenta][DEBUG][/bold magenta] {message}")

    def critical(self, message):
        self.console.print(f"[bold white on red][CRITICAL][/bold white on red] {message}")

log = Logger()