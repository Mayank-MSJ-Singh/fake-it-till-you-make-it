from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.console import Group
from rich.align import Align
from rich.prompt import Prompt

class UIManager:
    """Manages the rich terminal user interface."""
    
    def __init__(self):
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=5)
        )
        
        self.state = "starting"
        self.ema = 0.0
        self.time_sec = 0.0
        self.threshold = 0.70 # Default, gets updated
        self.chat_history = []
        self.debug_line = ""
        self.interrupted = False
        
    def generate_header(self) -> Panel:
        status_text = Text()
        if self.state == "starting":
            status_text.append("Loading models...", style="bold yellow")
        elif self.state == "waiting_for_user":
            status_text.append("Listening...", style="bold green")
        elif self.state == "user_speaking":
            status_text.append("Recording...", style="bold red")
        elif self.state == "bot_speaking":
            status_text.append("Assistant is speaking...", style="bold blue")
            
        header_text = Text("🎤 Fake It Till You Make It", style="bold white", justify="center")
        header_text.append("\nStatus: ")
        header_text.append(status_text)
        
        return Panel(header_text, style="cyan")

    def generate_body(self) -> Panel:
        messages = []
        for msg in self.chat_history:
            role = msg.get("role", "system")
            content = msg.get("content", "").strip()
            if not content:
                continue
                
            if role == "user":
                messages.append(Text(f"[YOU] {content}", style="bold green"))
            elif role == "assistant":
                messages.append(Text(f"[BOT] {content}", style="bold blue"))
            else:
                messages.append(Text(f"[SYS] {content}", style="dim"))
                
        # If there's an active debug line (like "Interrupting bot..."), append it
        if self.debug_line:
            messages.append(Text(f"\n⚡ {self.debug_line}", style="bold yellow"))
            
        return Panel(
            Group(*messages),
            title="Conversation"
        )
        
    def generate_footer(self) -> Panel:
        # Build EMA Bar manually
        bar_length = 20
        filled = int(self.ema * bar_length)
        empty = bar_length - filled
        bar = ("█" * filled) + ("░" * empty)
        
        # Color based on threshold
        bar_color = "red" if self.ema > self.threshold else "green"
        bar_text = Text("pause_prediction: ", style="dim")
        bar_text.append(f"{bar} {self.ema:.2f}", style=bar_color)
        
        state_text = Text(f"state: {self.state} | time: {self.time_sec:.1f}s", style="bold white")
        
        return Panel(
            Group(bar_text, state_text),
            title="Metrics"
        )

    def get_renderable(self):
        """Returns the full rich Layout to be rendered."""
        self.layout["header"].update(self.generate_header())
        self.layout["body"].update(self.generate_body())
        self.layout["footer"].update(self.generate_footer())
        return self.layout

    def update(self, history, state, ema, time_sec, threshold=0.70, debug_line=""):
        """Update internal state so the next render frame shows correct info."""
        self.chat_history = history
        self.state = state
        self.ema = ema
        self.time_sec = time_sec
        self.threshold = threshold
        self.debug_line = debug_line
