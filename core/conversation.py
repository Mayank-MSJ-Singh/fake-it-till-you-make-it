from typing import Literal


# The three possible states of a conversation
ConversationState = Literal["waiting_for_user", "user_speaking", "bot_speaking"]


class Conversation:
    """Manages chat history and tracks who is currently speaking.
    
    State machine mirrors Unmute's Chatbot exactly:
        waiting_for_user → (user says word)   → user_speaking
        user_speaking    → (pause detected)   → bot_speaking  
        bot_speaking     → (bot finishes)     → waiting_for_user
        bot_speaking     → (user interrupts)  → user_speaking
    """

    def __init__(self):
        # Chat history in OpenAI format: [{"role": "user"/"assistant", "content": "..."}]
        self.chat_history: list[dict[str, str]] = []

    def conversation_state(self) -> ConversationState:
        """Determine current state based on who spoke last.
        
        Mirrors Unmute's Chatbot.conversation_state() exactly:
          - assistant → "bot_speaking"
          - user with non-empty content → "user_speaking"
          - user with empty content "" → "waiting_for_user"
          - no history → "waiting_for_user"
        """
        if not self.chat_history:
            return "waiting_for_user"

        last_message = self.chat_history[-1]

        if last_message["role"] == "assistant":
            return "bot_speaking"
        elif last_message["role"] == "user":
            if last_message["content"].strip() != "":
                return "user_speaking"
            else:
                return "waiting_for_user"
        else:
            return "waiting_for_user"

    def add_message_delta(self, text: str, role: str) -> bool:
        """Append text to the current message, or start a new one.
        
        Mirrors Unmute's Chatbot.add_chat_message_delta():
          - If last message is a different role → start new message → return True
          - If last message is same role and was empty "" → append → return True 
          - If last message is same role and had content → append → return False
        
        The empty-message pattern is critical: after bot finishes, we add
        {"role": "user", "content": ""} to go to waiting_for_user state.
        When the first real word arrives, it appends to that empty message
        and returns True (is_new_message), which triggers the EMA reset.
        """
        if not self.chat_history or self.chat_history[-1]["role"] != role:
            self.chat_history.append({"role": role, "content": text})
            return True  # New message started!
        else:
            last_content = self.chat_history[-1]["content"]

            # Smart spacing (from Unmute's Chatbot)
            needs_space_left = last_content != "" and not last_content[-1].isspace()
            needs_space_right = text != "" and not text[0].isspace()
            if needs_space_left and needs_space_right:
                text = " " + text

            self.chat_history[-1]["content"] += text
            return last_content == ""  # True if message was empty → "new message"

    def mark_interruption(self):
        """Add '—' to the last assistant message to show it was interrupted."""
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] += "—"

    def get_last_user_text(self) -> str:
        """Get the most recent user message text."""
        for msg in reversed(self.chat_history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def preprocessed_messages(self) -> list[dict[str, str]]:
        """Clean up chat history for sending to the LLM.
        
        - Merges consecutive messages from the same role
        - Strips whitespace
        - Removes empty messages
        """
        if not self.chat_history:
            return []

        cleaned = []
        for msg in self.chat_history:
            content = msg["content"].strip()
            if not content:
                continue

            # Merge with previous if same role
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += " " + content
            else:
                cleaned.append({"role": msg["role"], "content": content})

        return cleaned
