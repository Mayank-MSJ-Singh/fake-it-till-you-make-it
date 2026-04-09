"""
Conversation — Chat History & State Machine
=============================================

Manages the conversation as a list of messages in OpenAI format:
  [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]

The STATE of the conversation is determined entirely by the LAST message:
  - Last message is assistant         → "bot_speaking"
  - Last message is user with text    → "user_speaking"
  - Last message is user with ""      → "waiting_for_user"
  - No messages at all                → "waiting_for_user"

The Empty Message Trick (from Unmute):
  After the bot finishes responding, we add {"role": "user", "content": ""}.
  This empty message makes the state "waiting_for_user" — pause detection
  is disabled. When the user says their first word, it appends to this
  empty message, and add_message_delta() returns True (is_new_message).
  The orchestrator uses that signal to reset the EMA to 0.0.

This mirrors Unmute's Chatbot class: unmute/llm/chatbot.py
"""
from typing import Literal


# The three possible conversation states
# These are checked by the orchestrator to decide what to do each frame
ConversationState = Literal["waiting_for_user", "user_speaking", "bot_speaking"]


class Conversation:
    """Manages chat history and tracks who is currently speaking.

    State machine (mirrors Unmute's Chatbot exactly):

        waiting_for_user ──(first word)──→ user_speaking
              ↑                                  │
              │                          (pause detected)
              │                                  │
              │                                  ▼
        (bot finishes) ←──────────────── bot_speaking
              │                                  ↑
              └──────(user interrupts)───────────┘
    """

    def __init__(self):
        # Chat history — list of {"role": str, "content": str}
        # This is the same format OpenAI's API uses, so it's ready
        # to send to any LLM when we add that in Phase 5.
        self.chat_history: list[dict[str, str]] = []

    def conversation_state(self) -> ConversationState:
        """What state is the conversation in right now?

        This is called every frame by the orchestrator to decide
        whether to check for pauses, silences, or interruptions.

        Returns:
            "waiting_for_user" — bot finished, waiting for user to start
            "user_speaking"    — user is in the middle of saying something
            "bot_speaking"     — bot is currently responding
        """
        # No messages yet → waiting for user to start the conversation
        if not self.chat_history:
            return "waiting_for_user"

        last_message = self.chat_history[-1]

        if last_message["role"] == "assistant":
            # Last message was from the bot → bot is speaking
            return "bot_speaking"

        elif last_message["role"] == "user":
            if last_message["content"].strip() != "":
                # User message with actual text → user is speaking
                return "user_speaking"
            else:
                # User message with empty string "" → waiting for user
                # This is the "empty message trick" — see module docstring
                return "waiting_for_user"
        else:
            return "waiting_for_user"

    def add_message_delta(self, text: str, role: str) -> bool:
        """Add text to the conversation. Returns True if this starts a NEW message.

        This is the core method that builds up messages word-by-word.
        Called by the orchestrator every time the STT produces a word.

        Behavior:
          1. If last message is a DIFFERENT role → create NEW message → True
          2. If last message is SAME role and was EMPTY "" → append → True
          3. If last message is SAME role and had content → append → False

        The return value (is_new_message) is critical:
          - True → orchestrator resets EMA to 0.0 (prevents false pause on first word)
          - False → just another word, no special action

        Args:
            text: The text to add (a word, or "" for empty message, or full bot response)
            role: "user" or "assistant"

        Returns:
            True if this is the START of a new message (new turn or first word after empty)
        """
        # Case 1: Different role or no history → start a brand new message
        if not self.chat_history or self.chat_history[-1]["role"] != role:
            self.chat_history.append({"role": role, "content": text})
            return True  # New message started!

        # Case 2 & 3: Same role → append to existing message
        else:
            last_content = self.chat_history[-1]["content"]

            # Smart spacing — add a space between words if needed
            # "Hello" + "world" → "Hello world" (space added)
            # "Hello " + "world" → "Hello world" (no extra space)
            # "" + "Hello" → "Hello" (no leading space)
            needs_space_left = last_content != "" and not last_content[-1].isspace()
            needs_space_right = text != "" and not text[0].isspace()
            if needs_space_left and needs_space_right:
                text = " " + text

            # Append to existing message
            self.chat_history[-1]["content"] += text

            # Return True if the message WAS empty (Case 2)
            # This means: first real word just arrived after bot finished
            return last_content == ""

    def mark_interruption(self):
        """Add '—' to the last assistant message to show it was interrupted.

        Called when the user starts speaking while the bot is talking.
        The dash makes it clear in the chat history that the bot was cut off.
        """
        if self.chat_history and self.chat_history[-1]["role"] == "assistant":
            self.chat_history[-1]["content"] += "—"

    def get_last_user_text(self) -> str:
        """Get the most recent user message text.

        Used by _generate_response() to know what the user said.
        Searches backwards through history to find the last user entry.
        """
        for msg in reversed(self.chat_history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def preprocessed_messages(self) -> list[dict[str, str]]:
        """Clean up chat history for sending to the LLM.

        The raw chat_history might have:
          - Empty messages (the "" entries from state transitions)
          - Consecutive messages from the same role
          - Extra whitespace

        This cleans it up into a format any LLM API will accept:
          - No empty messages
          - Consecutive same-role messages merged
          - Whitespace stripped
        """
        if not self.chat_history:
            return []

        cleaned = []
        for msg in self.chat_history:
            content = msg["content"].strip()
            if not content:
                continue  # Skip empty messages

            # Merge consecutive messages from the same role
            if cleaned and cleaned[-1]["role"] == msg["role"]:
                cleaned[-1]["content"] += " " + content
            else:
                cleaned.append({"role": msg["role"], "content": content})

        return cleaned
