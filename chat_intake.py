"""
chat_intake.py

Conversational RAG interface:
  user free text -> retrieval over active/recruiting trials -> grounded chatbot response

Run:
    python chat_intake.py
"""

import os

from retriever import chat_search_and_answer


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def run_chat() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    print("Clinical trial matcher chat")
    print("Chat naturally about your interests, health context, and preferences.")
    print("Type 'exit' to quit.\n")

    conversation_turns: list[str] = []
    while True:
        user_msg = input("You: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        conversation_turns.append(f"User: {user_msg}")
        context = "\n".join(conversation_turns[-8:])
        result = chat_search_and_answer(user_msg, conversation_context=context)

        answer = result.get("answer", "I could not generate a response.")
        print(f"\nAssistant: {answer}\n")

        top_trials = result.get("retrieval", {}).get("top_trials", [])
        if top_trials:
            print("Top retrieved trial IDs:", ", ".join(t.get("nct_id", "") for t in top_trials[:5]))
            print("")

        conversation_turns.append(f"Assistant: {answer}")


if __name__ == "__main__":
    run_chat()
