import gradio as gr
from app.auth.users import authenticate
from app.auth.jwt import create_access_token
from app.rag.retriever import query
from app.guardrails.pipeline import check_input, check_output, GuardrailViolation

# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login(username: str, password: str):
    """Returns (token, role, status_message)."""
    if not username or not password:
        return "", "", "Please enter username and password."

    user = authenticate(username.strip(), password.strip())
    if not user:
        return "", "", "❌ Incorrect username or password."

    token = create_access_token({"sub": username.strip()})
    role = user["role"]
    name = user["full_name"]
    return token, role, f"✅ Logged in as **{name}** — role: `{role}`"


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def chat(message: str, history: list, token: str, role: str):
    """
    Runs message through guardrails → RAG → guardrails.
    history is a list of (user, bot) tuples.
    Returns (updated_history, cleared_input).
    """
    if not token:
        history = history + [("", "Please log in first.")]
        return history, ""

    if not message.strip():
        return history, ""

    # Input guardrails
    try:
        check_input(message)
    except GuardrailViolation as e:
        history = history + [(message, f"🚫 **Blocked:** {e.detail}")]
        return history, ""

    # RAG query
    try:
        result = query(question=message, user_role=role)
    except Exception as e:
        history = history + [(message, f"⚠️ Error: {str(e)}")]
        return history, ""

    # Output guardrails
    try:
        check_output(result["answer"])
    except GuardrailViolation as e:
        history = history + [(message, f"🚫 **Response blocked:** {e.detail}")]
        return history, ""

    # Format answer + sources
    answer = result["answer"]
    if result["sources"]:
        sources = "  \n".join(f"📄 `{s}`" for s in result["sources"])
        answer = f"{answer}\n\n---\n**Sources:**  \n{sources}"

    history = history + [(message, answer)]
    return history, ""


# ---------------------------------------------------------------------------
# Build Gradio Blocks UI
# ---------------------------------------------------------------------------

def build():
    with gr.Blocks(title="Nexora Internal Chatbot") as demo:

        # Hidden session state
        token_state = gr.State("")
        role_state = gr.State("employee")

        gr.Markdown("# 💬 Nexora Internal Chatbot")
        gr.Markdown(
            "Ask questions about company policies, HR, finance, and more. "
            "Your access is determined by your role."
        )

        # --- Login ---
        with gr.Group():
            gr.Markdown("### 🔐 Login")
            with gr.Row():
                username_input = gr.Textbox(
                    label="Username", placeholder="e.g. alice", scale=1
                )
                password_input = gr.Textbox(
                    label="Password", type="password", placeholder="••••••", scale=1
                )
                login_btn = gr.Button("Login", variant="primary", scale=0)
            login_status = gr.Markdown("_Not logged in_")

        gr.Markdown("---")

        # --- Chat ---
        chatbot = gr.Chatbot(
            label="Chat",
            height=420,
            placeholder="Log in above, then ask a question.",
        )
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your question",
                placeholder="What is the leave policy?",
                scale=5,
                container=False,
                autofocus=True,
            )
            send_btn = gr.Button("Send ↵", variant="primary", scale=1)

        gr.Markdown(
            "**Demo users:** alice · bob · carol · dave · eve | "
            "**Passwords:** username + `123`  (e.g. alice123)"
        )

        # --- Events ---
        login_btn.click(
            fn=login,
            inputs=[username_input, password_input],
            outputs=[token_state, role_state, login_status],
        )
        password_input.submit(
            fn=login,
            inputs=[username_input, password_input],
            outputs=[token_state, role_state, login_status],
        )

        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, token_state, role_state],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, token_state, role_state],
            outputs=[chatbot, msg_input],
        )

    return demo
