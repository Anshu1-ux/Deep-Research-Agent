import uuid

import gradio as gr

from src.graph import get_checkpointed_graph


def run_research(query: str) -> str:
    if not query or not query.strip():
        return "Please enter a research question."

    thread_id = str(uuid.uuid4())
    with get_checkpointed_graph() as app:
        result = app.invoke(
            {"query": query},
            config={"configurable": {"thread_id": thread_id}},
        )

    return result["final_report"]


with gr.Blocks() as demo:
    gr.Markdown("## Deep Research Agent")

    with gr.Row():
        query_input = gr.Textbox(
            show_label=False,
            placeholder="e.g., What are the tradeoffs between RRF and weighted score fusion?",
            scale=4,
        )
        submit_btn = gr.Button("Start Research", variant="primary", scale=1)

    output_display = gr.Markdown(label="Research Report")

    submit_btn.click(fn=run_research, inputs=query_input, outputs=output_display)
    query_input.submit(fn=run_research, inputs=query_input, outputs=output_display)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
