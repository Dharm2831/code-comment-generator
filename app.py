"""Web app: paste Python or Java code, get it back with a comment above every function.

Run locally:      python app.py            -> open http://127.0.0.1:7860
Temporary link:   python app.py --share    -> public link while your laptop stays on
"""

import sys

import gradio as gr

from src.prediction import CommentGenerator

generator = CommentGenerator()   # loads the trained model once, at start-up

PYTHON_EXAMPLE = '''def is_palindrome(text):
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
'''

JAVA_EXAMPLE = '''public class MathUtils {
    public static int factorial(int n) {
        if (n <= 1) {
            return 1;
        }
        return n * factorial(n - 1);
    }

    public static boolean isEven(int number) {
        return number % 2 == 0;
    }
}
'''


def run(code, language):
    if not code or not code.strip():
        return "", [], "Paste some code first."
    lang = {"Auto-detect": "auto", "Python": "python", "Java": "java"}[language]
    commented, rows = generator.comment_code(code, lang)
    return commented, rows, f"Added {len(rows)} comment(s)."


with gr.Blocks(title="AI Code Comment Generator") as demo:
    gr.Markdown("# AI Code Comment Generator\n"
                "Paste Python or Java code. A fine-tuned CodeT5 Transformer writes a comment for each function.")
    with gr.Row():
        with gr.Column():
            code_in = gr.Code(label="Your code (paste here)", language=None, lines=18)
            language = gr.Radio(["Auto-detect", "Python", "Java"], value="Auto-detect", label="Language")
            button = gr.Button("Generate comments", variant="primary")
        with gr.Column():
            code_out = gr.Code(label="Commented code", language=None, lines=18, interactive=False)
            table = gr.Dataframe(headers=["Function", "Generated comment"], label="Comments", wrap=True)
            status = gr.Markdown()
    gr.Examples([[PYTHON_EXAMPLE, "Python"], [JAVA_EXAMPLE, "Java"]], inputs=[code_in, language])
    button.click(run, inputs=[code_in, language], outputs=[code_out, table, status])

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
