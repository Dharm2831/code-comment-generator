"""Use the trained model on new code.

    from src.prediction import CommentGenerator
    gen = CommentGenerator()                       # loads models/comment-generator
    new_code, rows = gen.comment_code(source_code) # comment above every function

Run a quick test from the terminal:
    python -m src.prediction
"""

import ast
import os
import re

import torch

from src import config
from src.model import get_device, load_model
from src.preprocessing import make_source

# ------------------------------------------------------------------------------
# Finding functions inside pasted code
# ------------------------------------------------------------------------------
_JAVA_METHOD = re.compile(
    r"^[ \t]*(?:@\w+(?:\([^)]*\))?\s+)*"                                  # annotations
    r"(?:(?:public|private|protected|static|final|abstract|synchronized|native|default|strictfp)\s+)*"
    r"(?:<[^>;{}()]+>\s+)?"                                               # generic method <T>
    r"(?:[\w.$]+(?:<[^;{}()]*>)?(?:\[\])*\s+)?"                           # return type (none for constructors)
    r"(?P<name>[A-Za-z_$][\w$]*)\s*\([^;{}()]*\)\s*"                      # name(parameters)
    r"(?:throws\s+[\w.$,\s]+?)?\s*\{",
    re.MULTILINE,
)
_NOT_METHOD_NAMES = {"if", "for", "while", "switch", "catch", "try", "synchronized", "do", "else", "return", "new"}
_NOT_RETURN_TYPES = re.compile(r"\b(new|return|throw|else|case)\b")


def detect_language(code: str) -> str:
    """Very small heuristic: 'python' or 'java'."""
    java = len(re.findall(r"[{};]\s*$", code, re.MULTILINE)) + 3 * len(
        re.findall(r"\b(public|private|protected|void|static)\b", code))
    python = 3 * len(re.findall(r"^\s*(def|elif|import|from)\b.*", code, re.MULTILINE)) + len(
        re.findall(r":\s*$", code, re.MULTILINE))
    return "java" if java > python else "python"


def find_python_functions(code: str):
    """Return [(name, first_line_index, indent, function_source_without_docstring)]."""
    tree = ast.parse(code)
    lines = code.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        body_lines = lines[node.lineno - 1: node.end_lineno]
        doc = node.body[0] if node.body else None
        if (isinstance(doc, ast.Expr) and isinstance(getattr(doc, "value", None), ast.Constant)
                and isinstance(doc.value.value, str)):
            a, b = doc.lineno - node.lineno, doc.end_lineno - node.lineno
            body_lines = body_lines[:a] + body_lines[b + 1:]       # drop the existing docstring
        indent = re.match(r"\s*", lines[first]).group(0)
        found.append((node.name, first, indent, "\n".join(body_lines)))
    return sorted(found, key=lambda f: f[1])


def _matching_brace(code: str, open_pos: int) -> int:
    """Index of the '}' that closes the '{' at open_pos (skips strings and comments)."""
    depth, i, n = 0, open_pos, len(code)
    while i < n:
        two = code[i:i + 2]
        if two == "//":
            i = code.find("\n", i)
            i = n if i == -1 else i
            continue
        if two == "/*":
            i = code.find("*/", i + 2)
            i = n if i == -1 else i + 2
            continue
        ch = code[i]
        if ch in "\"'":
            i += 1
            while i < n and code[i] != ch:
                i += 2 if code[i] == "\\" else 1
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def find_java_methods(code: str):
    """Return [(name, first_line_index, indent, method_source)]."""
    found = []
    for m in _JAVA_METHOD.finditer(code):
        header = m.group(0)
        if m.group("name") in _NOT_METHOD_NAMES or _NOT_RETURN_TYPES.search(header.split("(")[0]):
            continue
        end = _matching_brace(code, m.end() - 1)
        first_line = code.count("\n", 0, m.start())
        indent = re.match(r"[ \t]*", header).group(0)
        found.append((m.group("name"), first_line, indent, code[m.start(): end + 1]))
    return found


# ------------------------------------------------------------------------------
# The generator
# ------------------------------------------------------------------------------
class CommentGenerator:
    def __init__(self, model_path: str | None = None):
        # Order: explicit path -> MODEL_ID environment variable -> local trained model
        model_path = model_path or os.environ.get("MODEL_ID") or config.MODEL_DIR
        if model_path == config.MODEL_DIR and not os.path.isfile(os.path.join(model_path, "config.json")):
            raise SystemExit(f"No trained model found in '{model_path}'. Train first:  python -m src.training")
        self.device = get_device()
        self.tokenizer, self.model = load_model(model_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def summarize(self, codes: list[str], lang: str) -> list[str]:
        """One short comment for each piece of code in the list."""
        if not codes:
            return []
        sources = [make_source(c, lang) for c in codes]
        enc = self.tokenizer(sources, max_length=config.MAX_SOURCE_LEN, truncation=True,
                             padding=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**enc, num_beams=config.NUM_BEAMS, max_new_tokens=config.MAX_TARGET_LEN,
                                  no_repeat_ngram_size=3, early_stopping=True)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]

    def comment_code(self, code: str, lang: str = "auto"):
        """Insert a generated comment above every function / method.

        Returns (commented_code, rows) where rows = [[function name, comment], ...].
        """
        if not code.strip():
            return "", []
        if lang == "auto":
            lang = detect_language(code)

        try:
            units = find_python_functions(code) if lang == "python" else find_java_methods(code)
        except SyntaxError:
            units = []                                   # incomplete snippet: treat as one block

        if not units:                                    # no functions found: one comment for everything
            comment = self.summarize([code], lang)[0]
            mark = "# " if lang == "python" else "// "
            return f"{mark}{comment}\n{code}", [["(whole snippet)", comment]]

        comments = self.summarize([u[3] for u in units], lang)
        lines = code.splitlines()
        rows = []
        for (name, first, indent, _), comment in sorted(zip(units, comments), key=lambda p: -p[0][1]):
            new = f"{indent}# {comment}" if lang == "python" else f"{indent}/** {comment} */"
            lines.insert(first, new)                     # insert bottom-up so line numbers stay valid
            rows.append([name, comment])
        return "\n".join(lines), rows[::-1]


if __name__ == "__main__":
    demo = '''def is_palindrome(text):
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return cleaned == cleaned[::-1]
'''
    generator = CommentGenerator()
    print(generator.comment_code(demo)[0])
