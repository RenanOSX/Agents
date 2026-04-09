#!/usr/bin/env python3
import ast
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

API_ENV_KEY = "GEMINI_API_KEY"
GEMINI_MODEL = "gemini-2.5-flash"
WIKIPEDIA_API_URL = "https://pt.wikipedia.org/w/api.php"
HISTORY_MAX_ENTRIES = 4
HISTORY_ENTRY_MAX_CHARS = 400
GEMINI_MIN_INTERVAL = 51
RATE_LIMIT_STATE_FILE = ".react_agent_rate_limit.json"
MAX_GEMINI_ATTEMPTS = 5
_last_gemini_request_time: Optional[float] = None


def save_next_allowed_timestamp(next_allowed: float) -> None:
    data = {"next_allowed": next_allowed}
    with open(RATE_LIMIT_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file)


def load_next_allowed_timestamp() -> Optional[float]:
    try:
        with open(RATE_LIMIT_STATE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    value = data.get("next_allowed")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_retry_delay_seconds(error_body: str) -> Optional[float]:
    try:
        parsed = json.loads(error_body)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        details = (parsed.get("error") or {}).get("details") or []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                try:
                    return max(float(retry_delay[:-1]), 0.0)
                except ValueError:
                    pass

        message = (parsed.get("error") or {}).get("message", "")
        match = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", message)
        if match:
            return max(float(match.group(1)), 0.0)

    return None


def extract_api_error_message(error_body: str) -> str:
    try:
        parsed = json.loads(error_body)
    except json.JSONDecodeError:
        return error_body.strip() or "Erro desconhecido da API."

    message = (parsed.get("error") or {}).get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return error_body.strip() or "Erro desconhecido da API."


def wait_for_gemini_rate_limit() -> None:
    global _last_gemini_request_time
    now = time.time()
    next_allowed_candidates: List[float] = []

    if _last_gemini_request_time is not None:
        next_allowed_candidates.append(_last_gemini_request_time + GEMINI_MIN_INTERVAL)

    saved_next_allowed = load_next_allowed_timestamp()
    if saved_next_allowed is not None:
        next_allowed_candidates.append(saved_next_allowed)

    if not next_allowed_candidates:
        return

    next_allowed = max(next_allowed_candidates)
    delay = next_allowed - now
    if delay > 0:
        print(f"Aguardando {delay:.1f}s para respeitar o limite da API Gemini...", flush=True)
        time.sleep(delay)


@dataclass
class AgentState:
    question: str
    history: List[str]
    observation: Optional[str] = None


def load_api_key(env_key: str = API_ENV_KEY, env_file: str = ".env.local") -> str:
    value = os.environ.get(env_key)
    if value:
        return value.strip()

    try:
        with open(env_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(env_key + "="):
                    return line.split("=", 1)[1].strip()
    except FileNotFoundError:
        pass

    raise RuntimeError(f"Chave de API não encontrada em {env_key} ou {env_file}")


def trim_history_entries(entries: List[str], max_chars: int) -> List[str]:
    trimmed = []
    for entry in entries:
        if len(entry) > max_chars:
            trimmed.append(entry[:max_chars] + " ... [cortado]")
        else:
            trimmed.append(entry)
    return trimmed


def build_gemini_prompt(state: AgentState) -> str:
    recent_entries = state.history[-HISTORY_MAX_ENTRIES:]
    trimmed_entries = trim_history_entries(recent_entries, HISTORY_ENTRY_MAX_CHARS)
    history_text = "\n".join(trimmed_entries)
    return (
        "Você é um agente ReAct. Use as ferramentas disponíveis para responder.\n"
        "Responda em português.\n"
        "Histórico:\n"
        f"{history_text}\n"
        "Pergunta:\n"
        f"{state.question}\n"
        "Instruções:\n"
        "- Primeiro pense em voz alta, indicando seu raciocínio.\n"
        "- Em seguida escolha uma ação.\n"
        "- Use pelo menos duas ferramentas sempre que fizer sentido para a tarefa.\n"
        "- Se tiver a resposta final, use FINAL: <resposta>.\n"
        "- Caso contrário, use ACTION: <nome_da_ferramenta> [argumentos].\n"
        "Ferramentas disponíveis:\n"
        "1. wikipedia_search <termo> - pesquisa na Wikipédia em português.\n"
        "2. python_eval <expressão> - resolve cálculos simples com segurança.\n"
        "Formato esperado:\n"
        "THOUGHT: ...\n"
        "ACTION: <nome_da_ferramenta> <argumentos>\n"
        "Ou FINAL: <resposta>\n"
    )


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        for key in ("content", "outputText", "response", "output", "candidates", "responses", "parts"):
            if key in value:
                return extract_text(value[key])
        return ""
    if isinstance(value, list):
        return "".join(extract_text(item) for item in value)
    return ""


def send_gemini_request(api_key: str, prompt: str) -> str:
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ]
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
    )

    attempts = 1
    while True:
        wait_for_gemini_rate_limit()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in (429, 503, 502, 504)
            retry_delay = parse_retry_delay_seconds(body)
            if retry_delay is None:
                retry_delay = float(2 ** attempts)

            if should_retry and attempts < MAX_GEMINI_ATTEMPTS:
                next_allowed = time.time() + max(retry_delay, 0.0)
                save_next_allowed_timestamp(next_allowed)
                print(
                    f"API Gemini retornou HTTP {exc.code}. Nova tentativa em {retry_delay:.1f}s...",
                    flush=True,
                )
                attempts += 1
                continue

            message = extract_api_error_message(body)
            if exc.code == 429:
                raise RuntimeError(
                    "Limite da Gemini atingido. Aguarde alguns segundos e tente novamente. "
                    f"Detalhe: {message}"
                )
            raise RuntimeError(f"Erro Gemini HTTP {exc.code}: {message}")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Erro de conexão Gemini: {exc}")

    global _last_gemini_request_time
    _last_gemini_request_time = time.time()
    save_next_allowed_timestamp(_last_gemini_request_time + GEMINI_MIN_INTERVAL)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON inválido da Gemini: {exc}: {body[:300]}")

    result = extract_text(parsed.get("candidates") or parsed)
    if result:
        return result.strip()

    raise RuntimeError("Resposta vazia da Gemini")


def wikipedia_search(term: str) -> str:
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": term,
        "redirects": "1",
    }
    url = WIKIPEDIA_API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; ReActAgent/1.0; +https://example.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"Erro HTTP Wikipédia {exc.code}: {body.splitlines()[0] if body else exc.reason}"
    except urllib.error.URLError as exc:
        return f"Erro de conexão Wikipédia: {exc}"

    data = json.loads(body)
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            return extract.strip()
    return f"Nada encontrado para '{term}' na Wikipédia." 


def python_eval(expression: str) -> str:
    node = ast.parse(expression, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.operator,
        ast.unaryop,
        ast.Pow,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.FloorDiv,
        ast.UAdd,
        ast.USub,
        ast.Load,
        ast.Tuple,
        ast.List,
    )
    for element in ast.walk(node):
        if not isinstance(element, allowed_nodes):
            raise ValueError("Expressão não permitida")
    result = eval(compile(node, filename="<ast>", mode="eval"), {"__builtins__": {}})
    return str(result)


def extract_thought(text: str) -> str:
    thought_parts: List[str] = []
    collecting = False

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("THOUGHT:"):
            thought_parts.append(stripped[len("THOUGHT:") :].strip())
            collecting = True
            continue

        if upper.startswith("ACTION:") or upper.startswith("FINAL:"):
            break

        if collecting and stripped:
            thought_parts.append(stripped)

    return " ".join(thought_parts).strip()


def shorten_for_display(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def parse_agent_output(text: str) -> Tuple[str, str]:
    normalized = text.strip()
    for label in ["FINAL:", "ACTION:"]:
        if normalized.upper().startswith(label):
            return label[:-1].lower(), normalized[len(label) :].strip()

    for line in normalized.splitlines():
        line = line.strip()
        if line.upper().startswith("FINAL:"):
            return "final", line[len("FINAL:") :].strip()
        if line.upper().startswith("ACTION:"):
            return "action", line[len("ACTION:") :].strip()
    return "action", normalized


TOOLS: Dict[str, Callable[[str], str]] = {
    "wikipedia_search": wikipedia_search,
    "python_eval": python_eval,
}


def execute_tool(action_text: str) -> str:
    parts = action_text.split(None, 1)
    tool_name = parts[0]
    tool_arg = parts[1] if len(parts) > 1 else ""
    tool = TOOLS.get(tool_name)
    if not tool:
        return f"Ferramenta desconhecida: {tool_name}."
    try:
        return tool(tool_arg)
    except Exception as exc:
        return f"Erro ao executar {tool_name}: {exc}"


def run_agent(question: str, api_key: str, max_turns: int = 4) -> str:
    state = AgentState(question=question, history=[])
    tool_actions_count = 0

    for turn in range(1, max_turns + 1):
        prompt = build_gemini_prompt(state)
        model_output = send_gemini_request(api_key, prompt)

        # Loop principal: pensa, escolhe ação, observa resultado e atualiza histórico.
        # Aqui o agente faz uma iteração completa de Thought → Action → Observation.
        thought = extract_thought(model_output)
        if thought:
            print(f"Thought {turn}: {shorten_for_display(thought)}", flush=True)

        state.history.append(f"THOUGHT {turn}: {model_output}")

        kind, payload = parse_agent_output(model_output)
        if kind == "final":
            if tool_actions_count < 2 and turn < max_turns:
                reminder = "Use pelo menos duas ferramentas antes da resposta final."
                print(f"Observation {turn}: {reminder}", flush=True)
                state.history.append(f"OBSERVATION {turn}: {reminder}")
                continue
            print(f"Final: {payload}", flush=True)
            return payload

        print(f"Action {turn}: {payload}", flush=True)
        tool_actions_count += 1
        observation = execute_tool(payload)
        print(f"Observation {turn}: {shorten_for_display(observation, 320)}", flush=True)
        state.observation = observation
        state.history.append(f"OBSERVATION {turn}: {observation}")

    return "Não foi possível chegar a uma resposta final em número suficiente de passos."


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("Uso: python react_agent.py [pergunta em linguagem natural]")
        print("Se nenhuma pergunta for passada, o script pedirá a entrada via teclado.")
        return 0

    if len(argv) > 1:
        question = " ".join(argv[1:]).strip()
    else:
        question = input("Pergunta ou tarefa: ").strip()

    if not question:
        print("Pergunta não fornecida.")
        return 1

    try:
        api_key = load_api_key()
        answer = run_agent(question, api_key)
        print("Resposta final:", answer)
        return 0
    except Exception as exc:
        print("Erro:", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
