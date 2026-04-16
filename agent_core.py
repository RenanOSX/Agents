#!/usr/bin/env python3
import ast
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

API_ENV_KEY = "GEMINI_API_KEY"
GROQ_API_ENV_KEY = "GROQ_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
WIKIPEDIA_API_URL = "https://pt.wikipedia.org/w/api.php"
DEFAULT_MAX_GEMINI_ATTEMPTS = 5
MIN_RETRY_DELAY_SECONDS = 0.5
RETRYABLE_HTTP_CODES = (429, 502, 503, 504)


class GeminiError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_message: Optional[str] = None,
        fallback_eligible: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        self.fallback_eligible = fallback_eligible
        super().__init__(user_message)


class GroqError(RuntimeError):
    def __init__(self, user_message: str, technical_message: Optional[str] = None) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        super().__init__(user_message)


def log_status(message: str) -> None:
    print(f"[Gemini]: {message}", flush=True)


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
            if isinstance(retry_delay, str):
                parsed_delay = _parse_duration_text(retry_delay)
                if parsed_delay is not None:
                    return parsed_delay

        message = (parsed.get("error") or {}).get("message", "")
        match = re.search(
            r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)",
            message,
            re.IGNORECASE,
        )
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            return _convert_to_seconds(value, unit)

    return None


def _parse_duration_text(text: str) -> Optional[float]:
    value = text.strip().lower()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(ms|s|m|h)", value)
    if not match:
        return None
    return _convert_to_seconds(float(match.group(1)), match.group(2).lower())


def _convert_to_seconds(value: float, unit: str) -> float:
    if unit == "ms":
        return max(value / 1000.0, 0.0)
    if unit == "s":
        return max(value, 0.0)
    if unit == "m":
        return max(value * 60.0, 0.0)
    if unit == "h":
        return max(value * 3600.0, 0.0)
    return max(value, 0.0)


def extract_api_error_message(error_body: str) -> str:
    try:
        parsed = json.loads(error_body)
    except json.JSONDecodeError:
        return error_body.strip() or "Erro desconhecido da API."

    message = (parsed.get("error") or {}).get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return error_body.strip() or "Erro desconhecido da API."


def is_quota_exceeded_error(error_body: str) -> bool:
    message = extract_api_error_message(error_body).lower()
    indicators = (
        "quota exceeded",
        "billing details",
        "free_tier_requests",
        "daily limit",
        "resource_exhausted",
        "quota",
    )
    return any(indicator in message for indicator in indicators)


def _read_key_from_env_files(env_key: str, env_file: str) -> Optional[str]:
    env_files = [env_file, env_file.lstrip("."), "env.local"]
    seen: set[str] = set()
    for path in env_files:
        if path in seen:
            continue
        seen.add(path)
        try:
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if line.startswith(env_key + "="):
                        value = line.split("=", 1)[1].strip()
                        if value:
                            return value
        except FileNotFoundError:
            continue

    return None


def load_optional_api_key(env_key: str, env_file: str = ".env.local") -> Optional[str]:
    value = os.environ.get(env_key)
    if value and value.strip():
        return value.strip()

    return _read_key_from_env_files(env_key, env_file)


def load_api_key(env_key: str = API_ENV_KEY, env_file: str = ".env.local") -> str:
    value = load_optional_api_key(env_key, env_file)
    if value:
        return value

    raise GeminiError(
        "Chave de API nao encontrada. Configure GEMINI_API_KEY em .env.local ou env.local."
    )


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        for key in (
            "content",
            "outputText",
            "response",
            "output",
            "candidates",
            "responses",
            "parts",
        ):
            if key in value:
                return extract_text(value[key])
        return ""
    if isinstance(value, list):
        return "".join(extract_text(item) for item in value)
    return ""


def format_wait_hint(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return "alguns segundos"
    if seconds < 60:
        rounded = max(int(round(seconds)), 1)
        return f"{rounded} segundos"

    minutes = max(int(round(seconds / 60.0)), 1)
    if minutes == 1:
        return "1 minuto"
    return f"{minutes} minutos"


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        max_attempts: int = DEFAULT_MAX_GEMINI_ATTEMPTS,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_attempts = max_attempts

    def send_request(self, prompt: str) -> str:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
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
                "X-goog-api-key": self.api_key,
            },
        )

        attempts = 1
        while True:
            log_status("Solicitando resposta ao modelo...")
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                quota_exceeded = is_quota_exceeded_error(body)
                retry_delay = parse_retry_delay_seconds(body)
                should_retry = (
                    exc.code in RETRYABLE_HTTP_CODES
                    and not quota_exceeded
                    and attempts < self.max_attempts
                )

                if should_retry:
                    delay = retry_delay
                    if delay is None:
                        delay = float(2 ** attempts)
                    delay = max(delay, MIN_RETRY_DELAY_SECONDS)
                    log_status(
                        f"Falha temporaria HTTP {exc.code}. Nova tentativa em {delay:.1f}s."
                    )
                    time.sleep(delay)
                    attempts += 1
                    continue

                raise self._build_http_error(exc.code, body, retry_delay, quota_exceeded)
            except urllib.error.URLError as exc:
                raise GeminiError(
                    "Nao foi possivel terminar o pedido por falha de conexao com a API.",
                    technical_message=str(exc),
                )

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GeminiError(
                "Nao foi possivel interpretar a resposta da API. Tente novamente em instantes.",
                technical_message=f"JSON invalido: {exc}",
            )

        result = extract_text(parsed.get("candidates") or parsed)
        if result:
            return result.strip()

        if isinstance(parsed, dict) and parsed.get("error"):
            raise self._build_http_error(500, body, None, is_quota_exceeded_error(body))

        raise GeminiError("Resposta vazia da API. Tente novamente em alguns instantes.")

    def _build_http_error(
        self,
        status_code: int,
        body: str,
        retry_delay: Optional[float],
        quota_exceeded: bool,
    ) -> GeminiError:
        message = extract_api_error_message(body)

        if status_code == 429 and quota_exceeded:
            if retry_delay and retry_delay > 0:
                available_at = datetime.now() + timedelta(seconds=retry_delay)
                user_message = (
                    "Infelizmente voce atingiu o limite diario de pedidos. "
                    f"Por favor aguarde ate {available_at:%H:%M} para utilizar a API novamente."
                )
            else:
                user_message = (
                    "Infelizmente voce atingiu o limite diario de pedidos. "
                    "Por favor aguarde a renovacao da cota para utilizar a API novamente."
                )
            return GeminiError(
                user_message,
                technical_message=message,
                fallback_eligible=True,
            )

        if status_code == 429:
            wait_hint = format_wait_hint(retry_delay)
            return GeminiError(
                "Nao foi possivel terminar o pedido devido a quantidade de pedidos. "
                f"Por favor aguarde {wait_hint} e tente novamente.",
                technical_message=message,
                fallback_eligible=True,
            )

        if status_code in (502, 503, 504):
            wait_hint = format_wait_hint(retry_delay)
            return GeminiError(
                "A API esta instavel no momento e nao foi possivel concluir o pedido. "
                f"Tente novamente em {wait_hint}.",
                technical_message=message,
            )

        return GeminiError(
            f"Erro da API Gemini (HTTP {status_code}). Tente novamente em alguns instantes.",
            technical_message=message,
        )


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GROQ_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model

    def send_request(self, prompt: str) -> str:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = extract_api_error_message(body)
            retry_delay = parse_retry_delay_seconds(body)

            if exc.code == 429:
                wait_hint = format_wait_hint(retry_delay)
                raise GroqError(
                    "Nao foi possivel concluir o fallback porque o Groq tambem esta com limite de pedidos. "
                    f"Tente novamente em {wait_hint}.",
                    technical_message=message,
                )

            raise GroqError(
                f"Erro da API Groq (HTTP {exc.code}). Tente novamente em alguns instantes.",
                technical_message=message,
            )
        except urllib.error.URLError as exc:
            raise GroqError(
                "Falha de conexao ao acionar o fallback via Groq.",
                technical_message=str(exc),
            )

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GroqError(
                "Resposta invalida da API Groq durante fallback.",
                technical_message=f"JSON invalido: {exc}",
            )

        choices = parsed.get("choices") if isinstance(parsed, dict) else None
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        raise GroqError("Resposta vazia da API Groq durante fallback.")


class FailoverLLMClient:
    def __init__(
        self,
        primary_client: GeminiClient,
        fallback_client: Optional[GroqClient] = None,
    ) -> None:
        self.primary_client = primary_client
        self.fallback_client = fallback_client

    def send_request(self, prompt: str) -> str:
        try:
            return self.primary_client.send_request(prompt)
        except GeminiError as exc:
            if not exc.fallback_eligible or self.fallback_client is None:
                raise

            log_status("Limite da Gemini detectado. Tentando fallback automatico no Groq...")
            try:
                result = self.fallback_client.send_request(prompt)
            except GroqError as fallback_exc:
                technical_details = (
                    f"Gemini: {exc.technical_message or exc.user_message} | "
                    f"Groq: {fallback_exc.technical_message or fallback_exc.user_message}"
                )
                raise GeminiError(
                    fallback_exc.user_message,
                    technical_message=technical_details,
                    fallback_eligible=False,
                )

            log_status("Fallback no Groq executado com sucesso.")
            return result


def build_default_llm_client(
    gemini_api_key: str,
    groq_api_key: Optional[str] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
) -> FailoverLLMClient:
    primary_client = GeminiClient(api_key=gemini_api_key, model=gemini_model)
    resolved_groq_key = groq_api_key or load_optional_api_key(GROQ_API_ENV_KEY)

    if not resolved_groq_key:
        return FailoverLLMClient(primary_client=primary_client, fallback_client=None)

    fallback_client = GroqClient(api_key=resolved_groq_key, model=groq_model)
    return FailoverLLMClient(primary_client=primary_client, fallback_client=fallback_client)


def trim_history_entries(entries: List[str], max_chars: int) -> List[str]:
    trimmed: List[str] = []
    for entry in entries:
        if len(entry) > max_chars:
            trimmed.append(entry[:max_chars] + " ... [cortado]")
        else:
            trimmed.append(entry)
    return trimmed


def shorten_for_display(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


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

        if upper.startswith("PENSAMENTO:"):
            thought_parts.append(stripped[len("PENSAMENTO:") :].strip())
            collecting = True
            continue

        if (
            upper.startswith("ACTION:")
            or upper.startswith("AÇÃO:")
            or upper.startswith("ACAO:")
            or upper.startswith("FINAL:")
        ):
            break

        if collecting and stripped:
            thought_parts.append(stripped)

    return " ".join(thought_parts).strip()


def parse_agent_output(text: str) -> Tuple[str, str]:
    normalized = text.strip()
    if not normalized:
        return "invalid", "Saida vazia do modelo."

    upper_normalized = normalized.upper()
    if upper_normalized.startswith("FINAL:"):
        return "final", normalized[len("FINAL:") :].strip()
    for label in ("ACTION:", "AÇÃO:", "ACAO:"):
        if upper_normalized.startswith(label):
            return "action", normalized[len(label) :].strip()

    for line in normalized.splitlines():
        stripped = line.strip()
        upper_line = stripped.upper()
        if upper_line.startswith("FINAL:"):
            return "final", stripped[len("FINAL:") :].strip()
        if upper_line.startswith("ACTION:"):
            return "action", stripped[len("ACTION:") :].strip()
        if upper_line.startswith("AÇÃO:"):
            return "action", stripped[len("AÇÃO:") :].strip()
        if upper_line.startswith("ACAO:"):
            return "action", stripped[len("ACAO:") :].strip()

    return "invalid", normalized


def wikipedia_search(term: str) -> str:
    normalized_term = term.strip()
    if not normalized_term:
        return "Termo de busca vazio para wikipedia_search."

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "exintro": "1",
        "explaintext": "1",
        "titles": normalized_term,
        "redirects": "1",
    }
    url = WIKIPEDIA_API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; GeminiAgent/1.0; +https://example.com)",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        first_line = body.splitlines()[0] if body else exc.reason
        return f"Erro HTTP Wikipedia {exc.code}: {first_line}"
    except urllib.error.URLError as exc:
        return f"Erro de conexao Wikipedia: {exc}"

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return "Wikipedia retornou JSON invalido."

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        extract = page.get("extract")
        if extract:
            return extract.strip()[:2500]
    return f"Nada encontrado para '{normalized_term}' na Wikipedia."


def python_eval(expression: str) -> str:
    try:
        node = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Expressao invalida: {exc}")

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
            raise ValueError("Expressao nao permitida")

    result = eval(compile(node, filename="<ast>", mode="eval"), {"__builtins__": {}})
    return str(result)


TOOLS: Dict[str, Callable[[str], str]] = {
    "wikipedia_search": wikipedia_search,
    "python_eval": python_eval,
}


def execute_tool(
    action_text: str,
    tools: Optional[Dict[str, Callable[[str], str]]] = None,
) -> Tuple[str, bool]:
    normalized = action_text.strip()
    if not normalized:
        return "Acao invalida: ferramenta nao informada.", False

    parts = normalized.split(None, 1)
    tool_name = parts[0]
    tool_arg = parts[1] if len(parts) > 1 else ""
    available_tools = tools or TOOLS
    tool = available_tools.get(tool_name)
    if not tool:
        return f"Ferramenta desconhecida: {tool_name}.", False

    log_status(f"Utilizando ferramenta {tool_name} para processar.")
    try:
        return tool(tool_arg), True
    except Exception as exc:
        return f"Erro ao executar {tool_name}: {exc}", True