#!/usr/bin/env python3
import ast
import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Callable, Optional, Protocol, cast

API_ENV_KEY = "GEMINI_API_KEY"
GROQ_API_ENV_KEY = "GROQ_API_KEY"
OPEN_ROUTER_API_ENV_KEY = "OPEN_ROUTER_API_KEY"
MISTRAL_API_ENV_KEY = "MISTRAL_API_KEY"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPEN_ROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct"
DEFAULT_MISTRAL_MODEL = "mistral-small-latest"
WIKIPEDIA_API_URL = "https://pt.wikipedia.org/w/api.php"
WORLD_BANK_API_URL = "https://api.worldbank.org/v2"
DEFAULT_MAX_GEMINI_ATTEMPTS = 5
MIN_RETRY_DELAY_SECONDS = 0.5
RETRYABLE_HTTP_CODES = (429, 502, 503, 504)


JSONDict = dict[str, object]
DEFAULT_LOG_PROVIDER = "Gemini"
_current_log_provider = DEFAULT_LOG_PROVIDER


def set_log_provider(provider_name: str) -> None:
    global _current_log_provider
    normalized = provider_name.strip() if provider_name else ""
    _current_log_provider = normalized or DEFAULT_LOG_PROVIDER


def get_log_provider() -> str:
    return _current_log_provider


def format_log_line(message: str) -> str:
    return f"[{get_log_provider()}]: {message}"


class LLMClient(Protocol):
    def send_request(self, prompt: str) -> str:
        ...


class GeminiError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_message: Optional[str] = None,
        fallback_eligible: bool = False,
        rate_limited: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        self.fallback_eligible = fallback_eligible
        self.rate_limited = rate_limited
        super().__init__(user_message)


class GroqError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_message: Optional[str] = None,
        fallback_eligible: bool = False,
        rate_limited: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        self.fallback_eligible = fallback_eligible
        self.rate_limited = rate_limited
        super().__init__(user_message)


class OpenRouterError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_message: Optional[str] = None,
        fallback_eligible: bool = False,
        rate_limited: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        self.fallback_eligible = fallback_eligible
        self.rate_limited = rate_limited
        super().__init__(user_message)


class MistralError(RuntimeError):
    def __init__(
        self,
        user_message: str,
        technical_message: Optional[str] = None,
        fallback_eligible: bool = False,
        rate_limited: bool = False,
    ) -> None:
        self.user_message = user_message
        self.technical_message = technical_message
        self.fallback_eligible = fallback_eligible
        self.rate_limited = rate_limited
        super().__init__(user_message)


def log_status(message: str) -> None:
    print(format_log_line(message), flush=True)


def parse_retry_delay_seconds(error_body: str) -> Optional[float]:
    try:
        parsed_obj: object = json.loads(error_body)
    except json.JSONDecodeError:
        parsed_obj = None

    if isinstance(parsed_obj, dict):
        parsed = cast(JSONDict, parsed_obj)
        error_obj = parsed.get("error")
        details_obj: object = []
        if isinstance(error_obj, dict):
            details_obj = cast(JSONDict, error_obj).get("details")

        details: list[object] = cast(list[object], details_obj) if isinstance(details_obj, list) else []
        for detail_obj in details:
            if not isinstance(detail_obj, dict):
                continue
            detail = cast(JSONDict, detail_obj)
            retry_delay = detail.get("retryDelay")
            if isinstance(retry_delay, str):
                parsed_delay = _parse_duration_text(retry_delay)
                if parsed_delay is not None:
                    return parsed_delay

        message = ""
        if isinstance(error_obj, dict):
            raw_message = cast(JSONDict, error_obj).get("message")
            if isinstance(raw_message, str):
                message = raw_message
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
        parsed_obj: object = json.loads(error_body)
    except json.JSONDecodeError:
        return error_body.strip() or "Erro desconhecido da API."

    message_obj: object = None
    if isinstance(parsed_obj, dict):
        parsed = cast(JSONDict, parsed_obj)
        error_obj = parsed.get("error")
        if isinstance(error_obj, dict):
            message_obj = cast(JSONDict, error_obj).get("message")

    message = message_obj
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


def _resolve_provider_key(explicit_key: Optional[str], env_key: str) -> Optional[str]:
    if explicit_key is not None:
        normalized = explicit_key.strip()
        return normalized or None
    return load_optional_api_key(env_key)


def extract_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        typed_value = cast(JSONDict, value)
        text_value = typed_value.get("text")
        if isinstance(text_value, str):
            return text_value
        for key in (
            "content",
            "outputText",
            "response",
            "output",
            "candidates",
            "responses",
            "parts",
        ):
            next_value = typed_value.get(key)
            if next_value is not None:
                return extract_text(next_value)
        return ""
    if isinstance(value, list):
        values = cast(list[object], value)
        parts: list[str] = []
        for item in values:
            parts.append(extract_text(item))
        return "".join(parts)
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
            except (TimeoutError, socket.timeout) as exc:
                raise GeminiError(
                    "Nao foi possivel terminar o pedido por tempo limite da API. Tente novamente em alguns instantes.",
                    technical_message=str(exc),
                )

        try:
            parsed_obj: object = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GeminiError(
                "Nao foi possivel interpretar a resposta da API. Tente novamente em instantes.",
                technical_message=f"JSON invalido: {exc}",
            )

        if isinstance(parsed_obj, dict):
            parsed = cast(JSONDict, parsed_obj)
            candidate_or_root: object = parsed.get("candidates") or parsed
        else:
            candidate_or_root = parsed_obj

        result = extract_text(candidate_or_root)
        if result:
            return result.strip()

        if isinstance(parsed_obj, dict):
            parsed = cast(JSONDict, parsed_obj)
        else:
            parsed = None

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
                rate_limited=True,
            )

        if status_code == 429:
            wait_hint = format_wait_hint(retry_delay)
            return GeminiError(
                "Nao foi possivel terminar o pedido devido a quantidade de pedidos. "
                f"Por favor aguarde {wait_hint} e tente novamente.",
                technical_message=message,
                fallback_eligible=True,
                rate_limited=True,
            )

        if status_code in (502, 503, 504):
            wait_hint = format_wait_hint(retry_delay)
            return GeminiError(
                "A API esta instavel no momento e nao foi possivel concluir o pedido. "
                f"Tente novamente em {wait_hint}.",
                technical_message=message,
                fallback_eligible=True,
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
        payload: JSONDict = {
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

            if exc.code in (401, 403):
                raise GroqError(
                    "Nao foi possivel usar o fallback no Groq por falha de autenticacao/permissao. "
                    "Verifique a GROQ_API_KEY e as permissoes da conta.",
                    technical_message=message,
                    fallback_eligible=False,
                )

            if exc.code == 429:
                wait_hint = format_wait_hint(retry_delay)
                raise GroqError(
                    "Nao foi possivel concluir o fallback porque o Groq tambem esta com limite de pedidos. "
                    f"Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                    rate_limited=True,
                )

            if exc.code in (502, 503, 504):
                wait_hint = format_wait_hint(retry_delay)
                raise GroqError(
                    "A API Groq esta instavel no momento e nao foi possivel concluir o pedido. "
                    f"Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                )

            raise GroqError(
                f"Erro da API Groq (HTTP {exc.code}). Tente novamente em alguns instantes.",
                technical_message=message,
            )
        except urllib.error.URLError as exc:
            raise GroqError(
                "Falha de conexao ao acionar o fallback via Groq.",
                technical_message=str(exc),
                fallback_eligible=True,
            )
        except (TimeoutError, socket.timeout) as exc:
            raise GroqError(
                "Tempo limite ao acionar o fallback via Groq.",
                technical_message=str(exc),
                fallback_eligible=True,
            )

        try:
            parsed_obj: object = json.loads(body)
        except json.JSONDecodeError as exc:
            raise GroqError(
                "Resposta invalida da API Groq durante fallback.",
                technical_message=f"JSON invalido: {exc}",
            )

        choices: object = None
        if isinstance(parsed_obj, dict):
            parsed = cast(JSONDict, parsed_obj)
            choices = parsed.get("choices")
        if isinstance(choices, list) and choices:
            typed_choices = cast(list[object], choices)
            first_choice_obj = typed_choices[0]
            if isinstance(first_choice_obj, dict):
                first_choice = cast(JSONDict, first_choice_obj)
                message_obj = first_choice.get("message")
                if isinstance(message_obj, dict):
                    message = cast(JSONDict, message_obj)
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        raise GroqError("Resposta vazia da API Groq durante fallback.")


def _extract_openai_style_content(parsed_obj: object) -> Optional[str]:
    choices: object = None
    if isinstance(parsed_obj, dict):
        parsed = cast(JSONDict, parsed_obj)
        choices = parsed.get("choices")

    if isinstance(choices, list) and choices:
        typed_choices = cast(list[object], choices)
        first_choice_obj = typed_choices[0]
        if isinstance(first_choice_obj, dict):
            first_choice = cast(JSONDict, first_choice_obj)
            message_obj = first_choice.get("message")
            if isinstance(message_obj, dict):
                message = cast(JSONDict, message_obj)
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_OPEN_ROUTER_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model

    def send_request(self, prompt: str) -> str:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        payload: JSONDict = {
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
                "HTTP-Referer": "https://local.agents",
                "X-Title": "Agents-ReAct-Reflexion",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = extract_api_error_message(body)
            retry_delay = parse_retry_delay_seconds(body)

            if exc.code in (401, 403):
                raise OpenRouterError(
                    "OpenRouter recusou a autenticacao/permissao. Verifique OPEN_ROUTER_API_KEY.",
                    technical_message=message,
                    fallback_eligible=True,
                )

            if exc.code == 429:
                wait_hint = format_wait_hint(retry_delay)
                raise OpenRouterError(
                    f"OpenRouter esta com limite de pedidos. Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                    rate_limited=True,
                )

            if exc.code in (502, 503, 504):
                wait_hint = format_wait_hint(retry_delay)
                raise OpenRouterError(
                    f"OpenRouter esta instavel no momento. Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                )

            raise OpenRouterError(
                f"Erro da API OpenRouter (HTTP {exc.code}).",
                technical_message=message,
                fallback_eligible=True,
            )
        except urllib.error.URLError as exc:
            raise OpenRouterError(
                "Falha de conexao ao usar OpenRouter.",
                technical_message=str(exc),
                fallback_eligible=True,
            )
        except (TimeoutError, socket.timeout) as exc:
            raise OpenRouterError(
                "Tempo limite ao usar OpenRouter.",
                technical_message=str(exc),
                fallback_eligible=True,
            )

        try:
            parsed_obj: object = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(
                "Resposta invalida da API OpenRouter.",
                technical_message=f"JSON invalido: {exc}",
                fallback_eligible=True,
            )

        content = _extract_openai_style_content(parsed_obj)
        if content:
            return content

        raise OpenRouterError(
            "Resposta vazia da API OpenRouter.",
            fallback_eligible=True,
        )


class MistralClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MISTRAL_MODEL,
    ) -> None:
        self.api_key = api_key
        self.model = model

    def send_request(self, prompt: str) -> str:
        endpoint = "https://api.mistral.ai/v1/chat/completions"
        payload: JSONDict = {
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

            if exc.code in (401, 403):
                raise MistralError(
                    "Mistral recusou a autenticacao/permissao. Verifique MISTRAL_API_KEY.",
                    technical_message=message,
                    fallback_eligible=True,
                )

            if exc.code == 429:
                wait_hint = format_wait_hint(retry_delay)
                raise MistralError(
                    f"Mistral esta com limite de pedidos. Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                    rate_limited=True,
                )

            if exc.code in (502, 503, 504):
                wait_hint = format_wait_hint(retry_delay)
                raise MistralError(
                    f"Mistral esta instavel no momento. Tente novamente em {wait_hint}.",
                    technical_message=message,
                    fallback_eligible=True,
                )

            raise MistralError(
                f"Erro da API Mistral (HTTP {exc.code}).",
                technical_message=message,
                fallback_eligible=True,
            )
        except urllib.error.URLError as exc:
            raise MistralError(
                "Falha de conexao ao usar Mistral.",
                technical_message=str(exc),
                fallback_eligible=True,
            )
        except (TimeoutError, socket.timeout) as exc:
            raise MistralError(
                "Tempo limite ao usar Mistral.",
                technical_message=str(exc),
                fallback_eligible=True,
            )

        try:
            parsed_obj: object = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MistralError(
                "Resposta invalida da API Mistral.",
                technical_message=f"JSON invalido: {exc}",
                fallback_eligible=True,
            )

        content = _extract_openai_style_content(parsed_obj)
        if content:
            return content

        raise MistralError(
            "Resposta vazia da API Mistral.",
            fallback_eligible=True,
        )


class FailoverLLMClient:
    def __init__(
        self,
        primary_client: Optional[LLMClient] = None,
        fallback_client: Optional[LLMClient] = None,
        primary_name: str = "primario",
        fallback_name: str = "secundario",
        provider_chain: Optional[list[tuple[str, LLMClient]]] = None,
    ) -> None:
        self.primary_client = primary_client
        self.fallback_client = fallback_client
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._active_provider_index = 0

        if provider_chain is not None:
            self.provider_chain = provider_chain
        else:
            chain: list[tuple[str, LLMClient]] = []
            if primary_client is not None:
                chain.append((primary_name, primary_client))
            if fallback_client is not None:
                chain.append((fallback_name, fallback_client))
            self.provider_chain = chain

        if self.provider_chain:
            self.primary_name = self.provider_chain[0][0]
            self.primary_client = self.provider_chain[0][1]
            if len(self.provider_chain) > 1:
                self.fallback_name = self.provider_chain[1][0]
                self.fallback_client = self.provider_chain[1][1]
            set_log_provider(self.provider_chain[0][0])

    @staticmethod
    def _to_gemini_error(error: Exception) -> GeminiError:
        if isinstance(error, GeminiError):
            return error
        if isinstance(error, GroqError):
            return GeminiError(
                error.user_message,
                technical_message=error.technical_message,
                fallback_eligible=error.fallback_eligible,
                rate_limited=error.rate_limited,
            )
        if isinstance(error, OpenRouterError):
            return GeminiError(
                error.user_message,
                technical_message=error.technical_message,
                fallback_eligible=error.fallback_eligible,
                rate_limited=error.rate_limited,
            )
        if isinstance(error, MistralError):
            return GeminiError(
                error.user_message,
                technical_message=error.technical_message,
                fallback_eligible=error.fallback_eligible,
                rate_limited=error.rate_limited,
            )
        return GeminiError(
            "Erro inesperado no provedor de IA.",
            technical_message=str(error),
        )

    @staticmethod
    def _technical(error: Exception) -> str:
        if isinstance(error, GeminiError):
            return error.technical_message or error.user_message
        if isinstance(error, GroqError):
            return error.technical_message or error.user_message
        if isinstance(error, OpenRouterError):
            return error.technical_message or error.user_message
        if isinstance(error, MistralError):
            return error.technical_message or error.user_message
        return str(error)

    @staticmethod
    def _is_rate_limited(error: Exception) -> bool:
        if isinstance(error, GeminiError):
            return error.rate_limited
        if isinstance(error, GroqError):
            return error.rate_limited
        if isinstance(error, OpenRouterError):
            return error.rate_limited
        if isinstance(error, MistralError):
            return error.rate_limited
        return False

    @staticmethod
    def _is_fallback_eligible(error: Exception) -> bool:
        if isinstance(error, GeminiError):
            return error.fallback_eligible
        if isinstance(error, GroqError):
            return error.fallback_eligible
        if isinstance(error, OpenRouterError):
            return error.fallback_eligible
        if isinstance(error, MistralError):
            return error.fallback_eligible
        return False

    def send_request(self, prompt: str) -> str:
        if not self.provider_chain:
            raise GeminiError("Nenhum provedor de IA configurado para tentativa de resposta.")

        if self._active_provider_index >= len(self.provider_chain):
            self._active_provider_index = 0

        technical_errors: list[str] = []
        rate_limited_providers: list[str] = []
        last_error: Optional[Exception] = None
        attempted_count = 0

        for index in range(self._active_provider_index, len(self.provider_chain)):
            provider_name, provider_client = self.provider_chain[index]
            attempted_count += 1
            set_log_provider(provider_name)
            try:
                response = provider_client.send_request(prompt)
                self._active_provider_index = index
                return response
            except (GeminiError, GroqError, OpenRouterError, MistralError) as provider_exc:
                last_error = provider_exc
                technical_errors.append(f"{provider_name}: {self._technical(provider_exc)}")
                if self._is_rate_limited(provider_exc) and provider_name not in rate_limited_providers:
                    rate_limited_providers.append(provider_name)

                has_next = index < len(self.provider_chain) - 1
                if has_next:
                    next_provider_name = self.provider_chain[index + 1][0]
                    log_status(
                        f"Falha no provedor {provider_name}. Tentando sequencialmente o provedor {next_provider_name}..."
                    )
                    self._active_provider_index = index + 1
                    continue

        if attempted_count >= 2 and len(rate_limited_providers) >= 2:
            providers_text = ", ".join(rate_limited_providers)
            raise GeminiError(
                f"Infelizmente voce utilizou o limite das plataformas disponiveis ({providers_text}). "
                "Por favor aguarde a renovacao da cota em uma delas para continuar.",
                technical_message=" | ".join(technical_errors),
            )

        if last_error is not None:
            converted = self._to_gemini_error(last_error)
            raise GeminiError(
                converted.user_message,
                technical_message=" | ".join(technical_errors),
                fallback_eligible=False,
                rate_limited=converted.rate_limited,
            )

        raise GeminiError("Nao foi possivel obter resposta de nenhum provedor configurado.")


def build_default_llm_client(
    gemini_api_key: Optional[str] = None,
    groq_api_key: Optional[str] = None,
    open_router_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    open_router_model: str = DEFAULT_OPEN_ROUTER_MODEL,
    mistral_model: str = DEFAULT_MISTRAL_MODEL,
) -> FailoverLLMClient:
    resolved_groq_key = _resolve_provider_key(groq_api_key, GROQ_API_ENV_KEY)
    resolved_gemini_key = _resolve_provider_key(gemini_api_key, API_ENV_KEY)
    resolved_open_router_key = _resolve_provider_key(open_router_api_key, OPEN_ROUTER_API_ENV_KEY)
    resolved_mistral_key = _resolve_provider_key(mistral_api_key, MISTRAL_API_ENV_KEY)

    provider_chain: list[tuple[str, LLMClient]] = []

    if resolved_groq_key:
        provider_chain.append(("Groq", GroqClient(api_key=resolved_groq_key, model=groq_model)))
    if resolved_open_router_key:
        provider_chain.append(
            (
                "OpenRouter",
                OpenRouterClient(api_key=resolved_open_router_key, model=open_router_model),
            )
        )
    if resolved_mistral_key:
        provider_chain.append(("Mistral", MistralClient(api_key=resolved_mistral_key, model=mistral_model)))
    if resolved_gemini_key:
        provider_chain.append(("Gemini", GeminiClient(api_key=resolved_gemini_key, model=gemini_model)))

    if provider_chain:
        return FailoverLLMClient(provider_chain=provider_chain)

    raise GeminiError(
        "Nenhuma chave de API configurada. Defina GROQ_API_KEY, OPEN_ROUTER_API_KEY, "
        "MISTRAL_API_KEY e/ou GEMINI_API_KEY em env.local."
    )


def trim_history_entries(entries: list[str], max_chars: int) -> list[str]:
    trimmed: list[str] = []
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
    thought_parts: list[str] = []
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


def parse_agent_output(text: str) -> tuple[str, str]:
    def _collect_following_payload(lines: list[str], start_index: int) -> str:
        payload_lines: list[str] = []
        for candidate in lines[start_index + 1 :]:
            stripped_candidate = candidate.strip()
            if not stripped_candidate:
                continue
            upper_candidate = stripped_candidate.upper()
            if (
                upper_candidate.startswith("THOUGHT:")
                or upper_candidate.startswith("PENSAMENTO:")
                or upper_candidate.startswith("ACTION:")
                or upper_candidate.startswith("AÇÃO:")
                or upper_candidate.startswith("ACAO:")
                or upper_candidate.startswith("FINAL:")
            ):
                break
            payload_lines.append(stripped_candidate)

        return "\n".join(payload_lines).strip()

    normalized = text.strip()
    if not normalized:
        return "invalid", "Saida vazia do modelo."

    lines = normalized.splitlines()
    upper_normalized = normalized.upper()
    if upper_normalized.startswith("FINAL:"):
        payload = normalized[len("FINAL:") :].strip()
        if payload:
            return "final", payload
        if lines:
            multiline_payload = _collect_following_payload(lines, 0)
            if multiline_payload:
                return "final", multiline_payload
        return "final", ""

    for label in ("ACTION:", "AÇÃO:", "ACAO:"):
        if upper_normalized.startswith(label):
            payload = normalized[len(label) :].strip()
            if payload:
                return "action", payload
            if lines:
                multiline_payload = _collect_following_payload(lines, 0)
                if multiline_payload:
                    return "action", multiline_payload
            return "action", ""

    for index, line in enumerate(lines):
        stripped = line.strip()
        upper_line = stripped.upper()
        if upper_line.startswith("FINAL:"):
            payload = stripped[len("FINAL:") :].strip()
            if payload:
                return "final", payload
            multiline_payload = _collect_following_payload(lines, index)
            if multiline_payload:
                return "final", multiline_payload
            return "final", ""

        if upper_line.startswith("ACTION:"):
            payload = stripped[len("ACTION:") :].strip()
            if payload:
                return "action", payload
            multiline_payload = _collect_following_payload(lines, index)
            if multiline_payload:
                return "action", multiline_payload
            return "action", ""

        if upper_line.startswith("AÇÃO:"):
            payload = stripped[len("AÇÃO:") :].strip()
            if payload:
                return "action", payload
            multiline_payload = _collect_following_payload(lines, index)
            if multiline_payload:
                return "action", multiline_payload
            return "action", ""

        if upper_line.startswith("ACAO:"):
            payload = stripped[len("ACAO:") :].strip()
            if payload:
                return "action", payload
            multiline_payload = _collect_following_payload(lines, index)
            if multiline_payload:
                return "action", multiline_payload
            return "action", ""

    return "invalid", normalized


def wikipedia_search(term: str) -> str:
    normalized_term = term.strip()
    if not normalized_term:
        return "Termo de busca vazio para wikipedia_search."

    try:
        direct_data = _wikipedia_query_json(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "titles": normalized_term,
                "redirects": "1",
            }
        )
    except RuntimeError as exc:
        return str(exc)

    extract = _extract_first_wikipedia_extract(direct_data)
    if extract:
        return extract

    # Se o titulo exato nao existir, tenta busca por relevancia e abre o melhor resultado.
    try:
        search_data = _wikipedia_query_json(
            {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": normalized_term,
                "srlimit": "1",
                "srwhat": "text",
            }
        )
    except RuntimeError as exc:
        return str(exc)

    query_obj = search_data.get("query")
    if not isinstance(query_obj, dict):
        return f"Nada encontrado para '{normalized_term}' na Wikipedia."
    query = cast(JSONDict, query_obj)
    search_results_obj = query.get("search")
    search_results: list[object] = (
        cast(list[object], search_results_obj) if isinstance(search_results_obj, list) else []
    )
    if not search_results:
        return f"Nada encontrado para '{normalized_term}' na Wikipedia."

    top_result_obj = search_results[0]
    if not isinstance(top_result_obj, dict):
        return f"Nada encontrado para '{normalized_term}' na Wikipedia."
    top_result = cast(JSONDict, top_result_obj)
    top_title = top_result.get("title")
    if not isinstance(top_title, str) or not top_title.strip():
        return f"Nada encontrado para '{normalized_term}' na Wikipedia."

    try:
        fallback_data = _wikipedia_query_json(
            {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": "1",
                "explaintext": "1",
                "titles": top_title,
                "redirects": "1",
            }
        )
    except RuntimeError as exc:
        return str(exc)

    fallback_extract = _extract_first_wikipedia_extract(fallback_data)
    if fallback_extract:
        return fallback_extract

    return f"Nada encontrado para '{normalized_term}' na Wikipedia."


def _wikipedia_query_json(params: dict[str, str]) -> JSONDict:
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
        raise RuntimeError(f"Erro HTTP Wikipedia {exc.code}: {first_line}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Erro de conexao Wikipedia: {exc}")
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Tempo limite ao consultar Wikipedia: {exc}")

    try:
        data_obj: object = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError("Wikipedia retornou JSON invalido.")

    if not isinstance(data_obj, dict):
        raise RuntimeError("Wikipedia retornou formato inesperado.")

    return cast(JSONDict, data_obj)


def _extract_first_wikipedia_extract(data: JSONDict) -> Optional[str]:
    query_obj = data.get("query")
    if not isinstance(query_obj, dict):
        return None
    query = cast(JSONDict, query_obj)

    pages_obj = query.get("pages")
    if not isinstance(pages_obj, dict):
        return None
    pages = cast(JSONDict, pages_obj)

    for page_obj in pages.values():
        if not isinstance(page_obj, dict):
            continue
        page = cast(JSONDict, page_obj)
        extract = page.get("extract")
        if isinstance(extract, str) and extract.strip():
            return extract.strip()[:2500]
    return None


def _world_bank_fetch_json(url: str) -> JSONDict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; AgentsProject/1.0)",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        first_line = body.splitlines()[0] if body else exc.reason
        raise RuntimeError(f"Erro HTTP WorldBank {exc.code}: {first_line}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Erro de conexao WorldBank: {exc}")
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"Tempo limite ao consultar WorldBank: {exc}")

    try:
        parsed_obj: object = json.loads(body)
    except json.JSONDecodeError:
        raise RuntimeError("WorldBank retornou JSON invalido.")

    if isinstance(parsed_obj, list):
        parsed_list = cast(list[object], parsed_obj)
        # API do World Bank retorna [metadata, rows]
        if len(parsed_list) == 2:
            rows_obj = parsed_list[1]
            if isinstance(rows_obj, list):
                return {"rows": cast(list[object], rows_obj)}
            if rows_obj is None:
                return {"rows": []}

        error_message = _extract_world_bank_error_message(parsed_list)
        if error_message:
            raise RuntimeError(f"WorldBank retornou erro: {error_message}")
        raise RuntimeError("WorldBank retornou formato inesperado.")

    if isinstance(parsed_obj, dict):
        parsed_dict = cast(JSONDict, parsed_obj)
        error_message = _extract_world_bank_error_message(parsed_dict)
        if error_message:
            raise RuntimeError(f"WorldBank retornou erro: {error_message}")
        return parsed_dict

    raise RuntimeError("WorldBank retornou formato inesperado.")


def _extract_world_bank_error_message(payload: object) -> Optional[str]:
    if isinstance(payload, str):
        text = payload.strip()
        return text or None

    if isinstance(payload, list):
        for item in cast(list[object], payload):
            extracted = _extract_world_bank_error_message(item)
            if extracted:
                return extracted
        return None

    if isinstance(payload, dict):
        typed_payload = cast(JSONDict, payload)
        for key in ("message", "error", "detail", "details", "value"):
            if key not in typed_payload:
                continue
            extracted = _extract_world_bank_error_message(typed_payload.get(key))
            if extracted:
                return extracted
        return None

    return None


def _extract_country_code(row: JSONDict, requested_codes: set[str]) -> Optional[str]:
    iso3_obj = row.get("countryiso3code")
    if isinstance(iso3_obj, str):
        iso3_code = iso3_obj.strip().upper()
        if iso3_code:
            return iso3_code

    country_obj = row.get("country")
    if isinstance(country_obj, dict):
        country = cast(JSONDict, country_obj)
        raw_code = country.get("id")
        if isinstance(raw_code, str):
            normalized_code = raw_code.strip().upper()
            if normalized_code == "1W" and "WLD" in requested_codes:
                return "WLD"
            if normalized_code:
                return normalized_code

    return None


def _world_bank_latest_values(countries_csv: str, indicator: str) -> dict[str, tuple[int, float]]:
    normalized_codes = countries_csv.replace(";", ",")
    requested_codes_list: list[str] = []
    requested_codes_seen: set[str] = set()
    for raw_code in normalized_codes.split(","):
        code = raw_code.strip().upper()
        if not code or code in requested_codes_seen:
            continue
        requested_codes_seen.add(code)
        requested_codes_list.append(code)

    requested_codes = set(requested_codes_list)
    country_path = ";".join(requested_codes_list)

    url = (
        f"{WORLD_BANK_API_URL}/country/{urllib.parse.quote(country_path, safe=';')}/"
        f"indicator/{indicator}?format=json&per_page=2000"
    )
    data = _world_bank_fetch_json(url)
    rows_obj = data.get("rows")
    rows = cast(list[object], rows_obj) if isinstance(rows_obj, list) else []

    latest_by_country: dict[str, tuple[int, float]] = {}
    for row_obj in rows:
        if not isinstance(row_obj, dict):
            continue
        row = cast(JSONDict, row_obj)
        value_obj = row.get("value")
        date_obj = row.get("date")

        if not isinstance(value_obj, (int, float)):
            continue
        if not isinstance(date_obj, str):
            continue
        country_code = _extract_country_code(row, requested_codes)
        if not country_code:
            continue

        try:
            year = int(date_obj)
        except ValueError:
            continue

        existing = latest_by_country.get(country_code)
        if existing is None or year > existing[0]:
            latest_by_country[country_code] = (year, float(value_obj))

    return latest_by_country


def south_america_gdp_analysis(_: str) -> str:
    # Paises soberanos da America do Sul (ISO3)
    countries = "ARG,BOL,BRA,CHL,COL,ECU,GUY,PRY,PER,SUR,URY,VEN"
    try:
        gdp_latest = _world_bank_latest_values(countries, "NY.GDP.MKTP.CD")
        if len(gdp_latest) < 3:
            return "Dados insuficientes de PIB no WorldBank para identificar top 3 paises."

        top3 = sorted(gdp_latest.items(), key=lambda item: item[1][1], reverse=True)[:3]
        top3_codes = ",".join(code for code, _ in top3)

        gdp_per_capita_latest = _world_bank_latest_values(top3_codes, "NY.GDP.PCAP.CD")
        world_latest = _world_bank_latest_values("WLD", "NY.GDP.PCAP.CD")

        missing_codes = [code for code, _ in top3 if code not in gdp_per_capita_latest]
        if missing_codes:
            return (
                "Dados insuficientes de PIB per capita para os paises: "
                + ", ".join(missing_codes)
                + "."
            )

        world = world_latest.get("WLD")
        if world is None:
            return "Dados insuficientes de PIB per capita mundial no WorldBank."

        values = [gdp_per_capita_latest[code][1] for code, _ in top3]
        average = sum(values) / len(values)
        comparison = "maior" if average > world[1] else "menor"

        lines = [
            "Analise economica via WorldBank:",
            "Top 3 paises por PIB nominal (USD):",
        ]
        for code, (year, gdp_value) in top3:
            pc_year, pc_value = gdp_per_capita_latest[code]
            lines.append(
                f"- {code}: PIB={gdp_value:.2f} (ano {year}), PIB per capita={pc_value:.2f} (ano {pc_year})"
            )

        lines.append(f"Media PIB per capita top3={average:.2f}")
        lines.append(f"Media mundial PIB per capita={world[1]:.2f} (ano {world[0]})")
        lines.append(f"Comparacao: media top3 e {comparison} que a media mundial.")
        lines.append(
            "Use python_eval com os tres valores de PIB per capita para validar o calculo da media antes do FINAL."
        )
        return "\n".join(lines)
    except RuntimeError as exc:
        return str(exc)


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


TOOLS: dict[str, Callable[[str], str]] = {
    "wikipedia_search": wikipedia_search,
    "python_eval": python_eval,
    "south_america_gdp_analysis": south_america_gdp_analysis,
}


def execute_tool(
    action_text: str,
    tools: Optional[dict[str, Callable[[str], str]]] = None,
) -> tuple[str, bool]:
    normalized = action_text.strip()
    if not normalized:
        return "Acao invalida: ferramenta nao informada.", False

    parts = normalized.split(None, 1)
    tool_name = parts[0]
    tool_arg = parts[1] if len(parts) > 1 else ""
    if len(tool_arg) >= 2:
        quoted_double = tool_arg.startswith('"') and tool_arg.endswith('"')
        quoted_single = tool_arg.startswith("'") and tool_arg.endswith("'")
        if quoted_double or quoted_single:
            tool_arg = tool_arg[1:-1].strip()
    available_tools = tools or TOOLS
    tool = available_tools.get(tool_name)
    if not tool:
        return f"Ferramenta desconhecida: {tool_name}.", False

    log_status(f"Utilizando ferramenta {tool_name} para processar.")
    try:
        return tool(tool_arg), True
    except Exception as exc:
        return f"Erro ao executar {tool_name}: {exc}", True