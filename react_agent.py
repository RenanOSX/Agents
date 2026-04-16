#!/usr/bin/env python3
from dataclasses import dataclass
import sys
from typing import List, Optional

from agent_core import (
    API_ENV_KEY,
    DEFAULT_GEMINI_MODEL,
    GROQ_API_ENV_KEY,
    MISTRAL_API_ENV_KEY,
    OPEN_ROUTER_API_ENV_KEY,
    FailoverLLMClient,
    GeminiError,
    build_default_llm_client,
    execute_tool,
    extract_thought,
    format_log_line,
    get_log_provider,
    load_optional_api_key,
    log_status,
    parse_agent_output,
    shorten_for_display,
    trim_history_entries,
)

HISTORY_MAX_ENTRIES = 6
HISTORY_ENTRY_MAX_CHARS = 500


@dataclass
class AgentState:
    question: str
    history: List[str]
    observation: Optional[str] = None


def build_gemini_prompt(state: AgentState) -> str:
    return build_gemini_prompt_with_state(state, used_tools=set(), force_final=False)


def build_gemini_prompt_with_state(
    state: AgentState,
    used_tools: set[str],
    force_final: bool,
) -> str:
    recent_entries = state.history[-HISTORY_MAX_ENTRIES:]
    trimmed_entries = trim_history_entries(recent_entries, HISTORY_ENTRY_MAX_CHARS)
    history_text = "\n".join(trimmed_entries)
    used_tools_text = ", ".join(sorted(used_tools)) if used_tools else "nenhuma"

    final_instruction = ""
    if force_final:
        final_instruction = (
            "\nEstado do fluxo: voce ja usou south_america_gdp_analysis e python_eval.\n"
            "Resposta obrigatoria neste turno: FINAL: <resposta conclusiva>\n"
            "Nao chame ACTION novamente.\n"
        )

    return (
        "Voce e um agente ReAct. Responda em portugues e siga o formato estrito.\n"
        "Pergunta do usuario:\n"
        f"{state.question}\n\n"
        "Historico do loop:\n"
        f"{history_text}\n\n"
        "Ferramentas ja usadas:\n"
        f"{used_tools_text}\n\n"
        "Instrucoes obrigatorias:\n"
        "1. Sempre produza THOUGHT seguido de ACTION ou FINAL.\n"
        "2. Use ACTION quando precisar de ferramenta.\n"
        "3. Nao invente dados: use ferramentas para buscar/calcular.\n"
        "4. Antes de FINAL, use no minimo duas ferramentas.\n"
        "5. Se usar FINAL, responda diretamente a pergunta com justificativa curta.\n\n"
        "Ferramentas disponiveis:\n"
        "- wikipedia_search <termo>\n"
        "- south_america_gdp_analysis <contexto-livre>\n"
        "- python_eval <expressao>\n\n"
        "Regra importante para tarefa de PIB da America do Sul:\n"
        "- Primeiro use south_america_gdp_analysis.\n"
        "- Depois use python_eval para validar a media com os 3 valores retornados.\n"
        "- Em seguida responda em FINAL com a comparacao.\n\n"
        f"{final_instruction}"
        "Formato de resposta aceito:\n"
        "THOUGHT: ...\n"
        "ACTION: <ferramenta> <argumentos>\n"
        "ou\n"
        "FINAL: <resposta final>\n"
    )


def run_agent(
    question: str,
    api_key: str = "",
    max_turns: int = 6,
    min_tool_uses: int = 2,
    client: Optional[FailoverLLMClient] = None,
) -> str:
    if not question.strip():
        raise ValueError("Pergunta nao fornecida.")

    state = AgentState(question=question.strip(), history=[])
    model_client = client or build_default_llm_client(
        gemini_api_key=api_key or None,
        gemini_model=DEFAULT_GEMINI_MODEL,
    )
    tool_actions_count = 0
    used_tools: set[str] = set()

    for turn in range(1, max_turns + 1):
        force_final = (
            tool_actions_count >= min_tool_uses
            and "south_america_gdp_analysis" in used_tools
            and "python_eval" in used_tools
        )
        prompt = build_gemini_prompt_with_state(state, used_tools, force_final)
        model_output = model_client.send_request(prompt)

        thought = extract_thought(model_output)
        if thought:
            log_status(f"Estou pensando em: {shorten_for_display(thought)}")

        state.history.append(f"THOUGHT {turn}: {thought or '(nao informado)'}")
        state.history.append(f"MODEL_OUTPUT {turn}: {model_output}")

        kind, payload = parse_agent_output(model_output)
        if kind == "final":
            if tool_actions_count < min_tool_uses:
                reminder = "Use pelo menos duas ferramentas antes de concluir em FINAL."
                log_status(f"Observacao: {reminder}")
                state.history.append(f"OBSERVATION {turn}: {reminder}")
                continue

            log_status(f"Conclusao final: {payload}")
            return payload

        if kind == "invalid":
            reminder = (
                "Saida fora do formato ReAct. Use THOUGHT e depois ACTION: <ferramenta> <args> "
                "ou FINAL: <resposta>."
            )
            log_status(f"Observacao: {reminder}")
            state.history.append(f"OBSERVATION {turn}: {reminder}")
            continue

        if force_final:
            reminder = (
                "Voce ja usou south_america_gdp_analysis e python_eval. "
                "Agora responda somente com FINAL."
            )
            log_status(f"Observacao: {reminder}")
            state.history.append(f"OBSERVATION {turn}: {reminder}")
            continue

        log_status(f"Executando acao do passo {turn}: {payload}")
        observation, tool_used = execute_tool(payload)
        if tool_used:
            tool_actions_count += 1
            used_tools.add(payload.split(None, 1)[0].strip())

        shortened_observation = shorten_for_display(observation, 320)
        log_status(f"Observacao do passo {turn}: {shortened_observation}")
        state.observation = observation
        state.history.append(f"OBSERVATION {turn}: {observation}")

    failure_message = "Nao foi possivel chegar a uma resposta final em numero suficiente de passos."
    log_status(failure_message)
    return failure_message


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("Uso: python react_agent.py [pergunta em linguagem natural]")
        print("Se nenhuma pergunta for passada, o script pedira a entrada via teclado.")
        return 0

    if len(argv) > 1:
        question = " ".join(argv[1:]).strip()
    else:
        question = input("Pergunta ou tarefa: ").strip()

    if not question:
        print(format_log_line("Pergunta nao fornecida."))
        return 1

    try:
        api_key = load_optional_api_key(API_ENV_KEY)
        groq_api_key = load_optional_api_key(GROQ_API_ENV_KEY)
        open_router_api_key = load_optional_api_key(OPEN_ROUTER_API_ENV_KEY)
        mistral_api_key = load_optional_api_key(MISTRAL_API_ENV_KEY)
        if not api_key and not groq_api_key and not open_router_api_key and not mistral_api_key:
            raise GeminiError(
                "Nenhuma chave de API configurada. Defina GROQ_API_KEY, OPEN_ROUTER_API_KEY, "
                "MISTRAL_API_KEY e/ou GEMINI_API_KEY em env.local."
            )
        client = build_default_llm_client(
            gemini_api_key=api_key,
            groq_api_key=groq_api_key,
            open_router_api_key=open_router_api_key,
            mistral_api_key=mistral_api_key,
            gemini_model=DEFAULT_GEMINI_MODEL,
        )
        run_agent(question, api_key=api_key or "", client=client)
        return 0
    except GeminiError as exc:
        print(format_log_line(str(exc)))
        if exc.technical_message:
            print(f"[{get_log_provider()}][debug]: {exc.technical_message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(format_log_line("Ocorreu um erro interno inesperado. Tente novamente em instantes."))
        print(f"[{get_log_provider()}][debug]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
