#!/usr/bin/env python3
from dataclasses import dataclass, field
import sys
from typing import Optional

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

MAX_STEPS_DEFAULT = 6
MAX_TRIALS_DEFAULT = 3
TRIAL_HISTORY_MAX_ENTRIES = 8
TRIAL_HISTORY_ENTRY_MAX_CHARS = 500


def _new_str_list() -> list[str]:
    return []


@dataclass
class TrialResult:
    scratchpad: list[str] = field(default_factory=_new_str_list)
    answer: str = ""
    finished: bool = False
    tool_actions_count: int = 0


@dataclass
class ReflexionAgent:
    question: str
    client: FailoverLLMClient
    max_steps: int = MAX_STEPS_DEFAULT
    max_trials: int = MAX_TRIALS_DEFAULT
    min_tool_uses: int = 2
    reflections: list[str] = field(default_factory=_new_str_list)

    def _format_reflections(self) -> str:
        if not self.reflections:
            return ""
        items = "\n".join(f"- {item}" for item in self.reflections)
        return f"Reflexoes anteriores:\n{items}\n\n"

    def _build_trial_prompt(
        self,
        trial_index: int,
        scratchpad: list[str],
        used_tools: set[str],
        force_python_eval: bool,
        force_final: bool,
    ) -> str:
        recent_entries = scratchpad[-TRIAL_HISTORY_MAX_ENTRIES:]
        trimmed_entries = trim_history_entries(recent_entries, TRIAL_HISTORY_ENTRY_MAX_CHARS)
        history_text = "\n".join(trimmed_entries)
        used_tools_text = ", ".join(sorted(used_tools)) if used_tools else "nenhuma"

        flow_instruction = ""
        if force_python_eval:
            flow_instruction = (
                "\nEstado do fluxo: voce ja usou south_america_gdp_analysis.\n"
                "Resposta obrigatoria neste turno: ACTION: python_eval <expressao>\n"
                "Nao repita south_america_gdp_analysis agora.\n"
            )
        elif force_final:
            flow_instruction = (
                "\nEstado do fluxo: voce ja usou south_america_gdp_analysis e python_eval.\n"
                "Resposta obrigatoria neste turno: FINAL: <resposta conclusiva>\n"
                "Nao chame ACTION novamente.\n"
            )

        return (
            "Voce e um agente Reflexion baseado em ReAct.\n"
            f"Tentativa atual: {trial_index}/{self.max_trials}.\n"
            "Responda em portugues.\n\n"
            "Pergunta do usuario:\n"
            f"{self.question}\n\n"
            f"{self._format_reflections()}"
            "Historico da tentativa atual:\n"
            f"{history_text}\n\n"
            "Ferramentas ja usadas nesta tentativa:\n"
            f"{used_tools_text}\n\n"
            "Instrucoes obrigatorias:\n"
            "1. Siga o padrao THOUGHT -> ACTION/FINAL.\n"
            "2. Use ferramentas para buscar/calcular, sem inventar dados.\n"
            "3. Antes de FINAL, use no minimo duas ferramentas.\n"
            "4. Se a tentativa falhar, a proxima usara reflexao para melhorar.\n\n"
            "Ferramentas:\n"
            "- wikipedia_search <termo>\n"
            "- south_america_gdp_analysis <contexto-livre>\n"
            "- python_eval <expressao>\n\n"
            "Regra para tarefas de PIB da America do Sul:\n"
            "- Use south_america_gdp_analysis no inicio.\n"
            "- Use python_eval para validar a media antes do FINAL.\n\n"
            f"{flow_instruction}"
            "Formato valido:\n"
            "THOUGHT: ...\n"
            "ACTION: <ferramenta> <argumentos>\n"
            "ou\n"
            "FINAL: <resposta final>\n"
        )

    def _build_reflection_prompt(self, trial_result: TrialResult, trial_index: int) -> str:
        last_attempt = "\n".join(trial_result.scratchpad)
        trimmed_last_attempt = trim_history_entries([last_attempt], 2400)[0]

        previous_reflections = "\n".join(f"- {item}" for item in self.reflections)
        if previous_reflections:
            previous_reflections = f"Reflexoes anteriores:\n{previous_reflections}\n\n"

        return (
            "Voce e um avaliador reflexivo para um agente ReAct.\n"
            f"A tentativa {trial_index} falhou.\n"
            "Analise o historico e produza uma reflexao curta e acionavel para a proxima tentativa.\n\n"
            "Pergunta original:\n"
            f"{self.question}\n\n"
            f"{previous_reflections}"
            "Ultima tentativa:\n"
            f"{trimmed_last_attempt}\n\n"
            "Instrucoes:\n"
            "- Identifique o erro principal da tentativa.\n"
            "- Proponha a proxima acao concreta (busca, calculo, verificacao).\n"
            "- Escreva em no maximo 3 linhas.\n"
            "- Nao repita a resposta final; foque em melhorar o processo.\n"
        )

    def _run_single_trial(self, trial_index: int) -> TrialResult:
        trial_result = TrialResult()
        used_tools: set[str] = set()

        for step in range(1, self.max_steps + 1):
            force_python_eval = (
                "south_america_gdp_analysis" in used_tools
                and "python_eval" not in used_tools
            )
            force_final = (
                trial_result.tool_actions_count >= self.min_tool_uses
                and "south_america_gdp_analysis" in used_tools
                and "python_eval" in used_tools
            )

            prompt = self._build_trial_prompt(
                trial_index,
                trial_result.scratchpad,
                used_tools,
                force_python_eval,
                force_final,
            )
            output = self.client.send_request(prompt).strip()

            thought = extract_thought(output)
            if thought:
                log_status(f"Estou pensando em: {shorten_for_display(thought)}")

            trial_result.scratchpad.append(f"STEP {step} MODEL_OUTPUT: {output}")
            kind, payload = parse_agent_output(output)

            if kind == "final":
                if not payload.strip():
                    reminder = "Resposta FINAL vazia. Forneca uma conclusao objetiva e completa."
                    log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                    trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                    continue

                if trial_result.tool_actions_count < self.min_tool_uses:
                    reminder = "Resposta final prematura. Use pelo menos duas ferramentas antes de concluir."
                    log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                    trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                    continue

                if force_python_eval:
                    reminder = (
                        "Antes do FINAL, execute python_eval para validar a media com os valores coletados."
                    )
                    log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                    trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                    continue

                trial_result.answer = payload
                trial_result.finished = True
                log_status(f"Conclusao final: {payload}")
                return trial_result

            if kind == "invalid":
                reminder = (
                    "Saida fora do formato ReAct. Use THOUGHT e depois ACTION: <ferramenta> <args> "
                    "ou FINAL: <resposta>."
                )
                log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                continue

            if force_final:
                reminder = (
                    "Voce ja usou south_america_gdp_analysis e python_eval. "
                    "Agora responda somente com FINAL."
                )
                log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                continue

            action_name = payload.split(None, 1)[0].strip()
            action_parts = payload.split(None, 1)
            action_arg = action_parts[1].strip() if len(action_parts) > 1 else ""

            if force_python_eval and action_name != "python_eval":
                reminder = (
                    "Voce ja usou south_america_gdp_analysis nesta tentativa. "
                    "Proximo passo obrigatorio: ACTION: python_eval <expressao>."
                )
                log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                continue

            if action_name == "python_eval" and not action_arg:
                reminder = "python_eval requer uma expressao. Use: ACTION: python_eval (<expressao>)."
                log_status(f"Observacao da tentativa {trial_index}: {reminder}")
                trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {reminder}")
                continue

            log_status(f"Executando acao da tentativa {trial_index}, passo {step}: {payload}")
            observation, tool_used = execute_tool(payload)
            if tool_used:
                trial_result.tool_actions_count += 1
                used_tools.add(action_name)

            shortened_observation = shorten_for_display(observation, 320)
            log_status(f"Observacao da tentativa {trial_index}, passo {step}: {shortened_observation}")
            trial_result.scratchpad.append(f"STEP {step} OBSERVATION: {observation}")

        return trial_result

    def _reflect(self, trial_result: TrialResult, trial_index: int) -> None:
        if not trial_result.scratchpad:
            return

        prompt = self._build_reflection_prompt(trial_result, trial_index)
        reflection = self.client.send_request(prompt).strip()
        reflection_single_line = " ".join(reflection.split())
        if not reflection_single_line:
            return

        self.reflections.append(reflection_single_line)
        log_status(f"Reflexao registrada para proxima tentativa: {reflection_single_line}")

    def run(self) -> str:
        for trial_index in range(1, self.max_trials + 1):
            log_status(f"Iniciando tentativa {trial_index} de {self.max_trials}.")
            trial_result = self._run_single_trial(trial_index)

            if trial_result.finished and trial_result.answer:
                return trial_result.answer

            if trial_index < self.max_trials:
                log_status(
                    "Tentativa sem conclusao final valida. Iniciando etapa de reflexao para melhorar o proximo ciclo."
                )
                self._reflect(trial_result, trial_index)

        failure_message = "Nao foi possivel chegar a uma resposta final em numero suficiente de tentativas."
        log_status(failure_message)
        return failure_message


def run_reflexion_agent(
    question: str,
    api_key: str = "",
    max_trials: int = MAX_TRIALS_DEFAULT,
    max_steps: int = MAX_STEPS_DEFAULT,
    min_tool_uses: int = 2,
    client: Optional[FailoverLLMClient] = None,
) -> str:
    model_client = client or build_default_llm_client(
        gemini_api_key=api_key or None,
        gemini_model=DEFAULT_GEMINI_MODEL,
    )
    agent = ReflexionAgent(
        question=question,
        client=model_client,
        max_steps=max_steps,
        max_trials=max_trials,
        min_tool_uses=min_tool_uses,
    )
    return agent.run()


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("Uso: python reflexion_agent.py [pergunta em linguagem natural]")
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
        run_reflexion_agent(question, api_key=api_key or "", client=client)
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
