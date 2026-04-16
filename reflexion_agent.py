#!/usr/bin/env python3
from dataclasses import dataclass, field
import sys
from typing import List, Optional

from agent_core import (
    API_ENV_KEY,
    DEFAULT_GEMINI_MODEL,
    GROQ_API_ENV_KEY,
    FailoverLLMClient,
    GeminiError,
    build_default_llm_client,
    execute_tool,
    extract_thought,
    load_api_key,
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


@dataclass
class TrialResult:
    scratchpad: List[str] = field(default_factory=list)
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
    reflections: List[str] = field(default_factory=list)

    def _format_reflections(self) -> str:
        if not self.reflections:
            return ""
        items = "\n".join(f"- {item}" for item in self.reflections)
        return f"Reflexoes anteriores:\n{items}\n\n"

    def _build_trial_prompt(self, trial_index: int, scratchpad: List[str]) -> str:
        recent_entries = scratchpad[-TRIAL_HISTORY_MAX_ENTRIES:]
        trimmed_entries = trim_history_entries(recent_entries, TRIAL_HISTORY_ENTRY_MAX_CHARS)
        history_text = "\n".join(trimmed_entries)

        return (
            "Voce e um agente Reflexion baseado em ReAct.\n"
            f"Tentativa atual: {trial_index}/{self.max_trials}.\n"
            "Responda em portugues.\n\n"
            "Pergunta do usuario:\n"
            f"{self.question}\n\n"
            f"{self._format_reflections()}"
            "Historico da tentativa atual:\n"
            f"{history_text}\n\n"
            "Instrucoes obrigatorias:\n"
            "1. Siga o padrao THOUGHT -> ACTION/FINAL.\n"
            "2. Use ferramentas para buscar/calcular, sem inventar dados.\n"
            "3. Antes de FINAL, use no minimo duas ferramentas.\n"
            "4. Se a tentativa falhar, a proxima usara reflexao para melhorar.\n\n"
            "Ferramentas:\n"
            "- wikipedia_search <termo>\n"
            "- python_eval <expressao>\n\n"
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

        for step in range(1, self.max_steps + 1):
            prompt = self._build_trial_prompt(trial_index, trial_result.scratchpad)
            output = self.client.send_request(prompt).strip()

            thought = extract_thought(output)
            if thought:
                log_status(f"Estou pensando em: {shorten_for_display(thought)}")

            trial_result.scratchpad.append(f"STEP {step} MODEL_OUTPUT: {output}")
            kind, payload = parse_agent_output(output)

            if kind == "final":
                if trial_result.tool_actions_count < self.min_tool_uses:
                    reminder = "Resposta final prematura. Use pelo menos duas ferramentas antes de concluir."
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

            log_status(f"Executando acao da tentativa {trial_index}, passo {step}: {payload}")
            observation, tool_used = execute_tool(payload)
            if tool_used:
                trial_result.tool_actions_count += 1

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
    api_key: str,
    max_trials: int = MAX_TRIALS_DEFAULT,
    max_steps: int = MAX_STEPS_DEFAULT,
    min_tool_uses: int = 2,
    client: Optional[FailoverLLMClient] = None,
) -> str:
    model_client = client or build_default_llm_client(
        gemini_api_key=api_key,
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


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] in ("-h", "--help"):
        print("Uso: python reflexion_agent.py [pergunta em linguagem natural]")
        print("Se nenhuma pergunta for passada, o script pedira a entrada via teclado.")
        return 0

    if len(argv) > 1:
        question = " ".join(argv[1:]).strip()
    else:
        question = input("Pergunta ou tarefa: ").strip()

    if not question:
        print("[Gemini]: Pergunta nao fornecida.")
        return 1

    try:
        api_key = load_api_key(API_ENV_KEY)
        groq_api_key = load_optional_api_key(GROQ_API_ENV_KEY)
        client = build_default_llm_client(
            gemini_api_key=api_key,
            groq_api_key=groq_api_key,
            gemini_model=DEFAULT_GEMINI_MODEL,
        )
        answer = run_reflexion_agent(question, api_key, client=client)
        print(f"[Gemini]: Conclusao final: {answer}")
        return 0
    except GeminiError as exc:
        print(f"[Gemini]: {exc}")
        if exc.technical_message:
            print(f"[Gemini][debug]: {exc.technical_message}", file=sys.stderr)
        return 1
    except Exception as exc:
        print("[Gemini]: Ocorreu um erro interno inesperado. Tente novamente em instantes.")
        print(f"[Gemini][debug]: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
