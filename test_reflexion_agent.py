import unittest
from typing import Sequence, cast
from unittest.mock import patch

import agent_core
import reflexion_agent


class FakeGeminiClient:
    def __init__(self, responses: Sequence[str]) -> None:
        self.responses: list[str] = list(responses)
        self.prompts: list[str] = []

    def send_request(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Resposta de teste nao configurada para o cliente fake.")
        return self.responses.pop(0)


class ReflexionAgentTests(unittest.TestCase):
    def test_reflexion_uses_reflection_between_trials(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: vou buscar\nACTION: wikipedia_search Brasil",
            "THOUGHT: ja sei\nFINAL: resposta precoce",
            "THOUGHT: sigo sem dados\nFINAL: resposta ainda precoce",
            "Buscar tambem os outros paises antes de concluir.",
            "THOUGHT: buscar novamente\nACTION: wikipedia_search Argentina",
            "THOUGHT: calcular media\nACTION: python_eval (10+20+30)/3",
            "THOUGHT: agora sim\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "reflexion_agent.execute_tool",
            side_effect=[("obs1", True), ("obs2", True), ("obs3", True)],
        ) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=2,
                max_steps=3,
                min_tool_uses=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 3)
        self.assertEqual(len(client.prompts), 7)
        self.assertTrue(any("avaliador reflexivo" in prompt for prompt in client.prompts))
        self.assertTrue(any("Reflexoes anteriores" in prompt for prompt in client.prompts))

    def test_reflexion_returns_failure_after_max_trials(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: sem acao valida",
            "THOUGHT: ainda sem formato",
            "Use uma ferramenta antes de concluir.",
            "THOUGHT: continuo sem acao",
            "THOUGHT: nao concluido",
        ]
        client = FakeGeminiClient(model_responses)

        with patch("reflexion_agent.execute_tool", return_value=("obs", True)) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=2,
                max_steps=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(
            answer,
            "Nao foi possivel chegar a uma resposta final em numero suficiente de tentativas.",
        )
        self.assertEqual(mocked_tool.call_count, 0)

    def test_reflexion_forces_python_eval_after_south_america_tool(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: buscar dados\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: repetir sem necessidade\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: validar media\nACTION: python_eval (10+20+30)/3",
            "THOUGHT: concluir\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "reflexion_agent.execute_tool",
            side_effect=[("obs gdp", True), ("20.0", True)],
        ) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=1,
                max_steps=4,
                min_tool_uses=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 2)
        self.assertTrue(
            any(
                "Resposta obrigatoria neste turno: ACTION: python_eval <expressao>" in prompt
                for prompt in client.prompts
            )
        )

    def test_reflexion_forces_final_after_required_tools(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: buscar dados\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: validar media\nACTION: python_eval (10+20+30)/3",
            "THOUGHT: vou repetir\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: concluir\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "reflexion_agent.execute_tool",
            side_effect=[("obs gdp", True), ("20.0", True)],
        ) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=1,
                max_steps=4,
                min_tool_uses=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 2)
        self.assertTrue(
            any(
                "Resposta obrigatoria neste turno: FINAL: <resposta conclusiva>" in prompt
                for prompt in client.prompts
            )
        )

    def test_reflexion_ignores_python_eval_without_expression(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: buscar dados\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: validar media\nACTION: python_eval",
            "THOUGHT: validar media direito\nACTION: python_eval (10+20+30)/3",
            "THOUGHT: concluir\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "reflexion_agent.execute_tool",
            side_effect=[("obs gdp", True), ("20.0", True)],
        ) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=1,
                max_steps=4,
                min_tool_uses=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 2)

    def test_reflexion_rejects_empty_final_payload(self) -> None:
        model_responses: list[str] = [
            "THOUGHT: buscar dados\nACTION: south_america_gdp_analysis dados",
            "THOUGHT: validar media\nACTION: python_eval (10+20+30)/3",
            "THOUGHT: concluir\nFINAL:",
            "THOUGHT: concluir melhor\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "reflexion_agent.execute_tool",
            side_effect=[("obs gdp", True), ("20.0", True)],
        ) as mocked_tool:
            answer = reflexion_agent.run_reflexion_agent(
                question="Pergunta",
                api_key="fake",
                max_trials=1,
                max_steps=4,
                min_tool_uses=2,
                client=cast(agent_core.FailoverLLMClient, client),
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 2)

if __name__ == "__main__":
    unittest.main()
