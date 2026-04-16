import unittest
from unittest.mock import patch

import reflexion_agent


class FakeGeminiClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def send_request(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Resposta de teste nao configurada para o cliente fake.")
        return self.responses.pop(0)


class ReflexionAgentTests(unittest.TestCase):
    def test_reflexion_uses_reflection_between_trials(self) -> None:
        model_responses = [
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
                client=client,
            )

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 3)
        self.assertEqual(len(client.prompts), 7)
        self.assertTrue(any("avaliador reflexivo" in prompt for prompt in client.prompts))
        self.assertTrue(any("Reflexoes anteriores" in prompt for prompt in client.prompts))

    def test_reflexion_returns_failure_after_max_trials(self) -> None:
        model_responses = [
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
                client=client,
            )

        self.assertEqual(
            answer,
            "Nao foi possivel chegar a uma resposta final em numero suficiente de tentativas.",
        )
        self.assertEqual(mocked_tool.call_count, 0)


if __name__ == "__main__":
    unittest.main()
