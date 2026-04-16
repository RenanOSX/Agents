import unittest
from unittest.mock import patch

import react_agent


class FakeGeminiClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def send_request(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("Resposta de teste nao configurada para o cliente fake.")
        return self.responses.pop(0)


class ParseAgentOutputTests(unittest.TestCase):
    def test_parse_action_in_multiline_output(self) -> None:
        kind, payload = react_agent.parse_agent_output(
            "THOUGHT: vou calcular\nACTION: python_eval 2+2"
        )

        self.assertEqual(kind, "action")
        self.assertEqual(payload, "python_eval 2+2")

    def test_parse_portuguese_action_label(self) -> None:
        kind, payload = react_agent.parse_agent_output("AÇÃO: python_eval 7*6")

        self.assertEqual(kind, "action")
        self.assertEqual(payload, "python_eval 7*6")

    def test_parse_invalid_output(self) -> None:
        kind, payload = react_agent.parse_agent_output("Sem formato esperado")

        self.assertEqual(kind, "invalid")
        self.assertEqual(payload, "Sem formato esperado")


class RetryAndQuotaTests(unittest.TestCase):
    def test_parse_retry_delay_from_ms_message(self) -> None:
        error_body = (
            '{"error":{"message":"Quota exceeded. Please retry in 375.437126ms.","details":[]}}'
        )
        delay = react_agent.parse_retry_delay_seconds(error_body)

        self.assertIsNotNone(delay)
        assert delay is not None
        self.assertGreater(delay, 0.0)
        self.assertLess(delay, 1.0)

    def test_parse_retry_delay_from_minutes_message(self) -> None:
        error_body = '{"error":{"message":"Please retry in 2m","details":[]}}'
        delay = react_agent.parse_retry_delay_seconds(error_body)

        self.assertEqual(delay, 120.0)

    def test_detects_quota_exceeded_error(self) -> None:
        error_body = (
            '{"error":{"message":"You exceeded your current quota, please check your plan and billing details. '
            'Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests"}}'
        )
        self.assertTrue(react_agent.is_quota_exceeded_error(error_body))


class ExecuteToolTests(unittest.TestCase):
    def test_execute_tool_rejects_empty_action(self) -> None:
        observation, used = react_agent.execute_tool("   ")

        self.assertFalse(used)
        self.assertIn("Acao invalida", observation)


class RunAgentTests(unittest.TestCase):
    def test_requires_two_tool_uses_before_final(self) -> None:
        model_responses = [
            "THOUGHT: ja sei\nFINAL: resposta precoce",
            "THOUGHT: buscar contexto\nACTION: wikipedia_search Brasil",
            "THOUGHT: validar conta\nACTION: python_eval 20+22",
            "THOUGHT: concluir\nFINAL: resposta correta",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "react_agent.execute_tool",
            side_effect=[("obs wiki", True), ("obs calc", True)],
        ) as mocked_tool:
            answer = react_agent.run_agent("Pergunta", api_key="fake", max_turns=4, client=client)

        self.assertEqual(answer, "resposta correta")
        self.assertEqual(mocked_tool.call_count, 2)
        self.assertEqual(len(client.prompts), 4)

    def test_invalid_step_does_not_count_as_tool_usage(self) -> None:
        model_responses = [
            "THOUGHT: primeira saida sem acao",
            "THOUGHT: buscar\nACTION: wikipedia_search Terra",
            "THOUGHT: calcular\nACTION: python_eval 1+1",
            "THOUGHT: concluir\nFINAL: pronto",
        ]
        client = FakeGeminiClient(model_responses)

        with patch(
            "react_agent.execute_tool",
            side_effect=[("obs wiki", True), ("obs calc", True)],
        ) as mocked_tool:
            answer = react_agent.run_agent("Pergunta", api_key="fake", max_turns=4, client=client)

        self.assertEqual(answer, "pronto")
        self.assertEqual(mocked_tool.call_count, 2)

    def test_rejects_final_with_only_one_tool_even_on_last_turn(self) -> None:
        model_responses = [
            "THOUGHT: buscar\nACTION: wikipedia_search Brasil",
            "THOUGHT: ja posso finalizar\nFINAL: resposta precoce",
        ]
        client = FakeGeminiClient(model_responses)

        with patch("react_agent.execute_tool", return_value=("obs wiki", True)) as mocked_tool:
            answer = react_agent.run_agent("Pergunta", api_key="fake", max_turns=2, client=client)

        self.assertEqual(
            answer,
            "Nao foi possivel chegar a uma resposta final em numero suficiente de passos.",
        )
        self.assertEqual(mocked_tool.call_count, 1)


if __name__ == "__main__":
    unittest.main()
