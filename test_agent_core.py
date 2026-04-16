import unittest

import agent_core


class PrimaryLimitErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GeminiError(
            "Limite atingido.",
            technical_message="HTTP 429",
            fallback_eligible=True,
        )


class PrimaryNonLimitErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GeminiError(
            "Erro interno.",
            technical_message="HTTP 500",
            fallback_eligible=False,
        )


class SecondarySuccessClient:
    def __init__(self, response: str) -> None:
        self.calls = 0
        self.response = response

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class SecondaryErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GroqError("Groq indisponivel", technical_message="HTTP 503")


class FailoverClientTests(unittest.TestCase):
    def test_uses_groq_when_gemini_has_limit_error(self) -> None:
        primary = PrimaryLimitErrorClient()
        secondary = SecondarySuccessClient("resposta pelo fallback")
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        result = client.send_request("prompt")

        self.assertEqual(result, "resposta pelo fallback")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_does_not_use_groq_when_error_is_not_limit(self) -> None:
        primary = PrimaryNonLimitErrorClient()
        secondary = SecondarySuccessClient("nao deveria usar")
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        with self.assertRaises(agent_core.GeminiError):
            client.send_request("prompt")

        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 0)

    def test_returns_sanitized_error_if_groq_fallback_fails(self) -> None:
        primary = PrimaryLimitErrorClient()
        secondary = SecondaryErrorClient()
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        with self.assertRaises(agent_core.GeminiError) as ctx:
            client.send_request("prompt")

        self.assertIn("Groq indisponivel", str(ctx.exception))
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)


if __name__ == "__main__":
    unittest.main()
