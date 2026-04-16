import unittest
from unittest.mock import patch

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


class PrimaryGroqLimitErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GroqError(
            "Limite Groq atingido",
            technical_message="HTTP 429",
            fallback_eligible=True,
            rate_limited=True,
        )


class SecondaryGeminiLimitErrorClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GeminiError(
            "Limite Gemini atingido",
            technical_message="HTTP 429",
            fallback_eligible=False,
            rate_limited=True,
        )


class OpenRouterSuccessClient:
    def __init__(self, response: str) -> None:
        self.calls = 0
        self.response = response

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        return self.response


class AlwaysUnavailableGroqClient:
    def __init__(self) -> None:
        self.calls = 0

    def send_request(self, prompt: str) -> str:
        self.calls += 1
        raise agent_core.GroqError(
            "Groq indisponivel",
            technical_message="HTTP 503",
            fallback_eligible=True,
        )


class FailoverClientTests(unittest.TestCase):
    def test_uses_groq_when_gemini_has_limit_error(self) -> None:
        primary = PrimaryLimitErrorClient()
        secondary = SecondarySuccessClient("resposta pelo fallback")
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        result = client.send_request("prompt")

        self.assertEqual(result, "resposta pelo fallback")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_uses_next_provider_even_when_primary_error_is_not_limit(self) -> None:
        primary = PrimaryNonLimitErrorClient()
        secondary = SecondarySuccessClient("resposta do secundario")
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        result = client.send_request("prompt")

        self.assertEqual(result, "resposta do secundario")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_returns_sanitized_error_if_groq_fallback_fails(self) -> None:
        primary = PrimaryLimitErrorClient()
        secondary = SecondaryErrorClient()
        client = agent_core.FailoverLLMClient(primary_client=primary, fallback_client=secondary)

        with self.assertRaises(agent_core.GeminiError) as ctx:
            client.send_request("prompt")

        self.assertIn("Groq indisponivel", str(ctx.exception))
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_warns_when_both_platforms_hit_rate_limit(self) -> None:
        primary = PrimaryGroqLimitErrorClient()
        secondary = SecondaryGeminiLimitErrorClient()
        client = agent_core.FailoverLLMClient(
            primary_client=primary,
            fallback_client=secondary,
            primary_name="Groq",
            fallback_name="Gemini",
        )

        with self.assertRaises(agent_core.GeminiError) as ctx:
            client.send_request("prompt")

        self.assertIn("limite das plataformas disponiveis", str(ctx.exception).lower())
        self.assertEqual(primary.calls, 1)
        self.assertEqual(secondary.calls, 1)

    def test_tries_next_provider_in_sequence(self) -> None:
        primary = PrimaryLimitErrorClient()
        openrouter = OpenRouterSuccessClient("resposta do openrouter")
        gemini = SecondarySuccessClient("resposta do gemini")
        client = agent_core.FailoverLLMClient(
            provider_chain=[
                ("Groq", primary),
                ("OpenRouter", openrouter),
                ("Gemini", gemini),
            ]
        )

        result = client.send_request("prompt")

        self.assertEqual(result, "resposta do openrouter")
        self.assertEqual(primary.calls, 1)
        self.assertEqual(openrouter.calls, 1)
        self.assertEqual(gemini.calls, 0)

    def test_sticks_to_last_successful_provider_across_requests(self) -> None:
        groq = AlwaysUnavailableGroqClient()
        mistral = SecondarySuccessClient("resposta mistral")
        gemini = SecondarySuccessClient("resposta gemini")
        client = agent_core.FailoverLLMClient(
            provider_chain=[
                ("Groq", groq),
                ("Mistral", mistral),
                ("Gemini", gemini),
            ]
        )

        first = client.send_request("primeiro prompt")
        second = client.send_request("segundo prompt")

        self.assertEqual(first, "resposta mistral")
        self.assertEqual(second, "resposta mistral")
        self.assertEqual(groq.calls, 1)
        self.assertEqual(mistral.calls, 2)
        self.assertEqual(gemini.calls, 0)


class DefaultClientBuilderTests(unittest.TestCase):
    def test_uses_groq_as_default_primary_when_both_keys_exist(self) -> None:
        client = agent_core.build_default_llm_client(
            gemini_api_key="gemini-key",
            groq_api_key="groq-key",
            open_router_api_key="",
            mistral_api_key="",
        )

        self.assertEqual(client.primary_name, "Groq")
        self.assertEqual(client.fallback_name, "Gemini")
        self.assertIsInstance(client.primary_client, agent_core.GroqClient)
        self.assertIsInstance(client.fallback_client, agent_core.GeminiClient)

    def test_builds_full_provider_sequence_when_all_keys_exist(self) -> None:
        client = agent_core.build_default_llm_client(
            groq_api_key="groq-key",
            open_router_api_key="openrouter-key",
            mistral_api_key="mistral-key",
            gemini_api_key="gemini-key",
        )

        names = [name for name, _ in client.provider_chain]
        self.assertEqual(names, ["Groq", "OpenRouter", "Mistral", "Gemini"])


class GeminiClientResilienceTests(unittest.TestCase):
    def test_timeout_error_is_sanitized(self) -> None:
        client = agent_core.GeminiClient(api_key="fake-key")

        with patch("agent_core.urllib.request.urlopen", side_effect=TimeoutError("The read operation timed out")):
            with self.assertRaises(agent_core.GeminiError) as ctx:
                client.send_request("prompt")

        self.assertIn("tempo limite", str(ctx.exception).lower())


class WikipediaSearchFallbackTests(unittest.TestCase):
    def test_uses_search_fallback_when_exact_title_fails(self) -> None:
        direct_without_extract = {"query": {"pages": {"-1": {"missing": ""}}}}
        search_result = {"query": {"search": [{"title": "Produto Interno Bruto"}]}}
        fallback_with_extract = {
            "query": {
                "pages": {
                    "123": {
                        "extract": "Resumo encontrado via busca de relevancia.",
                    }
                }
            }
        }

        with patch(
            "agent_core._wikipedia_query_json",
            side_effect=[direct_without_extract, search_result, fallback_with_extract],
        ) as mocked_query:
            result = agent_core.wikipedia_search("Lista de paises da America do Sul por PIB nominal")

        self.assertIn("Resumo encontrado via busca de relevancia.", result)
        self.assertEqual(mocked_query.call_count, 3)


class ToolBehaviorTests(unittest.TestCase):
    def test_execute_tool_strips_wrapping_quotes(self) -> None:
        mocked_tools = {"wikipedia_search": lambda term: f"term={term}"}
        observation, used = agent_core.execute_tool(
            'wikipedia_search "PIB America do Sul"',
            tools=mocked_tools,
        )

        self.assertTrue(used)
        self.assertEqual(observation, "term=PIB America do Sul")

    def test_south_america_gdp_analysis_tool_is_registered(self) -> None:
        self.assertIn("south_america_gdp_analysis", agent_core.TOOLS)


class WorldBankParsingTests(unittest.TestCase):
    def test_world_bank_uses_semicolon_country_separator(self) -> None:
        with patch("agent_core._world_bank_fetch_json", return_value={"rows": []}) as mocked_fetch:
            agent_core._world_bank_latest_values("ARG,BRA,CHL", "NY.GDP.MKTP.CD")

        called_url = mocked_fetch.call_args[0][0]
        self.assertIn("/country/ARG;BRA;CHL/", called_url)

    def test_world_bank_maps_world_alias_to_wld(self) -> None:
        rows = [
            {
                "date": "2023",
                "value": 12500.0,
                "country": {"id": "1W"},
                "countryiso3code": "",
            }
        ]
        with patch("agent_core._world_bank_fetch_json", return_value={"rows": rows}):
            values = agent_core._world_bank_latest_values("WLD", "NY.GDP.PCAP.CD")

        self.assertIn("WLD", values)
        self.assertEqual(values["WLD"], (2023, 12500.0))


if __name__ == "__main__":
    unittest.main()
