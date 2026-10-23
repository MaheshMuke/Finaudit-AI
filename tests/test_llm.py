from finaudit.llm import GeminiProvider, OpenAIProvider, configured_provider_name


def test_openrouter_config_is_used_for_analysis_interpretation(monkeypatch):
    import openai

    class FakeCompletions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {"content": "Caveated interpretation."},
                                )()
                            },
                        )()
                    ]
                },
            )()

    fake_completions = FakeCompletions()
    fake_client = type(
        "Client",
        (),
        {
            "chat": type(
                "Chat", (), {"completions": fake_completions}
            )()
        },
    )()
    client_kwargs = {}

    def fake_openai(**kwargs):
        client_kwargs.update(kwargs)
        return fake_client

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    provider = OpenAIProvider()
    result = provider.interpret([], [])

    assert result == "Caveated interpretation."
    assert configured_provider_name() == "OpenRouter"
    assert provider.model_name == "nvidia/nemotron-3.5-lightning:free"
    assert client_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert fake_completions.kwargs["model"] == "nvidia/nemotron-3.5-lightning:free"


def test_gemini_provider_uses_generate_content_api(monkeypatch):
    import requests

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "candidates": [{
                    "content": {"parts": [{"text": "Gemini interpretation."}]}
                }]
            }

    calls = []
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, **kwargs: (calls.append((url, kwargs)) or FakeResponse()),
    )
    result = GeminiProvider().interpret([], [])
    assert result == "Gemini interpretation."
    assert calls[0][0].endswith("/gemini-2.5-flash-lite:generateContent")
    assert calls[0][1]["params"]["key"] == "test-gemini-key"
