# /// script
# dependencies = []
# ///
#
# Start the server first:
#
#   uv run --with 'msgflux[server,openai]' msgflux server examples/server_streaming_agent.py:registry --host 127.0.0.1
#
# Then run this streaming client:
#
#   uv run --with openai python examples/server_streaming_client.py
#
# The provider uses MSGFLUX_BASE_URL when set. Otherwise it defaults to:
#
#   http://127.0.0.1:8010/v1

import asyncio

import msgflux as mf


mf.load_dotenv()


async def stream_agent(model_path: str, message: str) -> None:
    model = mf.Model.chat_completion(
        model_path,
        variables={
            "tenant": "acme",
            "tier": "priority",
        },
    )
    response = await model.acall(
        [
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=True,
    )

    print(f"\n{model_path}: ", end="", flush=True)
    async for chunk in response.consume():
        print(chunk, end="", flush=True)
    print()


async def main() -> None:
    await stream_agent(
        "msgflux/support",
        "Meu pedido A1002 ainda nao chegou. O que aconteceu?",
    )
    await stream_agent(
        "msgflux/billing",
        "A fatura INV-44 falhou. O que devo fazer agora?",
    )


if __name__ == "__main__":
    asyncio.run(main())
