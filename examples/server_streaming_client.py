# /// script
# dependencies = []
# ///

import asyncio

import msgflux as mf


mf.load_dotenv()


async def main() -> None:
    model = mf.Model.chat_completion(
        "msgflux/support",
        variables={
            "tenant": "acme",
            "tier": "priority",
        },
    )
    response = await model.acall(
        [
            {
                "role": "user",
                "content": "Meu pedido A1002 ainda nao chegou. O que aconteceu?",
            }
        ],
        stream=True,
    )

    print("assistant: ", end="", flush=True)
    async for chunk in response.consume():
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
