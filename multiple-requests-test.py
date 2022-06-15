import aiohttp
import asyncio

async def do_request(session):
    async with session.post("http://127.0.0.1:5000/", json={'prompt': "CHAPTER I\n\n", 'top_p': 0, 'top_k': 40, 'temp': 0.7, 'maximum_tokens': 64,}) as resp:
        return await resp.json()


async def main():
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[do_request(session) for _ in range(4)])
        print(results)

asyncio.run(main())
