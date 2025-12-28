# -*- coding: utf-8 -*-
import os
import asyncio
import uvicorn
import sdrbot
from portal.app import app

async def start_bot():
    token = sdrbot.CFG.get('token')
    if not token:
        raise SystemExit('[ERROR] No token found. Set DISCORD_CFG or DISCORD_TOKEN.')
    await sdrbot.bot.start(token)

async def main():
    server_config = uvicorn.Config(app, host=os.getenv('PORTAL_HOST','0.0.0.0'), port=int(os.getenv('PORTAL_PORT','8080')), log_level='info')
    server = uvicorn.Server(server_config)
    t_bot = asyncio.create_task(start_bot())
    t_web = asyncio.create_task(server.serve())
    done, pending = await asyncio.wait({t_bot, t_web}, return_when=asyncio.FIRST_EXCEPTION)
    for t in pending: t.cancel()

if __name__ == '__main__':
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
