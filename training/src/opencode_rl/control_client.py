"""Private, bounded control plane; no generation retries or wire rewriting."""
import asyncio


class BoundedControlClientMixin:
    async def _call_server(self, server_url, endpoint, json=None, method='POST', params=None):
        import aiohttp
        from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import raise_for_status
        session = await self._get_session()
        # The engine drain itself allows 1200 s; short status/cache/control
        # operations must not inherit upstream's unlimited HTTP deadline.
        timeout = aiohttp.ClientTimeout(total=1230 if endpoint == '/ocrl/drain' else 180, sock_connect=10)
        async with session.request(method, server_url + endpoint, json=json, params=params, timeout=timeout) as response:
            raw = await response.read()
            body = await response.json() if raw else None  # also handles chunked JSON
            raise_for_status(response, body)
            return server_url, {'status': response.status, 'body': body}

    async def _call_all_servers(self, endpoint, json=None, method='POST', params=None):
        jobs = [asyncio.create_task(self._call_server(url, endpoint, json, method, params)) for url in self.server_urls]
        try:
            return dict(await asyncio.gather(*jobs))
        finally:
            # A failed fanout must not leave a late /open racing with /fail.
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)


def control_client(**kwargs):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient

    class OpenCodeControlClient(BoundedControlClientMixin, RemoteInferenceClient):
        pass

    return OpenCodeControlClient(**kwargs)
