import httpx
import pytest


@pytest.fixture
def sdk_transport():
    """Bridge SDK HTTPX requests to either Starlette TestClient transport implementation."""

    def create(api):
        def forward(request):
            response = api.request(
                request.method,
                request.url.path,
                params=list(request.url.params.multi_items()),
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx.Response(
                response.status_code, headers=dict(response.headers), content=response.content
            )

        return httpx.MockTransport(forward)

    return create
