from collections.abc import AsyncGenerator, Generator
from functools import partial, wraps
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient


def noop_wrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def wraps_of(func):
    # functools.wraps around a callable without __name__ (e.g. a partial)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def sub_dependency() -> str:
    return "sub"


def sync_dependency(value: str, sub: Annotated[str, Depends(sub_dependency)]) -> str:
    return f"{value}-{sub}"


async def async_dependency(
    value: str, sub: Annotated[str, Depends(sub_dependency)]
) -> str:
    return f"{value}-{sub}"


def gen_dependency(
    value: str, sub: Annotated[str, Depends(sub_dependency)]
) -> Generator[str, None, None]:
    yield f"{value}-{sub}"


async def async_gen_dependency(
    value: str, sub: Annotated[str, Depends(sub_dependency)]
) -> AsyncGenerator[str, None]:
    yield f"{value}-{sub}"


class ClassDependency:
    def __init__(self, value: str, sub: Annotated[str, Depends(sub_dependency)]):
        self.result = f"{value}-{sub}"


class SyncCallableDependency:
    def __call__(self, value: str, sub: Annotated[str, Depends(sub_dependency)]) -> str:
        return f"{value}-{sub}"


class AsyncCallableDependency:
    async def __call__(
        self, value: str, sub: Annotated[str, Depends(sub_dependency)]
    ) -> str:
        return f"{value}-{sub}"


class GenCallableDependency:
    def __call__(
        self, value: str, sub: Annotated[str, Depends(sub_dependency)]
    ) -> Generator[str, None, None]:
        yield f"{value}-{sub}"


class AsyncGenCallableDependency:
    async def __call__(
        self, value: str, sub: Annotated[str, Depends(sub_dependency)]
    ) -> AsyncGenerator[str, None]:
        yield f"{value}-{sub}"


def endpoint_result(item_id: int, value: str) -> dict[str, Any]:
    return {"item_id": item_id, "value": value}


app = FastAPI()

# Dependencies combining functools.wraps and functools.partial in any order

dependency_variants = {
    "sync": sync_dependency,
    "async": async_dependency,
    "gen": gen_dependency,
    "async-gen": async_gen_dependency,
    "class": ClassDependency,
    "sync-callable": SyncCallableDependency(),
    "async-callable": AsyncCallableDependency(),
    "gen-callable": GenCallableDependency(),
    "async-gen-callable": AsyncGenCallableDependency(),
}

for name, dep in dependency_variants.items():
    variants = {
        "partial": partial(dep),
        "partial-partial": partial(partial(dep)),
        "wraps": noop_wrap(dep),
        "partial-wraps": partial(noop_wrap(dep)),
        "wraps-partial": wraps_of(partial(dep)),
        "wraps-partial-wraps": wraps_of(partial(noop_wrap(dep))),
        "partial-wraps-partial": partial(wraps_of(partial(dep))),
    }
    for variant_name, variant in variants.items():
        route = f"/{variant_name}-{name}/"

        async def endpoint(
            value: Annotated[Any, Depends(variant)],
        ) -> str:
            if isinstance(value, ClassDependency):
                return value.result
            return value

        app.get(route)(endpoint)


# Path operations (endpoints) combining functools.wraps and functools.partial


async def async_endpoint(item_id: int, q: str = "default") -> dict[str, Any]:
    return endpoint_result(item_id, q)


def sync_endpoint(item_id: int, q: str = "default") -> dict[str, Any]:
    return endpoint_result(item_id, q)


class EndpointHandler:
    async def __call__(self, item_id: int, q: str = "default") -> dict[str, Any]:
        return endpoint_result(item_id, q)


endpoint_variants = {
    "async": async_endpoint,
    "sync": sync_endpoint,
    "callable": EndpointHandler(),
}

for name, endpoint in endpoint_variants.items():
    variants = {
        "partial": partial(endpoint),
        "wraps": noop_wrap(endpoint),
        "partial-wraps": partial(noop_wrap(endpoint)),
        "wraps-partial": wraps_of(partial(endpoint)),
    }
    for variant_name, variant in variants.items():
        app.get(f"/endpoint-{variant_name}-{name}/{{item_id}}")(variant)


client = TestClient(app)


@pytest.mark.parametrize(
    "route",
    [
        f"/{variant}-{name}/"
        for variant in [
            "partial",
            "partial-partial",
            "wraps",
            "partial-wraps",
            "wraps-partial",
            "wraps-partial-wraps",
            "partial-wraps-partial",
        ]
        for name in [
            "sync",
            "async",
            "gen",
            "async-gen",
            "class",
            "sync-callable",
            "async-callable",
            "gen-callable",
            "async-gen-callable",
        ]
    ],
)
def test_wrapped_partial_dependency(route: str) -> None:
    response = client.get(route, params={"value": "hello"})
    assert response.status_code == 200, response.text
    assert response.json() == "hello-sub"


@pytest.mark.parametrize(
    "route",
    [
        f"/endpoint-{variant}-{name}/"
        for variant in ["partial", "wraps", "partial-wraps", "wraps-partial"]
        for name in ["async", "sync", "callable"]
    ],
)
def test_wrapped_partial_endpoint(route: str) -> None:
    response = client.get(f"{route}5", params={"q": "hello"})
    assert response.status_code == 200, response.text
    assert response.json() == {"item_id": 5, "value": "hello"}
    response = client.get(f"{route}42")
    assert response.status_code == 200, response.text
    assert response.json() == {"item_id": 42, "value": "default"}
