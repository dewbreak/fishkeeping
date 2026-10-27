"""HTTP boundary. Deploy behind an identity-aware gateway for multi-user access."""

from __future__ import annotations

import secrets
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from fishkeeping import __version__
from fishkeeping.config import Settings
from fishkeeping.embedding_factory import create_embeddings
from fishkeeping.jobs import EventQueue
from fishkeeping.models import (
    ConsolidateRequest,
    ConsolidateResponse,
    ContextRequest,
    ContextResponse,
    DeleteResponse,
    EventRequest,
    EventResponse,
    IndexRequest,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    JobInfo,
    MaintenanceRequest,
    MaintenanceResponse,
    Memory,
    MemoryInput,
    RecallRequest,
    RecallResponse,
    Scope,
    SourcePage,
    StatsResponse,
)
from fishkeeping.providers import DeepSeekProvider, ProviderError
from fishkeeping.service import MemoryService, NotConfiguredError
from fishkeeping.storage import ConflictError, InvalidSourceError, NotFoundError, Store
from fishkeeping.video import VideoBusyError, VideoProcessor
from fishkeeping.video_events import enqueue_video
from fishkeeping.video_models import VideoEventRequest, VideoEvidence, VideoRequest
from fishkeeping.vision import ImageInputError, ImageProcessor
from fishkeeping.vision_events import enqueue_image
from fishkeeping.vision_models import (
    ImageEventRequest,
    ImageEventResponse,
    ImageEvidence,
    ImageRequest,
)


def create_app(settings: Settings | None = None, service: MemoryService | None = None) -> FastAPI:
    settings = settings or Settings()
    images = ImageProcessor()
    videos = VideoProcessor()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is not None:
            app.state.service = service
            app.state.queue = EventQueue(service.store, settings)
            yield
            return
        store = Store(settings.db_path)
        llm = None
        embeddings = None
        try:
            llm = DeepSeekProvider(settings) if settings.deepseek_api_key else None
            embeddings = create_embeddings(settings)
            app.state.service = MemoryService(store, settings, llm=llm, embeddings=embeddings)
            app.state.queue = EventQueue(store, settings)
            yield
        finally:
            if embeddings is not None:
                embeddings.close()
            if llm is not None:
                llm.close()
            store.close()

    app = FastAPI(title="Fishkeeping Memory API", version=__version__, lifespan=lifespan)
    app.state.images = images
    app.state.videos = videos

    def current_service(request: Request) -> MemoryService:
        return request.app.state.service

    def current_queue(request: Request) -> EventQueue:
        return request.app.state.queue

    def principal(x_api_key: Annotated[str | None, Header()] = None) -> str | None:
        if not settings.api_keys:
            return None
        if x_api_key is not None:
            for expected, namespace in settings.api_keys.items():
                if secrets.compare_digest(x_api_key.encode(), expected.encode()):
                    return namespace
        raise HTTPException(401, "Valid X-API-Key required")

    def authorize(scope: Scope, namespace: str | None) -> None:
        if namespace is not None and scope.namespace != namespace:
            raise HTTPException(403, "API key is not authorized for this namespace")

    def query_scope(
        user_id: Annotated[str, Query(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")],
        namespace: Annotated[
            str, Query(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")
        ] = "default",
        agent_id: Annotated[
            str, Query(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")
        ] = "default",
    ) -> Scope:
        return Scope(namespace=namespace, user_id=user_id, agent_id=agent_id)

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(InvalidSourceError)
    async def source_handler(request: Request, exc: InvalidSourceError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(ImageInputError)
    async def image_handler(request: Request, exc: ImageInputError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(VideoBusyError)
    async def video_busy_handler(request: Request, exc: VideoBusyError):
        return JSONResponse(status_code=429, content={"detail": "Local video decoder is busy"})

    @app.exception_handler(NotConfiguredError)
    async def config_handler(request: Request, exc: NotConfiguredError):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(ProviderError)
    async def provider_handler(request: Request, exc: ProviderError):
        return JSONResponse(
            status_code=502,
            content={"detail": "Model provider request failed", "code": exc.code},
        )

    @app.get("/healthz")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def ready(svc: MemoryService = Depends(current_service)):
        svc.store.stats(Scope(user_id="_readiness"))
        return {"status": "ready"}

    @app.post("/v1/memories", response_model=Memory, status_code=201)
    def remember(
        body: MemoryInput,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.store.put(body)

    @app.get("/v1/memories", response_model=list[Memory])
    def list_memories(
        scope: Scope = Depends(query_scope),
        include_inactive: bool = False,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(scope, namespace)
        return svc.store.list(scope, include_inactive=include_inactive, limit=limit, offset=offset)

    @app.get("/v1/memories/{memory_id}", response_model=Memory)
    def get_memory(
        memory_id: str,
        scope: Scope = Depends(query_scope),
        include_inactive: bool = False,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(scope, namespace)
        memory = svc.store.get(scope, memory_id, include_inactive=include_inactive)
        if memory is None:
            raise HTTPException(404, "Memory not found")
        return memory

    @app.get("/v1/memories/{memory_id}/sources", response_model=SourcePage)
    def sources(
        memory_id: str,
        scope: Scope = Depends(query_scope),
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(scope, namespace)
        page = svc.store.source_page(scope, memory_id, limit=limit, offset=offset)
        if page is None:
            raise HTTPException(404, "Memory not found")
        return page

    @app.delete("/v1/memories/{memory_id}", response_model=DeleteResponse)
    def delete_memory(
        memory_id: str,
        scope: Scope = Depends(query_scope),
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(scope, namespace)
        deleted = svc.store.delete(scope, memory_id)
        if not deleted:
            raise HTTPException(404, "Memory not found")
        app.state.images.clear_scope(scope)
        return {"deleted_ids": deleted}

    @app.post("/v1/memories/{memory_id}/rehearse", response_model=Memory)
    def rehearse(
        memory_id: str,
        body: Scope,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body, namespace)
        result = svc.store.rehearse(body, memory_id)
        if result is None:
            raise HTTPException(404, "Active memory not found")
        return result

    @app.post("/v1/ingest", response_model=IngestResponse)
    def ingest(
        body: IngestRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.ingest(body)

    @app.post("/v1/events", response_model=EventResponse, status_code=202)
    def enqueue(
        body: EventRequest,
        namespace: str | None = Depends(principal),
        queue: EventQueue = Depends(current_queue),
    ):
        authorize(body.scope, namespace)
        return queue.enqueue(body)

    @app.post("/v1/images/analyze", response_model=ImageEvidence)
    def analyze_image(
        body: ImageRequest,
        namespace: str | None = Depends(principal),
    ):
        authorize(body.scope, namespace)
        return app.state.images.analyze(body)

    @app.post("/v1/images/events", response_model=ImageEventResponse, status_code=202)
    def image_event(
        body: ImageEventRequest,
        namespace: str | None = Depends(principal),
        queue: EventQueue = Depends(current_queue),
    ):
        authorize(body.scope, namespace)
        return enqueue_image(body, queue, app.state.images)

    @app.get("/v1/events/{event_id}", response_model=EventResponse)
    def get_event(
        event_id: str,
        scope: Scope = Depends(query_scope),
        namespace: str | None = Depends(principal),
        queue: EventQueue = Depends(current_queue),
    ):
        authorize(scope, namespace)
        result = queue.get(scope, event_id)
        if result is None:
            raise HTTPException(404, "Event not found")
        return result

    @app.post("/v1/events/{event_id}/retry", response_model=JobInfo)
    def retry_event(
        event_id: str,
        body: Scope,
        namespace: str | None = Depends(principal),
        queue: EventQueue = Depends(current_queue),
    ):
        authorize(body, namespace)
        return queue.retry(body, event_id)

    @app.post("/v1/index", response_model=IndexResponse)
    def index_memories(
        body: IndexRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.index(body)

    @app.post("/v1/videos/analyze", response_model=VideoEvidence)
    def analyze_video(body: VideoRequest, namespace: str | None = Depends(principal)):
        authorize(body.scope, namespace)
        return app.state.videos.analyze(body)

    @app.post("/v1/videos/events", response_model=EventResponse, status_code=202)
    def video_event(
        body: VideoEventRequest,
        namespace: str | None = Depends(principal),
        queue: EventQueue = Depends(current_queue),
    ):
        authorize(body.scope, namespace)
        return enqueue_video(body, queue, app.state.videos)

    @app.post("/v1/recall", response_model=RecallResponse)
    def recall(
        body: RecallRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.recall(body)

    @app.post("/v1/context", response_model=ContextResponse)
    def context(
        body: ContextRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.context(body)

    @app.post("/v1/consolidate", response_model=ConsolidateResponse)
    def consolidate(
        body: ConsolidateRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.consolidate(body)

    @app.post("/v1/maintenance", response_model=MaintenanceResponse)
    def maintenance(
        body: MaintenanceRequest,
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(body.scope, namespace)
        return svc.maintain(body)

    @app.get("/v1/stats", response_model=StatsResponse)
    def stats(
        scope: Scope = Depends(query_scope),
        namespace: str | None = Depends(principal),
        svc: MemoryService = Depends(current_service),
    ):
        authorize(scope, namespace)
        return svc.store.stats(scope)

    return app
