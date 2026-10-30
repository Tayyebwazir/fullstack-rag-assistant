import uuid
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter
from weaviate.classes.tenants import Tenant
from app.config import (
    COLLECTION_NAME,
    WEAVIATE_MODE,
    WEAVIATE_URL,
    WEAVIATE_API_KEY,
)


def connect_weaviate():
    if WEAVIATE_MODE == "cloud":
        if not WEAVIATE_URL or not WEAVIATE_API_KEY:
            raise RuntimeError("Set Weaviate credentials in .env")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
        )
    return weaviate.connect_to_local()


def ensure_collection(client):
    if not client.collections.exists(COLLECTION_NAME):
        client.collections.create(
            name=COLLECTION_NAME,
            multi_tenancy_config=Configure.multi_tenancy(enabled=True),
            vector_config=Configure.Vectors.text2vec_google_gemini(
                source_properties=["content"],
                model="gemini-embedding-001",
                dimensions=3072,
            ),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="page", data_type=DataType.INT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )
    return client.collections.get(COLLECTION_NAME)


def ensure_tenant(collection, tenant_id: str) -> str:
    """Creates the user's isolated tenant in Weaviate if it does not already exist.
    If the 3-tenant sandbox limit on Weaviate Cloud free tier is reached, gracefully
    reuses an existing active tenant instead of crashing with a 429 USAGE_LIMIT_EXCEEDED.
    """
    existing_tenants = list(collection.tenants.get().keys())
    if tenant_id in existing_tenants:
        return tenant_id

    # If already at or exceeding Weaviate Cloud Sandbox 3-tenant free tier limit
    if len(existing_tenants) >= 3:
        fallback = existing_tenants[0]
        print(f"[Weaviate] Sandbox limit reached ({len(existing_tenants)}/3). Mapping '{tenant_id}' to active tenant '{fallback}'.")
        return fallback

    try:
        collection.tenants.create([Tenant(name=tenant_id)])
        return tenant_id
    except Exception as e:
        err_str = str(e)
        if "USAGE_LIMIT_EXCEEDED" in err_str or "limit of 3 reached" in err_str:
            existing = list(collection.tenants.get().keys())
            if existing:
                fallback = existing[0]
                print(f"[Weaviate] Sandbox 3-tenant limit hit during create. Falling back to active tenant '{fallback}'.")
                return fallback
        raise e


def add_manual_tenants(tenant_names: list[str]) -> list[str]:
    """Manually add multiple tenants to the collection."""
    with connect_weaviate() as client:
        collection = ensure_collection(client)
        existing = collection.tenants.get()
        new_tenants = [Tenant(name=name) for name in tenant_names if name not in existing]
        if new_tenants:
            try:
                collection.tenants.create(tenants=new_tenants)
            except Exception as e:
                print(f"[Weaviate] Note adding tenants: {e}")
        return list(collection.tenants.get().keys())


def ingest_chunks(
    chunks: list[dict],
    source: str,
    vectors: list[list[float]],
    tenant_id: str = "default_user",
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector")

    with connect_weaviate() as client:
        collection = ensure_collection(client)
        active_tenant = ensure_tenant(collection, tenant_id)

        # Scope operations strictly to the resolved tenant
        tenant_collection = collection.with_tenant(active_tenant)
        tenant_collection.data.delete_many(where=Filter.by_property("source").equal(source))

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            object_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{active_tenant}:{source}:{index}")
            tenant_collection.data.insert(
                uuid=object_id,
                properties={
                    "content": chunk["content"],
                    "source": source,
                    "page": chunk["page"],
                    "chunk_index": index,
                },
                vector=vector,
            )
        return len(chunks)


def search_chunks(
    query_vector: list[float],
    tenant_id: str = "default_user",
    source_filter: str | None = None,
    limit: int = 4,
) -> list[dict]:
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            raise ValueError("No ingested collection found. Please ingest the PDF documents first.")

        collection = client.collections.get(COLLECTION_NAME)
        existing_tenants = list(collection.tenants.get().keys())
        if not existing_tenants:
            return []

        # If user's tenant exists, query that; otherwise fallback to the primary active tenant
        active_tenant = tenant_id if tenant_id in existing_tenants else existing_tenants[0]

        # Query within this tenant
        tenant_collection = collection.with_tenant(active_tenant)

        filters = None
        if source_filter and source_filter != "all":
            filters = Filter.by_property("source").equal(source_filter)

        response = tenant_collection.query.near_vector(
            near_vector=query_vector,
            filters=filters,
            limit=limit,
        )
        return [
            {
                "source": obj.properties["source"],
                "page": obj.properties["page"],
                "chunk_index": obj.properties["chunk_index"],
                "content": obj.properties["content"],
            }
            for obj in response.objects
        ]
