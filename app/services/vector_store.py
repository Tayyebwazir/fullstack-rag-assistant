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


def ensure_tenant(collection, tenant_id: str):
    """Creates the user's isolated tenant in Weaviate if it does not already exist."""
    existing_tenants = collection.tenants.get()
    if tenant_id not in existing_tenants:
        collection.tenants.create([Tenant(name=tenant_id)])


def add_manual_tenants(tenant_names: list[str]) -> list[str]:
    """Manually add multiple tenants to the collection."""
    with connect_weaviate() as client:
        collection = ensure_collection(client)
        existing = collection.tenants.get()
        new_tenants = [Tenant(name=name) for name in tenant_names if name not in existing]
        if new_tenants:
            collection.tenants.create(tenants=new_tenants)
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
        ensure_tenant(collection, tenant_id)

        # Scope operations strictly to this user's tenant
        tenant_collection = collection.with_tenant(tenant_id)
        tenant_collection.data.delete_many(where=Filter.by_property("source").equal(source))

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            object_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{tenant_id}:{source}:{index}")
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
        existing_tenants = collection.tenants.get()
        if tenant_id not in existing_tenants:
            return []

        # Query only within this user's isolated tenant
        tenant_collection = collection.with_tenant(tenant_id)

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
