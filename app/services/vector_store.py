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

VALID_TENANTS = ["alpha", "beta", "tri"]

TENANT_DOCUMENTS = {
    "alpha": "Attention is all you need.pdf",
    "beta": "LLM-based autonomous agents.pdf",
    "tri": "OOP_15_Page_Notes.pdf",
}

TENANT_METADATA = [
    {
        "id": "alpha",
        "name": "Tenant Alpha",
        "document": "Attention is all you need.pdf",
        "description": "Transformers & Attention Mechanism (Research Paper)",
    },
    {
        "id": "beta",
        "name": "Tenant Beta",
        "document": "LLM-based autonomous agents.pdf",
        "description": "Autonomous Agents, Planning & Memory (Research Paper)",
    },
    {
        "id": "tri",
        "name": "Tenant Tri",
        "document": "OOP_15_Page_Notes.pdf",
        "description": "Object-Oriented Programming & System Design (Study Notes)",
    },
]


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
    """Ensures the tenant is one of ['alpha', 'beta', 'tri'] and exists in Weaviate."""
    target = tenant_id if tenant_id in VALID_TENANTS else "alpha"
    existing = list(collection.tenants.get().keys())
    if target not in existing:
        try:
            collection.tenants.create([Tenant(name=target)])
        except Exception as e:
            print(f"[Weaviate] Note ensuring tenant {target}: {e}")
    return target


def ingest_chunks(
    chunks: list[dict],
    source: str,
    vectors: list[list[float]],
    tenant_id: str = "alpha",
) -> int:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector")

    with connect_weaviate() as client:
        collection = ensure_collection(client)
        active_tenant = ensure_tenant(collection, tenant_id)

        # Scope operations strictly to the authorized tenant
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


def delete_tenant_document(source_name: str, tenant_id: str = "alpha") -> int:
    """Deletes all chunks belonging to source_name inside tenant_id's shard."""
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            return 0
        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = tenant_id if tenant_id in VALID_TENANTS else "alpha"
        ensure_tenant(collection, target_tenant)
        tenant_collection = collection.with_tenant(target_tenant)
        res = tenant_collection.data.delete_many(where=Filter.by_property("source").equal(source_name))
        return res.successful if hasattr(res, "successful") else 1


def get_tenant_documents_summary(tenant_id: str = "alpha") -> list[dict]:
    """Returns unique document sources and their chunk count in tenant_id's shard."""
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            return []
        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = tenant_id if tenant_id in VALID_TENANTS else "alpha"
        ensure_tenant(collection, target_tenant)
        tenant_collection = collection.with_tenant(target_tenant)

        objects = tenant_collection.query.fetch_objects(limit=1000).objects
        counts = {}
        for obj in objects:
            src = obj.properties.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        return [
            {"source": src, "chunk_count": count, "tenant": target_tenant}
            for src, count in counts.items()
        ]


def search_chunks(
    query_vector: list[float],
    tenant_id: str = "alpha",
    source_filter: str | None = None,
    limit: int = 4,
) -> list[dict]:
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            raise ValueError("No ingested collection found. Please ingest the PDF documents first.")

        collection = client.collections.get(COLLECTION_NAME)

        # Enforce strict tenant isolation: only 'alpha', 'beta', or 'tri'
        target_tenant = tenant_id if tenant_id in VALID_TENANTS else "alpha"
        ensure_tenant(collection, target_tenant)

        # Query strictly within this specific tenant's shard
        tenant_collection = collection.with_tenant(target_tenant)

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
                "tenant": target_tenant,
            }
            for obj in response.objects
        ]
