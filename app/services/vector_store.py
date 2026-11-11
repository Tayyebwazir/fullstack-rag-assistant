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
                Property(name="doc_id", data_type=DataType.TEXT),
            ],
        )
    collection = client.collections.get(COLLECTION_NAME)
    # Ensure all required schema properties exist
    try:
        existing_props = {p.name for p in collection.config.get().properties}
        required_props = [
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="chunk_index", data_type=DataType.INT),
            Property(name="doc_id", data_type=DataType.TEXT),
        ]
        for prop in required_props:
            if prop.name not in existing_props:
                try:
                    collection.config.add_property(prop)
                    print(f"[Weaviate] Added missing property '{prop.name}' to schema.")
                except Exception as pe:
                    print(f"[Weaviate] Error adding property '{prop.name}': {pe}")
    except Exception as e:
        print(f"[Weaviate] Error checking schema properties: {e}")
    return collection


def get_weaviate_tenants() -> list[str]:
    """Retrieves all active tenant shards directly from Weaviate DB without any hardcoded fallbacks."""
    try:
        with connect_weaviate() as client:
            if not client.collections.exists(COLLECTION_NAME):
                return []
            col = client.collections.get(COLLECTION_NAME)
            tenants = list(col.tenants.get().keys())
            return tenants
    except Exception as e:
        print(f"[Weaviate] Error fetching tenants: {e}")
        return []


def create_weaviate_tenant(tenant_id: str | None = None) -> str:
    """Creates a new tenant shard in Weaviate using a UUIDv4 (matching weaviate_connection.py)."""
    target = tenant_id.strip() if tenant_id and tenant_id.strip() else str(uuid.uuid4())
    with connect_weaviate() as client:
        collection = ensure_collection(client)
        existing = list(collection.tenants.get().keys())
        if target not in existing:
            collection.tenants.create([Tenant(name=target)])
            print(f"[Weaviate] Dynamically provisioned dedicated UUID tenant shard: '{target}'")
    return target


def ensure_tenant(collection, tenant_id: str) -> str:
    """Dynamically creates the tenant shard in Weaviate if it does not already exist."""
    target = tenant_id.strip() if tenant_id and tenant_id.strip() else str(uuid.uuid4())
    existing = list(collection.tenants.get().keys())
    if target not in existing:
        try:
            collection.tenants.create([Tenant(name=target)])
            print(f"[Weaviate] Dynamically provisioned new tenant shard: '{target}'")
        except Exception as e:
            print(f"[Weaviate] Note ensuring tenant {target}: {e}")
    return target


def ingest_chunks(
    chunks: list[dict],
    source: str,
    vectors: list[list[float]],
    tenant_id: str = "default",
    doc_id: str = "",
) -> int:
    """Inserts chunk vectors strictly into the dynamically authorized tenant's shard."""
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector")

    with connect_weaviate() as client:
        collection = ensure_collection(client)
        active_tenant = ensure_tenant(collection, tenant_id)

        tenant_collection = collection.with_tenant(active_tenant)

        if doc_id:
            try:
                tenant_collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
            except Exception:
                pass
        tenant_collection.data.delete_many(where=Filter.by_property("source").equal(source))

        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            object_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{active_tenant}:{doc_id or source}:{index}")
            tenant_collection.data.insert(
                uuid=object_id,
                properties={
                    "content": chunk["content"],
                    "source": source,
                    "page": chunk["page"],
                    "chunk_index": index,
                    "doc_id": doc_id,
                },
                vector=vector,
            )
        return len(chunks)


def delete_tenant_document_by_id(doc_id: str, tenant_id: str = "default") -> int:
    """Deletes all chunks belonging to doc_id inside the tenant's isolated shard."""
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            return 0
        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = ensure_tenant(collection, tenant_id)
        tenant_collection = collection.with_tenant(target_tenant)
        try:
            res = tenant_collection.data.delete_many(where=Filter.by_property("doc_id").equal(doc_id))
            return res.successful if hasattr(res, "successful") else 1
        except Exception as exc:
            print(f"[Weaviate] Delete error for doc_id {doc_id}: {exc}")
            return 0


def delete_tenant_document(source_name: str, tenant_id: str = "default") -> int:
    """Deletes all chunks belonging to source_name inside tenant_id's shard."""
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            return 0
        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = ensure_tenant(collection, tenant_id)
        tenant_collection = collection.with_tenant(target_tenant)
        res = tenant_collection.data.delete_many(where=Filter.by_property("source").equal(source_name))
        return res.successful if hasattr(res, "successful") else 1


def get_tenant_documents_summary(tenant_id: str = "default") -> list[dict]:
    """Returns unique document sources and their chunk count in tenant_id's shard."""
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            return []
        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = ensure_tenant(collection, tenant_id)
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
    tenant_id: str = "default",
    source_filter: str | None = None,
    limit: int = 4,
) -> list[dict]:
    with connect_weaviate() as client:
        if not client.collections.exists(COLLECTION_NAME):
            raise ValueError("No ingested collection found. Please ingest documents first.")

        collection = client.collections.get(COLLECTION_NAME)
        target_tenant = ensure_tenant(collection, tenant_id)
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
                "source": obj.properties.get("source", "unknown"),
                "page": obj.properties.get("page", 1),
                "chunk_index": obj.properties.get("chunk_index", 0),
                "content": obj.properties.get("content", ""),
                "doc_id": obj.properties.get("doc_id", ""),
                "tenant": target_tenant,
            }
            for obj in response.objects
        ]
