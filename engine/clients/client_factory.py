from abc import ABC
from typing import List, Type

from engine.base_client.client import (
    BaseClient,
    BaseConfigurator,
    BaseSearcher,
    BaseUploader,
)

ENGINE_CONFIGURATORS = {
    "qdrant": "engine.clients.qdrant.QdrantConfigurator",
    "weaviate": "engine.clients.weaviate.WeaviateConfigurator",
    "milvus": "engine.clients.milvus.MilvusConfigurator",
    "elasticsearch": "engine.clients.elasticsearch.ElasticConfigurator",
    "opensearch": "engine.clients.opensearch.OpenSearchConfigurator",
    "redis": "engine.clients.redis.RedisConfigurator",
    "pgvector": "engine.clients.pgvector.PgVectorConfigurator",
}

ENGINE_UPLOADERS = {
    "qdrant": "engine.clients.qdrant.QdrantUploader",
    "weaviate": "engine.clients.weaviate.WeaviateUploader",
    "milvus": "engine.clients.milvus.MilvusUploader",
    "elasticsearch": "engine.clients.elasticsearch.ElasticUploader",
    "opensearch": "engine.clients.opensearch.OpenSearchUploader",
    "redis": "engine.clients.redis.RedisUploader",
    "pgvector": "engine.clients.pgvector.PgVectorUploader",
}

ENGINE_SEARCHERS = {
    "qdrant": "engine.clients.qdrant.QdrantSearcher",
    "weaviate": "engine.clients.weaviate.WeaviateSearcher",
    "milvus": "engine.clients.milvus.MilvusSearcher",
    "elasticsearch": "engine.clients.elasticsearch.ElasticSearcher",
    "opensearch": "engine.clients.opensearch.OpenSearchSearcher",
    "redis": "engine.clients.redis.RedisSearcher",
    "pgvector": "engine.clients.pgvector.PgVectorSearcher",
}


def _get_class(class_path: str):
    module_name, class_name = class_path.rsplit(".", 1)
    # The importlib.import_module function is used to dynamically import a module.
    # It takes the module name as a string and returns the module object.
    # The getattr function is then used to get the class from the module.
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class ClientFactory(ABC):
    def __init__(self, host):
        self.host = host
        self.engine = None

    def _create_configurator(self, experiment) -> BaseConfigurator:
        self.engine = experiment["engine"]
        engine_configurator_class = _get_class(ENGINE_CONFIGURATORS[experiment["engine"]])
        engine_configurator = engine_configurator_class(
            self.host,
            collection_params={**experiment.get("collection_params", {})},
            connection_params={**experiment.get("connection_params", {})},
        )
        return engine_configurator

    def _create_uploader(self, experiment) -> BaseUploader:
        engine_uploader_class = _get_class(ENGINE_UPLOADERS[experiment["engine"]])
        engine_uploader = engine_uploader_class(
            self.host,
            connection_params={**experiment.get("connection_params", {})},
            upload_params={**experiment.get("upload_params", {})},
        )
        return engine_uploader

    def _create_searchers(self, experiment) -> List[BaseSearcher]:
        engine_searcher_class: Type[BaseSearcher] = _get_class(ENGINE_SEARCHERS[
            experiment["engine"]
        ])

        engine_searchers = [
            engine_searcher_class(
                self.host,
                connection_params={**experiment.get("connection_params", {})},
                search_params=search_params,
            )
            for search_params in experiment.get("search_params", [{}])
        ]

        return engine_searchers

    def build_client(self, experiment, drop_caches: bool = False):
        return BaseClient(
            name=experiment["name"],
            engine=experiment["engine"],
            configurator=self._create_configurator(experiment),
            uploader=self._create_uploader(experiment),
            searchers=self._create_searchers(experiment),
            drop_caches=drop_caches,
        )
