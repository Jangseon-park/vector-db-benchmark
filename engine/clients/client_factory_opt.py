from abc import ABC
from typing import List, Type

from engine.base_client.client_opt import (
    BaseClient,
)
from engine.base_client.client import (
    BaseConfigurator,
    BaseSearcher,
    BaseUploader,
)
from engine.clients.milvus import MilvusConfigurator, MilvusSearcher, MilvusUploader

ENGINE_CONFIGURATORS = {
    "milvus": MilvusConfigurator,
}

ENGINE_UPLOADERS = {
    "milvus": MilvusUploader,
}

ENGINE_SEARCHERS = {
    "milvus": MilvusSearcher,
}


class ClientFactory(ABC):
    def __init__(self, host):
        self.host = host
        self.engine = None

    def _create_configurator(self, experiment) -> BaseConfigurator:
        self.engine = experiment["engine"]
        engine_configurator_class = ENGINE_CONFIGURATORS[experiment["engine"]]
        engine_configurator = engine_configurator_class(
            self.host,
            collection_params={**experiment.get("collection_params", {})},
            connection_params={**experiment.get("connection_params", {})},
        )
        return engine_configurator

    def _create_uploader(self, experiment) -> BaseUploader:
        engine_uploader_class = ENGINE_UPLOADERS[experiment["engine"]]
        engine_uploader = engine_uploader_class(
            self.host,
            connection_params={**experiment.get("connection_params", {})},
            upload_params={**experiment.get("upload_params", {})},
        )
        return engine_uploader

    def _create_searchers(self, experiment) -> List[BaseSearcher]:
        engine_searcher_class: Type[BaseSearcher] = ENGINE_SEARCHERS[
            experiment["engine"]
        ]

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
