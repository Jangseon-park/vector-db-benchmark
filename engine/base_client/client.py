import json
import os
from datetime import datetime
from typing import List, Optional

from benchmark import ROOT_DIR
from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.search import BaseSearcher
from engine.base_client.upload import BaseUploader

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DETAILED_RESULTS = bool(int(os.getenv("DETAILED_RESULTS", False)))


class BaseClient:
    def __init__(
        self,
        name: str,  # name of the experiment
        engine: str,  # name of the engine
        configurator: BaseConfigurator,
        uploader: BaseUploader,
        searchers: List[BaseSearcher],
        drop_caches: bool = False,
    ):
        self.name = name
        self.engine = engine
        self.configurator = configurator
        self.uploader = uploader
        self.searchers = searchers
        self.drop_caches = drop_caches

    @property
    def sparse_vector_support(self):
        return self.configurator.SPARSE_VECTOR_SUPPORT

    def save_search_results(
        self, dataset_name: str, results: dict, search_id: int, search_params: dict
    ):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        experiments_file = (
            f"{self.name}-{dataset_name}-search-{search_id}-{timestamp}.json"
        )
        result_path = RESULTS_DIR / experiments_file
        with open(result_path, "w") as out:
            out.write(
                json.dumps(
                    {
                        "params": {
                            "dataset": dataset_name,
                            "experiment": self.name,
                            "engine": self.engine,
                            **search_params,
                        },
                        "results": results,
                    },
                    indent=2,
                )
            )
        return result_path

    def save_upload_results(
        self, dataset_name: str, results: dict, upload_params: dict
    ):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        experiments_file = f"{self.name}-{dataset_name}-upload-{timestamp}.json"
        with open(RESULTS_DIR / experiments_file, "w") as out:
            upload_stats = {
                "params": {
                    "experiment": self.name,
                    "engine": self.engine,
                    "dataset": dataset_name,
                    **upload_params,
                },
                "results": results,
            }
            out.write(json.dumps(upload_stats, indent=2))

    def run_experiment(
        self,
        dataset: Dataset,
        skip_upload: bool = False,
        skip_search: bool = False,
        skip_if_exists: bool = True,
        skip_configure: Optional[bool] = False,
    ):
        execution_params = self.configurator.execution_params(
            distance=dataset.config.distance, vector_size=dataset.config.vector_size
        )

        reader = dataset.get_reader(execution_params.get("normalize", False))

        if skip_if_exists:
            glob_pattern = f"{self.name}-{dataset.config.name}-search-*-*.json"
            existing_results = list(RESULTS_DIR.glob(glob_pattern))
            if len(existing_results) == len(self.searchers):
                print(
                    f"Skipping run for {self.name} since it already ran {len(self.searchers)} search configs previously"
                )
                return

        if not skip_upload:
            if not skip_configure:
                print("Experiment stage: Configure")
                self.configurator.configure(dataset)

            print("Experiment stage: Upload")
            upload_stats = self.uploader.upload(
                distance=dataset.config.distance, records=reader.read_data()
            )

            if not DETAILED_RESULTS:
                # Remove verbose stats from upload results
                upload_stats.pop("latencies", None)

            self.save_upload_results(
                dataset.config.name,
                upload_stats,
                upload_params={
                    **self.uploader.upload_params,
                    **self.configurator.collection_params,
                },
            )

        if not skip_search:
            print("Experiment stage: Search")
            # If upload was skipped, the collection may not be loaded in Milvus
            # after a server restart. If the uploader provides an ensure_loaded()
            # helper (MilvusUploader), call it once to wait for recovery.
            if skip_upload and hasattr(self.uploader, "ensure_loaded"):
                try:
                    print("Upload skipped — initializing uploader client and waiting for collection to become available...")
                    # Ensure uploader client is initialized (class-level init_client may be required)
                    try:
                        # uploader.init_client is a classmethod on uploader implementations
                        self.uploader.init_client(
                            self.uploader.host,
                            dataset.config.distance,
                            self.uploader.connection_params,
                            self.uploader.upload_params,
                        )
                    except Exception:
                        # ignore failures from init_client; ensure_loaded will still try
                        pass

                    # default timeout is implemented by the uploader.ensure_loaded
                    self.uploader.ensure_loaded()
                    print("Collection is available — proceeding with search")
                except Exception as e:
                    print("Warning: waiting for collection availability failed:", e)
                    # Continue and let individual searches handle failures
            for search_id, searcher in enumerate(self.searchers):

                if skip_if_exists:
                    glob_pattern = (
                        f"{self.name}-{dataset.config.name}-search-{search_id}-*.json"
                    )
                    existing_results = list(RESULTS_DIR.glob(glob_pattern))
                    print("Pattern", glob_pattern, "Results:", existing_results)
                    if len(existing_results) >= 1:
                        print(
                            f"Skipping search {search_id} as it already exists",
                        )
                        continue
                # add flush system call
                if self.drop_caches:
                    if os.geteuid() == 0:
                        # add flush system call
                        os.system("sync")
                        os.system("echo 3 > /proc/sys/vm/drop_caches")
                    else:
                        # add flush system call
                        os.system("sync")
                        os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")

                search_params = {**searcher.search_params}
                search_stats = searcher.search_all(
                    dataset.config.distance, reader.read_queries()
                )
                if not DETAILED_RESULTS:
                    # Remove verbose stats from search results
                    search_stats.pop("latencies", None)
                    search_stats.pop("precisions", None)

                self.save_search_results(
                    dataset.config.name, search_stats, search_id, search_params
                )
        print("Experiment stage: Done")
        print("Results saved to: ", RESULTS_DIR)

    def delete_client(self):
        self.uploader.delete_client()
        self.configurator.delete_client()

        for s in self.searchers:
            s.delete_client()
