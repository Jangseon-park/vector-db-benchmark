import traceback
import json
import stopit
import typer

# Defer heavy imports that pull optional dependencies (like h5py)
# into runtime so `--help` and other CLI introspection won't fail
# when those optional packages aren't installed.
from engine.base_client import IncompatibilityError
from engine.clients.client_factory_opt import ClientFactory

app = typer.Typer()


@app.command()
def run(
    host: str = "localhost",
    timeout: float = 86400.0,
    drop_caches: bool = typer.Option(
        False, "--drop-caches/--no-drop-caches", help="Drop caches before search"
    ),
):
    """
    Examples:

    python3 optimized_run_milvus.py
    """
    with open("experiments/configurations/milvus-single-node_self.json", "r") as f:
        engine_config = json.load(f)[0]

    with open("datasets/datasets.json", "r") as f:
        all_datasets = json.load(f)
        dataset_config = next(
            (item for item in all_datasets if item["name"] == "glove-25-angular"), None
        )

    if engine_config is None:
        raise ValueError("Engine config not found")
    if dataset_config is None:
        raise ValueError("Dataset config not found")

    engine_name = engine_config["name"]
    dataset_name = dataset_config["name"]

    print(f"Running experiment: {engine_name} - {dataset_name}")
    client = ClientFactory(host).build_client(engine_config, drop_caches=drop_caches)
    try:

        # Import Dataset here to avoid requiring optional deps at import time
        from benchmark.dataset import Dataset

        dataset = Dataset(dataset_config)
        if dataset.config.type == "sparse" and not client.sparse_vector_support:
            raise IncompatibilityError(
                f"{client.name} engine does not support sparse vectors"
            )
        dataset.download()

        with stopit.ThreadingTimeout(timeout) as tt:
            client.run_experiment(
                dataset,
                skip_upload=True,
                skip_if_exists=False,
            )
        client.delete_client()

        # If the timeout is reached, the server might be still in the
        # middle of some background processing, like creating the index.
        # Next experiment should not be launched. It's better to reset
        # the server state manually.
        if tt.state != stopit.ThreadingTimeout.EXECUTED:
            print(
                f"Timed out {engine_name} - {dataset_name}, "
                f"exceeded {timeout} seconds"
            )
            exit(2)
    except IncompatibilityError as e:
        print(
            f"Skipping {engine_name} - {dataset_name}, incompatible params:", e
        )
    except KeyboardInterrupt:
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"Experiment {engine_name} - {dataset_name} interrupted")
        traceback.print_exc()
        raise e


if __name__ == "__main__":
    app()
