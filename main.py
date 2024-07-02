from pathlib import Path
import pickle
import flwr as fl
from omegaconf import DictConfig, OmegaConf

import hydra
from dataset import prepare_dataset
from client import generate_client_fn
from server import get_evaluate_fn, get_on_fit_config
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ##1. Parse config and get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ##2. Prepare dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )
    print(len(trainloaders), len(trainloaders[0].dataset))
    ##3. Define your clients
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ##4. Define your strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.000001,
        min_fit_clients=cfg.num_clients_per_round_fit,
        fraction_evaluate=0.000001,
        min_available_clients=cfg.num_clients,
        min_evaluate_clients=cfg.num_clients_per_round_eva,
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # this is used for local training and evaluating
    # the fraction_fit and fraction_evaluate are low because we assume that all client are available
    ##5. Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_round),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.25},
    )
    ##6. Save your results
    save_path = HydraConfig.get().runtime.output_dir
    results_path = Path(save_path) / "result.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()