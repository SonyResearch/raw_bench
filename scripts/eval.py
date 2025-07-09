import hydra

@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(config):
    if config.model_type == 'silentcipher':
        from raw_bench.solver import SolverSilentCipher
        solver = SolverSilentCipher(config)

    elif config.model_type == 'audioseal':
        from raw_bench.solver import SolverAudioSeal        
        solver = SolverAudioSeal(config)

    elif config.model_type == 'timbre':
        from raw_bench.solver import SolverTimbre
        solver = SolverTimbre(config)

    elif config.model_type == 'wavmark':
        from raw_bench.solver import SolverWavMark
        solver = SolverWavMark(config)
    else:   
        raise ValueError(f"Entered model type {config.model_type} not supported!")
        
    if config.checkpoint is None:
        raise ValueError("checkpoint must be provided in test mode")
    
    assert config.mode == 'test'
    solver.eval(write_to_disk=True)
    solver.close_eval()
    
 
if __name__ == '__main__':
    main()