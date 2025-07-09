from functools import partial
import os
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter
from typing import Optional

from .utils import get_saftdict


try:
    import wandb
    EXTERNAL_LOGGING_AVAILABLE = True
except Exception as e:
    EXTERNAL_LOGGING_AVAILABLE = False

class ExperimentLogger(object):
    def __init__(
        self,
        root_dir: str,
        project_name: Optional[str] = None,
        exp_name: Optional[str] = None,
        use_wandb: bool = False
    ):
        """
        Initialize an Experiment object, set up directories and logging.

        Args:
            root_dir: str
                Root directory for the experiment.
            use_wandb: bool, optional
                Whether to use Weights & Biases for logging.
        """
        # define dirs
        self.dir = root_dir
        self.project_name = project_name
        self.exp_name = exp_name

        # create dirs
        os.makedirs(self.dir, exist_ok=True)
        logger.info(f"experiment logging directory: {self.dir}")

        self.use_wandb = False
        self.wandb_fn = None

        # use wandb, if available
        if use_wandb:
            self.use_wandb = 'WANDB_API_KEY' in os.environ or wandb.Api().api_key is not None
            if self.use_wandb:
                self.wandb_fn = partial(wandb.init,
                                        project=self.project_name,
                                        name=self.exp_name,
                                        dir=self.dir)
            else:
                logger.warning("WANDB_API_KEY is not set, use tensorboard for logging")

        if not self.use_wandb:        
            # create tensorboard writer
            tb_dir = Path(self.dir) / f"{self.project_name}_{self.exp_name}"
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)


    def log_hparams(self, config, hparams=None):
        """
        Initialize Weights & Biases (wandb) logging for the experiment.

        Args:
            config (DictConfig): Experiment configuration.
            hparams (dict, optional): Additional hyperparameters to log.
        """
        try:
            resolved_config = DictConfig(config).copy()
            OmegaConf.resolve(resolved_config)                    
            resolved_config = get_saftdict(resolved_config)
            if hparams is not None:
                resolved_config['train_configs'] = hparams
            
            ckpt_filename = Path(config.checkpoint).name
    
            if self.use_wandb and self.wandb_fn is not None:
                
                self.wandb_fn(config=resolved_config)
                logger.info('Initialized wandb!')
            else:
                # resolved_config = {str(k): str(v) for k, v in resolved_config.items()} <- this is ugly
                self.tb_writer.add_text(
                    'config',
                    f'```yaml\n{OmegaConf.to_yaml(resolved_config)}\n```'
                )
                
        except Exception as e:
            logger.warning(f'Failed to save hparams: {e}')  

    def log_metric(
        self,
        metrics_dict: dict,
        step: int = None
        ):
        """
        Log metrics to TensorBoard and optionally to Weights & Biases.

        Args:
            metrics_dict: dict
                Dictionary of metrics to log.
            step: int, optional
                Step or epoch number.

        """
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)
        else:
            _metrics_dict = {
                k if k.count('/') <= 1 else k.split('/', 1)[0] + '/' 
                + k.split('/', 1)[1].replace('/', '_'): v
                for k, v in metrics_dict.items()
            }
        
            for k,v in _metrics_dict.items():
                # log in tensorboard
                self.tb_writer.add_scalar(k, v, step)

    def log_wandb_if_possible(self, content):
        """
        Log content to Weights & Biases if available.

        Args:
            content: dict
                Content to log.
        """
        if self.use_wandb:
            wandb.log(content)
        else:
            logger.warning("Weights & Biases is not available, skipping logging.")
    
    def close(self):
        """
        Close the logger and any associated resources.
        """
        if self.use_wandb:
            wandb.finish()
        else:
            self.tb_writer.close()
            logger.info("Closed TensorBoard writer.")