import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Tuple

import torch
import hydra
import gradio as gr
from omegaconf import DictConfig

from src import utils
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torchvision.transforms as transforms
from typing import Dict

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)
    #print(model)
    log.info(f"Loaded Model: {model}")

       
    def predict(inp_img:Image):# -> Dict[str, float]:
        img_tensor=transforms.ToTensor()(inp_img).unsqueeze(0).to(device="cuda")
        #img_tensor=transform(inp_img).unsqeeze(0)
        #print(img_tensor)
        #transform = transforms.Compose([transforms.PILToTensor()])
        #img_tensor = transform(inp_img)
        with torch.no_grad():
            out = model.forward_jit(img_tensor)
        #print(out)
        #    probabilities = torch.nn.functional.softmax(out[0], dim=0)
        #    confidences = {categories[i]: float(probabilities[i]) for i in range(1000)}
        #return confidences
        return str(img_tensor.shape)
    
    
    #im = gr.Image(shape=(32, 32),image_mode="L")

    demo = gr.Interface(
        fn=predict,
        #fn=
        inputs=gr.Image(shape=(32, 32),image_mode="RGB"),
        outputs="label",
        #outputs=[gr.Label(num_top_classes=10)],
        live=True,
    ).launch(share=True)

@hydra.main(
    version_base="1.2", config_path=root / "configs", config_name="demo_scripted.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()