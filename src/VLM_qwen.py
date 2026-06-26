from PIL import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
from torchvision.transforms.functional import to_pil_image
import tempfile


def save_temp_image(img: Image.Image, img_name) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=f"{img_name}.png", delete=False)
    img.save(tmp.name)
    return tmp.name


class VLM_SemanticFilter:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
        self.device = device
        self.model_name = model_name

        # Load model with automatic device mapping
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

        self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

    def build_prompt(self, user_prompt: str) -> str:
        """
        Create the system prompt given the user's task.
        """
        return (
            f"A red robot arm is autonomously executing a user-defined task.\n"
            f'Here is the initial task provided by the user: "{user_prompt}"\n'
            f"- The robot may be at any step of the task (it might not have started yet).\n"
            f"- The anomaly detector has triggered.\n"
            f"We provide three images:"
            f"1. An expert demonstration at the same timestep (could be not perfectly aligned)."
            f"2. The current situation that triggered the anomaly detector.  "
            f"3. The anomaly heatmap that spatially localize the anomaly detected (the heatmap is superimposed on image 2)"
            f"Your task:  "
            f"1. Identify the anomaly by comparing both images and analysing the heatmap.  "
            f"2. Decide if the anomaly is:  "
            f"- 0: a false positive, there is no threat to the completion of the task.  "
            f"- 1: a true failure has occured, the task is not properly done or is jeopardized"
            f"Guidelines:  "
            f"- **If nothing consistent appears on the heatmap, chose 0**"
            f"Start your answer with either `0` or `1`, then justify your reasoning.  "
        )

    def analyze(
        self,
        cfg,
        image_expert: Image.Image,
        image_anomaly: Image.Image,
        user_prompt: str,
        last_vlm_prompt: str,
    ):
        """
        Analyze the scene using Qwen2.5-VL and return:
        - severity: 0 (benign) or -1 (failure)
        - new_prompt: string
        """

        # Convert Torch tensor to PIL image
        if isinstance(image_expert, torch.Tensor):
            if image_expert.ndim == 4:
                # shape: (1, C, H, W)
                image_expert = image_expert.squeeze(0)
            if image_expert.dtype == torch.float:
                image_expert = image_expert.clamp(0, 1)  # Ensure valid range
            image_expert = to_pil_image(image_expert)

        if isinstance(image_anomaly, torch.Tensor):
            if image_anomaly.ndim == 4:
                # shape: (1, C, H, W)
                image_anomaly = image_anomaly.squeeze(0)
            if image_anomaly.dtype == torch.float:
                image_anomaly = image_anomaly.clamp(0, 1)  # Ensure valid range
            image_anomaly = to_pil_image(image_anomaly)

        # Build structured message for Qwen-VL

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_expert},
                    {"type": "image", "image": image_anomaly},
                    {
                        "type": "text",
                        "text": self.build_prompt(user_prompt) + last_vlm_prompt,
                    },
                ],
            }
        ]

        # Format inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        # Remove prompt tokens to get clean response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        try:
            severity = int(output_text[0])
        except Exception:
            severity = -1

        return severity, output_text
