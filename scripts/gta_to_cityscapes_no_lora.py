# Datasets
from datasets import load_from_disk
# Model
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
# Preprocessing
import json
from huggingface_hub import cached_download, hf_hub_url
from torchvision.transforms import ColorJitter
from PIL import Image
import numpy as np
# Metrics
import torch
from torch import nn
import evaluate
# Train
from transformers import TrainingArguments, Trainer


##################### DATASET #####################
# Load the dataset from disk
loaded_dataset = load_from_disk("../cityscapes_train_1000_dataset_v3")

# Prepare train and test splits
loaded_dataset = loaded_dataset.train_test_split(test_size=0.1)
train_ds = loaded_dataset["train"]
test_ds = loaded_dataset["test"]

##################### IDs and LABELs #####################
repo_id = "huggingface/label-files"
filename = "cityscapes-id2label.json"
id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

##################### BASE MODEL #####################
# id of the model to load, SegFormer pretrained on gta 
model_id = "hector-alvarez/segformerb4-gta"

image_processor = SegformerImageProcessor()

model = SegformerForSemanticSegmentation.from_pretrained(model_id)


##################### TRANSFORMATIONS #####################

# Transofrms the color properities

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)




def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

test_ds.set_transform(val_transforms)
   

##################### MÉTRICAS #####################
# Cargar la métrica mean_iou
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # Escalar los logits al tamaño de las etiquetas
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],  # altura y ancho
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.do_reduce_labels,
        )

        # Añadir precisión y IoU por categoría
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics
    
##################### TRAINING #####################

model_name = model_id.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"./noLora-{model_name}-cityscapes",
    per_device_eval_batch_size=14,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)

eval_results = trainer.evaluate()
model_id = "segformerB0-gta-cityscapes-lora"
print("Evaluation results:", eval_results)