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
# LoRA
from peft import LoraConfig, get_peft_model
# Train
from transformers import TrainingArguments, Trainer
import transformers


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
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

def handle_grayscale_image(image):
    np_image = np.array(image)
    if np_image.ndim == 2:
        tiled_image = np.tile(np.expand_dims(np_image, -1), 3)
        return Image.fromarray(tiled_image)
    else:
        return Image.fromarray(np_image)

def train_transforms(example_batch):
    images = [jitter(handle_grayscale_image(x)) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs


def val_transforms(example_batch):
    images = [handle_grayscale_image(x) for x in example_batch["image"]]
    labels = [x for x in example_batch["annotation"]]
    inputs = image_processor(images, labels)
    return inputs

train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

##################### METRICS #####################

metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad(): # Don't want to store the gradients while computing this metric since it's validation
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        # scale the logits to the size of the label
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        # currently using _compute instead of compute
        # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=len(id2label),
            ignore_index=0,
            reduce_labels=image_processor.do_reduce_labels,
        )
        # add per category metrics as individual key-value pairs
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

        return metrics
    
##################### LORA MODEL #####################

config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="lora_only",
    modules_to_save= ["decode_head"],
    use_rslora = True # scaling factor alpha/sqrt(r) instead of sqrt(r)
)
lora_model = get_peft_model(model, config)

##################### TRAINING #####################

model_name = model_id.split("/")[-1]

lr = 5e-4
epochs = 100
batch_size = 4
training_args = TrainingArguments(
    output_dir=f"LORA_test2/{model_name}-cityscapes",
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    remove_unused_columns=False,
    label_names=["labels"]
)

optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = (len(train_ds) // batch_size) * epochs
scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)


trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler)
)

"""
# if we want to see the trainable parameters
def count_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros entrenables: {trainable_params}")
    print(f"Parámetros totales: {total_params}")
    print(f"Porcentaje de parámetros entrenables: {100 * trainable_params / total_params:.2f}%")
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parámetro entrenable: {name}, Tamaño: {param.size()}")


count_trainable_parameters(lora_model)
"""
trainer.train()

model_id = "segformerB4-gta-cityscapes-lora-(r=32-alpha=16,dropout=0.1,target_modules=[query,value],modules_to_save=[decode_head]],use_rslora, scheduler)"
lora_model.save_pretrained(f'./{model_id}')
