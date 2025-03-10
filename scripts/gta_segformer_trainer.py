# Dataset
from datasets import load_dataset
# Labels
import json
from huggingface_hub import cached_download, hf_hub_url
# Transformations
from torchvision.transforms import ColorJitter, RandomHorizontalFlip
from transformers import SegformerImageProcessor
from PIL import Image
import numpy as np
# Model training
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer
import transformers
# Metric
import torch
from torch import nn
import evaluate

##################### DATASET #####################

# Load dataset from hugginface
hf_datasets = load_dataset("guimCC/gta5-cityscapes-labeling")

# Split the dataset
train_ds = hf_datasets["train"]
test_ds = hf_datasets["test"].train_test_split(test_size=0.1)['test']
val_ds = hf_datasets["validation"].train_test_split(test_size=0.1)['test']]

##################### IDs and LABELs #####################


# Download the label files
repo_id = "huggingface/label-files"
filename = "cityscapes-id2label.json"
id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r")) 
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# Add ignore label
id2label[19] = 'ignore'
label2id['ignore'] = 19

num_labels = len(id2label)

# Prepare the transformations
processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)  # Augmentation technique
flip = RandomHorizontalFlip(p=0.5)  # Augmentation technique

    
def train_transforms(example_batch):
    images = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch['image']]
    labels = [Image.fromarray(np.array(x, dtype=np.uint8), mode='L') for x in example_batch['mask']]
    
    augmented_images = []
    augmented_labels = []
    for img, label in zip(images, labels):
        img = jitter(img)  
        if np.random.rand() > 0.5:  
            img = flip(img)
            label = flip(label)
        augmented_images.append(img)
        augmented_labels.append(label)
    
    labels = [Image.fromarray(np.minimum(np.array(label), num_labels - 1), mode='L') for label in augmented_labels]
    
    inputs = processor(images=augmented_images, segmentation_maps=labels, return_tensors="pt")
    return inputs    
    
def val_transforms(example_batch):
    images = [Image.fromarray(np.array(x, dtype=np.uint8)) for x in example_batch['image']]
    labels = [Image.fromarray(np.array(x, dtype=np.uint8), mode='L') for x in example_batch['mask']]
    
    labels = [Image.fromarray(np.minimum(np.array(label), num_labels - 1), mode='L') for label in labels]
    
    inputs = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)
val_ds.set_transform(val_transforms)

##################### MODEL #####################

pretrained_model_name = "nvidia/mit-b4" # trained in ImageNet-1k
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
##################### TRAINING #####################

epochs = 100
lr = 0.00004
batch_size = 4

name = "segformerb4-gta"

training_args = TrainingArguments(
    output_dir=name,
    overwrite_output_dir=False,
    learning_rate=lr,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=1,
    eval_accumulation_steps=10,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_num_workers=15,
)

metric = evaluate.load("mean_iou") 

def compute_metrics(eval_pred):
  with torch.no_grad():
    logits, labels = eval_pred
    logits_tensor = torch.from_numpy(logits)
    # scale the logits to the size of the label
    logits_tensor = nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:], # height, width
        mode="bilinear", 
        align_corners=False,
    ).argmax(dim=1)

    
    pred_labels = logits_tensor.detach().cpu().numpy()
    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(id2label),
        ignore_index=19,
        reduce_labels=processor.do_reduce_labels,
    )
    
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
    return metrics

optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
total_steps = (len(train_ds) // batch_size) * epochs

scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    tokenizer=None,
    optimizers=(optimizer, scheduler) # first for optimizer, second for scheduler
)

# Train the model
transformers.logging.set_verbosity_info()
trainer.train()

# Save the model
model.save_pretrained(f'./{name}')
