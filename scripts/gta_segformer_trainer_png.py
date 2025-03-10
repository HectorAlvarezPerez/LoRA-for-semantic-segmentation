# Dataset
from datasets import load_dataset, Dataset
import os
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

imagesGTA = "./all_gta_images.txt"
masksGTA = "./all_gta_masks.txt"

with open(imagesGTA, 'r') as f:
    imageGTA_paths = [line.strip() for line in f.readlines()]

with open(masksGTA, 'r') as f:
    maskGTA_paths = [line.strip() for line in f.readlines()]

dataGTA = {
    "image": imageGTA_paths,
    "mask": maskGTA_paths
}

if len(dataGTA["image"]) != len(dataGTA["mask"]):
    raise ValueError("El número de imágenes no coincide con el número de máscaras.")

hf_datasetGTA = Dataset.from_dict(dataGTA)


imagesCityscapes = "./all_cityscapes_test_images_1000_run_3.txt"
masksCityscapes = "./all_cityscapes_test_masks_1000_run_3.txt"

with open(imagesCityscapes, 'r') as f:
    imagesCityscapes_paths = [line.strip() for line in f.readlines()]

with open(masksCityscapes, 'r') as f:
    masksCityscapes_paths = [line.strip() for line in f.readlines()]

dataCityscapes = {
    "image": imagesCityscapes_paths,
    "mask": masksCityscapes_paths
}

hf_datasetCityscapes = Dataset.from_dict(dataCityscapes)

train_ds = hf_datasetGTA.train_test_split(test_size=0.1)['train']
test_ds = hf_datasetGTA.train_test_split(test_size=0.1)['test']


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

##################### TRANSFORMATIONS #####################
    
# Prepare the transformations
processor = SegformerImageProcessor()
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)  # Augmentation technique
flip = RandomHorizontalFlip(p=0.5)  # Augmentation technique

def train_transforms(example_batch):
    images = [Image.open(image_path).convert('RGB') for image_path in example_batch['image']]
    labels = [Image.open(mask_path).convert('L') for mask_path in example_batch['mask']]

    
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
    images = [Image.open(image_path).convert('RGB') for image_path in example_batch['image']]
    labels = [Image.open(mask_path).convert('L') for mask_path in example_batch['mask']]
    

    inputs = processor(images=images, segmentation_maps=labels, return_tensors="pt")
    return inputs


# Set transforms
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)

##################### MODEL #####################

pretrained_model_name = "nvidia/mit-b5" # trained in ImageNet-1k
model = SegformerForSemanticSegmentation.from_pretrained(
    pretrained_model_name,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)
##################### TRAINING #####################

epochs = 20
lr = 0.00006
batch_size = 2

# Calculate the total steps
total_examples = len(train_ds)
total_steps = (total_examples // batch_size) * epochs
print("Total calculated steps:", total_steps)

name = "segformerb5-gta"

training_args = TrainingArguments(
    output_dir= "./models/" + name,
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
model.save_pretrained(f'./models/{name}-prova')

### INFERENCIA ###
"""
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = logits.argmax(dim=1).detach().cpu().numpy()[0]

real_mask = Image.open(mask_path).convert("L")
real_mask = np.array(real_mask)  

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title("Imagen original")
axes[0].axis("off")

axes[1].imshow(real_mask, cmap="jet")
axes[1].set_title("Máscara real")
axes[1].axis("off")

axes[2].imshow(predictions, cmap="jet")
axes[2].set_title("Máscara predicha")
axes[2].axis("off")

plt.show()
"""