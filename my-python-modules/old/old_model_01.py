import pytorch_lightning as pl 
# import transformers as t
# from transformers import DetrForObjectDetection
from transformers import DetrForObjectDetection, DetrImageProcessor



import torch
# import torchvision 
# from torchvision.models.detection import detr
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.detr_resnet50 import detr_resnet50

class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay, 
                 pretrained_model_name_or_path, num_labels):
        super().__init__()
        # pretrained_model_name_or_path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model/detr/detr-r50-e632da11.pth'
        # pretrained_model_name_or_path = 'detr-r50-e632da11.pth'
        
        # pretrained_model_name_or_path = 
        # '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model/detr-resnet-50-model'
        # pretrained_model_name_or_path = 'detr-resnet-50'

        # self.model = DetrForObjectDetection.from_pretrained(
        #     '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model/detr-resnet-50-model',
        #     revision="no_timm",
        #     )

        # image_processor = DetrImageProcessor.from_pretrained(
        #     'facebook/detr-resnet-50'
        #     )
        # self.model = DetrForObjectDetection.from_pretrained(
        #     'facebook/detr-resnet-50'
        #     )

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model/detr-resnet-50-model',
            cache_dir='/home/lovelace/proj/proj939/rubenscp/.cache/huggingface/hub'
        )
            # cache_dir='/home/lovelace/proj/proj939/rubenscp/.cache/huggingface/hub/models--facebook-detr-resnet-50'
        # self.model = DetrForObjectDetection.from_pretrained(
        #     '/home/lovelace/proj/proj939/rubenscp/.cache/huggingface/hub/models--facebook-detr-resnet-50',
        #     # revision="no_timm",
        # )

        # self.model = DetrForObjectDetection.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path, 
        #     revision="no_timm",
        #     # num_labels=len(id2label),
        #     num_labels=num_labels,            
        #     ignore_mismatched_sizes=True
        # )
        # self.model = detr.detr.resnet50(pretrained=True, num_classes=num_labels)
        
        # self.model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        # self.model = torch.hub.load(
        #     '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model',
        #     # 'detr-r50-e632da11', 
        #     'detr-r50-e632da11.pth', 
        #     # 'detr_resnet50',
        #     source='local',
        #     pretrained=True
        #     )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())
            
        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here: 
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    # def train_dataloader(self):
    #     return TRAIN_DATALOADER

    # def val_dataloader(self):
    #     return VAL_DATALOADER