"""Candidate checkpoint roster for the x9 ImageNet-scale pool.

Deliberately over-complete: tags here are CANDIDATES, and `verify_manifest.py`
prunes them against the installed timm registry (wrong guesses are expected
and harmless; the verified manifest is the authoritative artifact). Targets
per the plan (broad pool, ~45): ResNet-50 8-12, ViT-B/16+DeiT 8-10,
Swin 4-6, ConvNeXt 4-6, breadth 6-10.

All candidates must be ImageNet-1k classifiers at 224x224 with a standard
classifier head (num_classes=1000); the verifier enforces this.
"""

# (family, timm tag, recipe note)
CANDIDATES = [
    # ---- ResNet-50: one architecture, many training recipes ----
    ("resnet50", "resnet50.tv_in1k", "torchvision v1 recipe"),
    ("resnet50", "resnet50.tv2_in1k", "torchvision v2 recipe"),
    ("resnet50", "resnet50.a1_in1k", "ResNet strikes back A1"),
    ("resnet50", "resnet50.a2_in1k", "ResNet strikes back A2"),
    ("resnet50", "resnet50.a3_in1k", "ResNet strikes back A3"),
    ("resnet50", "resnet50.b1k_in1k", "rsb B1k"),
    ("resnet50", "resnet50.b2k_in1k", "rsb B2k"),
    ("resnet50", "resnet50.c1_in1k", "rsb C1"),
    ("resnet50", "resnet50.c2_in1k", "rsb C2"),
    ("resnet50", "resnet50.d_in1k", "rsb D"),
    ("resnet50", "resnet50.ram_in1k", "timm RandAug+mixup era"),
    ("resnet50", "resnet50.am_in1k", "timm AugMix era"),
    ("resnet50", "resnet50.ra_in1k", "timm RandAug era"),
    ("resnet50", "resnet50.gluon_in1k", "GluonCV recipe"),
    ("resnet50", "resnet50.fb_ssl_yfcc100m_ft_in1k", "semi-supervised YFCC pretrain, in1k ft"),
    ("resnet50", "resnet50.fb_swsl_ig1b_ft_in1k", "weakly-supervised IG-1B pretrain, in1k ft"),
    # ---- ViT-B/16 and DeiT family ----
    ("vit_b16", "vit_base_patch16_224.augreg_in1k", "AugReg, in1k only"),
    ("vit_b16", "vit_base_patch16_224.augreg_in21k_ft_in1k", "AugReg, in21k pretrain"),
    ("vit_b16", "vit_base_patch16_224.augreg2_in21k_ft_in1k", "AugReg2, in21k pretrain"),
    ("vit_b16", "vit_base_patch16_224.orig_in21k_ft_in1k", "original JFT-era recipe"),
    ("vit_b16", "vit_base_patch16_224.sam_in1k", "SAM optimizer"),
    ("vit_b16", "deit_base_patch16_224.fb_in1k", "DeiT"),
    ("vit_b16", "deit_base_distilled_patch16_224.fb_in1k", "DeiT distilled"),
    ("vit_b16", "deit3_base_patch16_224.fb_in1k", "DeiT3, in1k only"),
    ("vit_b16", "deit3_base_patch16_224.fb_in22k_ft_in1k", "DeiT3, in22k pretrain"),
    # ---- Swin ----
    ("swin", "swin_tiny_patch4_window7_224.ms_in1k", "Swin-T, in1k"),
    ("swin", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "Swin-T, in22k pretrain"),
    ("swin", "swin_small_patch4_window7_224.ms_in1k", "Swin-S, in1k"),
    ("swin", "swin_small_patch4_window7_224.ms_in22k_ft_in1k", "Swin-S, in22k pretrain"),
    ("swin", "swin_base_patch4_window7_224.ms_in1k", "Swin-B, in1k"),
    ("swin", "swin_base_patch4_window7_224.ms_in22k_ft_in1k", "Swin-B, in22k pretrain"),
    # ---- ConvNeXt ----
    ("convnext", "convnext_tiny.fb_in1k", "ConvNeXt-T, in1k"),
    ("convnext", "convnext_tiny.fb_in22k_ft_in1k", "ConvNeXt-T, in22k pretrain"),
    ("convnext", "convnext_small.fb_in1k", "ConvNeXt-S, in1k"),
    ("convnext", "convnext_small.fb_in22k_ft_in1k", "ConvNeXt-S, in22k pretrain"),
    ("convnext", "convnext_base.fb_in1k", "ConvNeXt-B, in1k"),
    ("convnext", "convnext_base.fb_in22k_ft_in1k", "ConvNeXt-B, in22k pretrain"),
    # ---- breadth: widen the pool's NC range ----
    ("breadth", "vgg13.tv_in1k", "VGG-13 at ImageNet scale (training-pool architecture)"),
    ("breadth", "vgg16.tv_in1k", "VGG-16"),
    ("breadth", "densenet121.ra_in1k", "DenseNet-121"),
    ("breadth", "densenet121.tv_in1k", "DenseNet-121 torchvision"),
    ("breadth", "efficientnet_b0.ra_in1k", "EfficientNet-B0"),
    ("breadth", "efficientnet_b3.ra2_in1k", "EfficientNet-B3"),
    ("breadth", "mobilenetv3_large_100.ra_in1k", "MobileNetV3-L"),
    ("breadth", "regnety_032.ra_in1k", "RegNetY-3.2GF"),
    ("breadth", "resnext50_32x4d.a1_in1k", "ResNeXt-50 rsb A1"),
    ("breadth", "resnext50_32x4d.tv_in1k", "ResNeXt-50 torchvision"),
    ("breadth", "wide_resnet50_2.tv_in1k", "WideResNet-50-2"),
    ("breadth", "maxvit_tiny_tf_224.in1k", "MaxViT-T"),
    ("breadth", "inception_v3.tv_in1k", "Inception-v3"),
]
