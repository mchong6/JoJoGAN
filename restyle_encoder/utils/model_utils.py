# specify the encoder types for pSp and e4e - this is mainly used for the inference scripts
ENCODER_TYPES = {
    'pSp': ['GradualStyleEncoder', 'ResNetGradualStyleEncoder', 'BackboneEncoder', 'ResNetBackboneEncoder'],
    'e4e': ['ProgressiveBackboneEncoder', 'ResNetProgressiveBackboneEncoder']
}

RESNET_MAPPING = {
    'layer1.0': 'body.0',
    'layer1.1': 'body.1',
    'layer1.2': 'body.2',
    'layer2.0': 'body.3',
    'layer2.1': 'body.4',
    'layer2.2': 'body.5',
    'layer2.3': 'body.6',
    'layer3.0': 'body.7',
    'layer3.1': 'body.8',
    'layer3.2': 'body.9',
    'layer3.3': 'body.10',
    'layer3.4': 'body.11',
    'layer3.5': 'body.12',
    'layer4.0': 'body.13',
    'layer4.1': 'body.14',
    'layer4.2': 'body.15',
}
