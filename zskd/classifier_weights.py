import torch


def extract_classifier_weights(model: torch.nn.Module) -> torch.Tensor:
    """Assuming that the classifier is the last named parameter of a module, 
    this function returns the weights of the classifier."""
    
    # Find last layer
    classifier = tuple(model.children())[-1]
    
    while isinstance(classifier, torch.nn.Sequential):
        classifier = tuple(classifier.children())[-1]

    if isinstance(classifier, torch.nn.Linear):
        return classifier.weight
    else:
        return next(iter(classifier.parameters()))