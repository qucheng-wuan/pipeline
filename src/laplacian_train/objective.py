"""Joint final-depth and independent-normal objectives; labels stay here."""
import torch
import torch.nn.functional as F
from .losses import independent_normal_geometry, normal_pyramid_loss


def mean_images(value, mask):
    mask=mask.bool()
    clean=torch.where(mask,value,torch.zeros_like(value))
    count=mask.flatten(1).sum(1)
    means=clean.flatten(1).sum(1)/count.clamp_min(1)
    present=count>0
    return (means*present).sum()/present.sum().clamp_min(1)


def labels_for_geometry(labels):
    return {'target':labels['target'],'valid':labels['valid'],'transparent':labels['transparent'],
        'intrinsics':labels['K'],'normal_camera_valid':labels['camera_valid'],
        'target_normal':labels['target_normal'],'normal_valid':labels['normal_valid'],
        'normal_reference_available':torch.ones_like(labels['camera_valid'],dtype=torch.bool)}


def objective(depth, gates, labels, pyramid_weight=.05):
    valid=labels['valid'].bool() & labels['core'].bool()
    foreground=valid & labels['transparent'].bool()
    background=valid & ~labels['transparent'].bool()
    truth=torch.where(valid,labels['target'].float(),torch.ones_like(depth))
    pred=torch.where(valid,depth,torch.ones_like(depth))
    logerror=pred.log()-truth.log()
    huber=F.huber_loss(logerror,torch.zeros_like(logerror),delta=.1,reduction='none')
    relative=(pred-truth).abs()/truth
    metric=mean_images(huber+.1*relative,valid)
    regional=.25*(mean_images(huber,foreground)+mean_images(huber,background))
    geometry=independent_normal_geometry(depth,labels_for_geometry(labels))
    support=geometry['valid_support'] & labels['core'].bool()
    angle=1-(geometry['predicted_normal']*geometry['target_normal']).sum(1,keepdim=True).clamp(-1,1)
    angular=mean_images(angle,support)+.5*mean_images(angle,support & labels['transparent'].bool())
    if pyramid_weight:
        pyramid=normal_pyramid_loss(geometry['predicted_normal'],geometry['target_normal'],
            geometry['valid_support'],focus_mask=1+.5*labels['transparent'].float(),core_mask=labels['core'])
        frequency=pyramid['loss']
    else:
        frequency=depth.sum()*0
    regularization=gates.square().mean()
    loss=metric+regional+.2*angular+pyramid_weight*frequency+.001*regularization
    return loss,{'loss':loss.detach(),'metric':metric.detach(),'regional':regional.detach(),
        'angular':angular.detach(),'pyramid':frequency.detach(),'gate_regularization':regularization.detach(),
        'normal_pixels':support.sum().detach(),'depth_mae_mm':mean_images((pred-truth).abs()*1000,valid).detach()}
