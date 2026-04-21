import argparse
import yaml
import torch
from Mahalanobis.sptnet_wrapper import SPTNetWrapper
from Mahalanobis.data_loader import create_loaders, create_ood_test_loader
from Mahalanobis.pipeline import MahalanobisPipeline
from Mahalanobis.mahalanobis_scorer import MahalanobisScorer
from Mahalanobis.utils import get_default_transform

def main():
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'])
    
    args = argparse.Namespace()
    args.dataset_name = config['dataset_name']
    args.batch_size = config['batch_size']
    args.num_workers = config['num_workers']
    args.transform = 'imagenet'
    args.image_size = 224
    args.n_views = 2
    
    args.prop_train_labels = 0.5
    args.use_ssb_splits = True
    args.grad_from_block = 11
    args.lr = 0.1
    args.lr2 = 0.1
    args.momentum = 0.9
    args.weight_decay = 1e-4
    args.epochs = 200
    args.sup_weight = 0.35
    args.lamb = 0.1
    args.model = 'dino'
    args.freq_rep_learn = 20
    args.prompt_size = 1
    args.prompt_type = 'all'
    args.memax_weight = 2
    args.warmup_teacher_temp = 0.07
    args.teacher_temp = 0.04
    args.warmup_teacher_temp_epochs = 30
    args.fp16 = True
    args.eval_freq = 1
    
    # Create data loaders
    labelled_train_loader, val_loader, test_loader = create_loaders(args, val_split=config.get('val_split', 0.2))
    
    # Initialize SPTNet wrapper
    sptnet = SPTNetWrapper(checkpoint_path=config['checkpoint_path'], prompt_type=config['prompt_type'], device=device)
    
    scorer = MahalanobisScorer(device=device)
    pipeline  = MahalanobisPipeline(sptnet, scorer, device=device)
    
    pipeline.compute_stats_from_loader(labelled_train_loader, num_classes=config['num_classes'])
    
    pipeline.select_threshold(val_loader, method='tpr', desired_tpr=config.get('desired_tpr', 0.95))
    
    test_trasform = get_default_transform()
    
    ood_config = config.get('ood_config')
    
    ood_test_loaders = create_ood_test_loader(ood_config, test_trasform, batch_size=config['batch_size'], num_workers=config['num_workers'])
    if ood_test_loaders is None:
        return
    
    # Evaluate on OOD dataset
    all_ood_scores = []
    for name, ood_loader in ood_test_loaders.items():
        print(f"Computing scores for OOD dataset: {name}")
        ood_scores = pipeline.compute_scores_for_loader(ood_loader)
        all_ood_scores.extend(ood_scores)

    # Calculate metrics in all OOD datasets
    metrics = pipeline.evaluate(test_loader, np.array(all_ood_scores))
    print(f"Final combined AUROC: {metrics['auc']:.4f}, FPR95: {metrics['fpr95']:.4f}")
           
if __name__ == "__main__":
    main()
