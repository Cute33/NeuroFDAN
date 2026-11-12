"""独立LIF宽度、旧配置兼容和长程CLI的CPU测试，待服务器运行。"""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch

from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, frontdoor_loss
from ModelTrain import normalized_model_config, load_checkpoint, validate_resume_objective
from mainNeuroFDAN import get_args


class LongProfileTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(1)

    def test_cli_dataset_defaults_and_explicit_override(self):
        for ds,batch in [('DEAP',400),('HCI',100)]:
            a=get_args(argv=['--dataset',ds,'--label','arousal'])
            self.assertEqual((a.batch_size,a.eval_batch_size),(batch,batch))
            self.assertEqual((a.lr,a.epochs,a.fire_weight,a.lif_neurons),(.00002,300,2.8,256))
        a=get_args(argv=['--label','arousal','--lif_neurons','128','--lr','.00001',
                         '--epochs','500','--eval_batch_size','40'])
        self.assertEqual((a.lif_neurons,a.lr,a.epochs,a.eval_batch_size),(128,.00001,500,40))

    def test_lif_width_separate_from_hidden_and_mediator(self):
        cfg=FrontDoorConfig(hidden_dim=8,lif_neurons=16,mediator_dim=4,S_train=2,dropout=0.)
        model=FrontDoorEstimator(5,3,cfg)
        x,a=torch.randn(3,5),torch.randn(3,3)
        out=model(x,a,fd_mode='none')
        self.assertEqual(model.cns.encoder[0].out_features,8)
        self.assertEqual(model.cns.snn.cell.num_neurons,16)
        self.assertEqual(model.ans.cns_current.out_features,16)
        self.assertEqual(out['mu_c'].shape,(3,4))
        self.assertEqual(out['mu_a'].shape,(3,2,4))
        self.assertEqual(model.outcome.net[0].out_features,8)
        loss=frontdoor_loss(out,torch.tensor([0,1,0]),cfg)['loss']
        opt=torch.optim.Adam(model.parameters(),lr=.00002)
        opt.zero_grad();loss.backward();opt.step()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_legacy_width_preserves_initial_weights(self):
        old=FrontDoorConfig(hidden_dim=8,mediator_dim=4)
        explicit=FrontDoorConfig(hidden_dim=8,mediator_dim=4,lif_neurons=8)
        torch.manual_seed(4);a=FrontDoorEstimator(5,3,old)
        torch.manual_seed(4);b=FrontDoorEstimator(5,3,explicit)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        state=asdict(old);state.pop('lif_neurons')
        self.assertEqual(normalized_model_config(state),normalized_model_config(asdict(explicit)))

    def test_resume_lr_and_fire_mismatch_rejected(self):
        state=dict(optimizer=dict(param_groups=[dict(lr=.00002)]),arguments=dict(lambda_obs=1.,fire_weight=2.8))
        args=SimpleNamespace(lr=.00002,lambda_obs=1.,fire_weight=2.8)
        validate_resume_objective(state,args,'DEAP')
        args.lr=.00001
        with self.assertRaises(ValueError):validate_resume_objective(state,args,'DEAP')
        args.lr=.00002;args.fire_weight=0.
        with self.assertRaises(ValueError):validate_resume_objective(state,args,'DEAP')

    def test_checkpoint_roundtrip_and_legacy_load(self):
        from NeuroFDAN import NeuroFDAN
        cfg=FrontDoorConfig(hidden_dim=8,mediator_dim=4,lif_neurons=16)
        model=NeuroFDAN(ecg_channels=2,frontdoor_config=cfg)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pt'
            torch.save(dict(format_version=2,frontdoor_config=asdict(cfg),model=model.state_dict()),path)
            restored=NeuroFDAN(ecg_channels=2,frontdoor_config=cfg)
            load_checkpoint(path,restored)
            self.assertTrue(all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()))
            legacy_cfg=FrontDoorConfig(hidden_dim=8,mediator_dim=4)
            legacy=NeuroFDAN(ecg_channels=2,frontdoor_config=legacy_cfg)
            config=asdict(legacy_cfg);config.pop('lif_neurons')
            torch.save(dict(format_version=2,frontdoor_config=config,model=legacy.state_dict()),path)
            load_checkpoint(path,NeuroFDAN(ecg_channels=2,frontdoor_config=FrontDoorConfig(hidden_dim=8,mediator_dim=4,lif_neurons=8)))
            with self.assertRaises(ValueError):load_checkpoint(path,restored)

    def test_fire_target_formula_remains_two_branch_average(self):
        from NeuroFDAN import NeuroFDAN
        cfg=FrontDoorConfig(hidden_dim=8,lif_neurons=16,mediator_dim=4,S_train=2,ref_dim=8)
        model=NeuroFDAN(ecg_channels=2,frontdoor_config=cfg)
        out=model(torch.randn(2,32,768),torch.randn(2,3,768),fd_mode='none')
        c=model.frontdoor.cns.snn.last_rates; a=model.frontdoor.ans.snn.last_rates
        expected=.5*((c.mean(0)-.05).square().mean()+(a.mean(0)-.05).square().mean())
        self.assertTrue(torch.equal(out['fire_loss'],expected))


if __name__=='__main__':unittest.main()
