"""服务器CPU测试：归一化范围、默认兼容、梯度、标量退化与训练字段。"""
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, GaussianMediator
from ModelTrain import normalized_model_config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cns_norm_synthetic import view, fit
from scheme_a_fit_check import fit_gate


class CNSInputNormTests(unittest.TestCase):
    def models(self):
        c=FrontDoorConfig(mediator_dim=4, hidden_dim=8, lif_neurons=8, dropout=0.)
        torch.manual_seed(42); a=FrontDoorEstimator(16,8,c)
        torch.manual_seed(42); b=FrontDoorEstimator(16,8,replace(c,cns_input_norm='layernorm'))
        return a,b

    def test_same_parameters_and_initial_rng_no_affine_parameters(self):
        a,b=self.models()
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(all(torch.equal(a.state_dict()[k],v) for k,v in b.state_dict().items()))
        self.assertEqual(list(b.cns.input_norm.parameters()),[])
        self.assertFalse(hasattr(b.ans,'input_norm'))

    def test_default_distribution_exactly_matches_original_gaussian_path(self):
        a,_=self.models(); x=torch.randn(3,16)
        original=GaussianMediator.distribution(a.cns,x)
        actual=a.cns.distribution(x)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(original,actual)))

    def test_only_cns_entry_changes_reference_is_identical_and_gradient_finite(self):
        a,b=self.models(); x=torch.randn(3,16,requires_grad=True); xa=torch.randn(3,8)
        mu,lv=b.cns.distribution(x)
        expected=GaussianMediator.distribution(b.cns,b.cns.input_norm(x))
        self.assertTrue(all(torch.equal(v,w) for v,w in zip((mu,lv),expected)))
        torch.testing.assert_close(a.reference_encoder(x,xa),b.reference_encoder(x,xa),rtol=0,atol=0)
        (mu.square().mean()+lv.square().mean()).backward()
        self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(float(x.grad.abs().sum()),0.)

    def test_scalar_rejected_and_old_checkpoint_norm_default_is_none(self):
        c=FrontDoorConfig(cns_input_norm='layernorm')
        with self.assertRaises(ValueError): FrontDoorEstimator(1,8,c)
        old={'hidden_dim':8,'lif_neurons':8}
        self.assertEqual(normalized_model_config(old)['cns_input_norm'],'none')
        self.assertNotEqual(normalized_model_config(old),normalized_model_config(dict(old,cns_input_norm='layernorm')))

    def test_observation_training_view_excludes_hidden_fields(self):
        import numpy as np
        raw={k:np.ones((2,4)) for k in ('x_c','x_a','m_c','m_a','u')};raw['y']=np.array([0,1])
        self.assertEqual(set(view(raw)),{'x_c','x_a','y'})
        with self.assertRaises(ValueError):fit(view(raw,True),'none',False,1,42)

    def test_acc_gate_rejects_majority_only_predictions(self):
        good=dict(acc=.9,min_class_recall=.7,integrated_observational_nll=.3)
        self.assertTrue(fit_gate([good]*3,.665,metric='acc'))
        bad=dict(good,acc=.95,min_class_recall=0.)
        self.assertFalse(fit_gate([bad]*3,.665,metric='acc'))

    def test_amplitude_shift_loss_is_explicit(self):
        _,b=self.models(); x=torch.arange(16,dtype=torch.float32)[None]
        torch.testing.assert_close(b.cns.input_norm(x),b.cns.input_norm(x+4.))
