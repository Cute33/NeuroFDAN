"""服务器CPU测试：归一化范围、默认兼容、梯度、标量退化与训练字段。"""
import sys
import unittest
import contextlib
import io
import json
import tempfile
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path
import torch
from frontdoor_scm import FrontDoorConfig, FrontDoorEstimator, GaussianMediator
from ModelTrain import normalized_model_config
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cns_norm_synthetic import view, fit
import cns_norm_synthetic as n0
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

    def test_evaluation_view_has_mediators_but_observation_training_does_not(self):
        import numpy as np
        raw={k:np.ones((2,4)) for k in ('x_c','x_a','m_c','m_a','u')};raw['y']=np.array([0,1])
        self.assertEqual(set(n0.evaluation_view(raw)), {'x_c','x_a','y','m_c','m_a'})
        self.assertEqual(set(view(raw)), {'x_c','x_a','y'})
        with self.assertRaises(ValueError): fit(n0.evaluation_view(raw),'none',False,1,42)

    def test_full_n0_entry_completes_all_groups_and_keeps_training_permissions(self):
        training_keys=[]
        original_fit=n0.fit
        def record_fit(data,norm,oracle,steps,seed):
            training_keys.append((oracle,set(data)))
            return original_fit(data,norm,oracle,steps,seed)
        threads=torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as root, torch.random.fork_rng(devices=[]):
                output=Path(root)/'n0'
                with patch.object(sys,'argv',['n0','--output_dir',str(output),'--steps','1']), \
                     patch.object(n0,'fit',side_effect=record_fit), contextlib.redirect_stdout(io.StringIO()):
                    n0.main()
                report=json.loads((output/'probe_summary.json').read_text())
                self.assertTrue(report['completed'])
                self.assertEqual([r['name'] for r in report['rows']],
                    ['observation_only','observation_norm','oracle_mediator'])
                self.assertEqual(len({r['initial_parameter_sha256'] for r in report['rows']}),1)
                self.assertEqual(training_keys,[(False,{'x_c','x_a','y'}),
                    (False,{'x_c','x_a','y'}),(True,{'x_c','x_a','y','m_c','m_a'})])
                for row in report['rows']:
                    for key in ('observed_nll','do_mae','conditional_nll_c','conditional_nll_a'):
                        self.assertTrue(torch.isfinite(torch.tensor(row[key])))
                self.assertFalse(report['automatically_start_real_training'])
                self.assertIn('amplitude_counterexample',report)
        finally:
            torch.set_num_threads(threads)
