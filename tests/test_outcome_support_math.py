"""支持/后验诊断的数学小测试；不加载NeuroFDAN或生理数据，本轮云端未运行。"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_outcome_support import mixture_logp, posterior_statistics, marginal_ans_moments, scaled_distance


class OutcomeSupportMathTests(unittest.TestCase):
    def test_single_component_density_matches_diagonal_normal(self):
        x=torch.tensor([[0.,1.],[2.,-1.]],dtype=torch.double)
        mu=torch.tensor([[[.2,.3]]],dtype=torch.double)
        lv=torch.tensor([[[.1,-.4]]],dtype=torch.double)
        expected=torch.distributions.Normal(mu[0,0],(.5*lv[0,0]).exp()).log_prob(x).sum(-1)
        self.assertTrue(torch.allclose(mixture_logp(x,mu,lv)[:,0],expected))

    def test_equal_mixture_component_permutation_does_not_change_density(self):
        x=torch.tensor([[1.,2.]])
        mu=torch.tensor([[[0.,0.],[2.,2.]]]);lv=torch.zeros_like(mu)
        self.assertTrue(torch.allclose(mixture_logp(x,mu,lv),mixture_logp(x,mu.flip(1),lv.flip(1))))

    def test_constant_mean_can_have_input_dependent_sigma(self):
        mu=torch.zeros(4,2);lv=torch.tensor([[0.,0.],[.2,.2],[.4,.4],[.6,.6]])
        stats=posterior_statistics(mu,lv)
        self.assertEqual(stats['var_X_mu']['max'],0.)
        self.assertGreater(stats['var_X_sigma']['min'],0.)
        prior=posterior_statistics(mu,torch.zeros_like(mu))
        self.assertEqual(prior['gaussian_conditional_kl_float64']['max'],0.)

    def test_total_variance_includes_conditional_mean_variation(self):
        mu=torch.tensor([[[-1.],[1.]],[[0.],[0.]]]);lv=torch.zeros_like(mu)
        stats=marginal_ans_moments(mu,lv)
        self.assertEqual(stats['E_conditional_variance']['mean'],1.)
        self.assertEqual(stats['Var_Mc_conditional_mean']['max'],1.)
        self.assertGreater(stats['var_X_total_std']['mean'],0.)

    def test_distance_normalization_and_nonnegative_roundoff(self):
        x=torch.tensor([[1.,2.],[3.,4.]])
        d=scaled_distance(x,x,2.)
        self.assertTrue((d>=0).all())
        self.assertTrue(torch.allclose(d.diag(),torch.zeros(2,dtype=torch.double)))
        self.assertAlmostEqual(float(d[0,1]),1.)


if __name__=='__main__':
    unittest.main()
