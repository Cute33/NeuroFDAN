"""N1来源/单因素和初始化守卫；服务器CPU执行。"""
import copy
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cns_norm_pairing import check_arguments, check_sources, parameter_hash
from frontdoor_scm import FrontDoorConfig
from NeuroFDAN import NeuroFDAN


def fixture():
    old=dict(dataset='DEAP',label='arousal',held_out_subjects=[2],seed=42,beta=0.,lr=.00002,
             fire_weight=.028,batch_size=400,output_dir='/old',run_name='old')
    base=dict(source_version={'commit':'bfe108e703b5d089fc53d59d254b61d6fcbbb885'},
        completed=True,passed=False,actual_steps=300,arguments=old,
        fit_controls={'fit_max_steps':300,'fit_check_every':50},
        selected_source_ids=[[1,1,0]],reference_source_ids=[[3,1,0]],true_counts=[153,247],full_epoch_steps=31,
        model=asdict(FrontDoorConfig(beta=0.,lif_neurons=256)))
    new=copy.deepcopy(base);new['arguments'].update(cns_input_norm='layernorm',output_dir='/new',run_name='new')
    new['model']['cns_input_norm']='layernorm'
    controls=dict(fit_max_steps=300,fit_check_every=50,fit_gate_metric='acc')
    return base,new,controls


class PairingTests(unittest.TestCase):
    def test_only_normalization_may_change(self):
        b,n,c=fixture();check_arguments(b,n['arguments'],c)
        for key,value in [('lr',.001),('beta',.001),('fire_weight',0.),('batch_size',40)]:
            with self.assertRaises(ValueError):check_arguments(b,dict(n['arguments'],**{key:value}),c)

    def test_budget_and_acc_metric_guard(self):
        b,n,c=fixture()
        for change in [dict(fit_max_steps=500),dict(fit_gate_metric='balanced_accuracy')]:
            with self.assertRaises(ValueError):check_arguments(b,n['arguments'],dict(c,**change))

    def test_sources_and_model_config_guard(self):
        b,n,c=fixture();check_sources(b,n)
        for key in ('selected_source_ids','reference_source_ids','true_counts'):
            with self.assertRaises(ValueError):check_sources(b,dict(n,**{key:[]}))
        bad=copy.deepcopy(n);bad['model']['S_train']=8
        with self.assertRaises(ValueError):check_sources(b,bad)

    def test_none_template_and_norm_initialization_match_without_rng_consumption(self):
        cfg=FrontDoorConfig(hidden_dim=8,lif_neurons=8,mediator_dim=4,cns_input_norm='layernorm')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42);before=torch.random.get_rng_state().clone()
            with torch.random.fork_rng(devices=[]):
                template=NeuroFDAN(ecg_channels=2,frontdoor_config=replace(cfg,cns_input_norm='none'))
                expected=parameter_hash(template)
            self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
            actual=NeuroFDAN(ecg_channels=2,frontdoor_config=cfg)
            self.assertEqual(expected,parameter_hash(actual))
