"""第四阶段轻量接口测试；随机窗口与产物夹具不是服务器真实数据实验。"""
import copy
import json
import random
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from frontdoor_scm import FrontDoorConfig, ReferenceBank, frontdoor_loss
from ModelTrain import (BestCheckpoint, confusion_metrics, evaluate, evaluate_training_set,
                        make_train_evaluation_loader, seed_everything, train_one_epoch, run_fold)
from NeuroFDAN import NeuroFDAN
from training_diagnostics import (GradientDiagnostics, MediatorDiagnostics, Moments,
                                  PredictionDiagnostics, save_epoch_diagnostics, log_evaluation)


class TinyWindows(Dataset):
    """只含5个随机窗口，验证尾batch与来源，不模拟正式数据实验。"""
    def __init__(self):
        self.xc, self.xa = torch.randn(5, 32, 768), torch.randn(5, 4, 768)
        self.y = torch.tensor([0, 0, 1, 1, 1])
        self.ids = torch.tensor([[1, i, 0] for i in range(5)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return dict(x_c=self.xc[i], x_a=self.xa[i], Labels=self.y[i], source_ids=self.ids[i]), i


def assert_nested_equal(case, first, second):
    if torch.is_tensor(first):
        case.assertTrue(torch.equal(first, second))
    elif isinstance(first, dict):
        case.assertEqual(first.keys(), second.keys())
        for key in first:
            assert_nested_equal(case, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        case.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_nested_equal(case, a, b)
    else:
        case.assertEqual(first, second)


class MetricTests(unittest.TestCase):
    def test_matrix_order_and_both_positive_classes(self):
        # 真值0的3条预测0、1条预测1；真值1的2条预测0、4条预测1。
        d = PredictionDiagnostics()
        y = torch.tensor([0]*4 + [1]*6)
        pred = torch.tensor([0,0,0,1,0,0,1,1,1,1])
        p = torch.nn.functional.one_hot(pred, 2).float()
        d.update(p, p, y)
        result = d.finish()['primary']
        self.assertEqual(result['confusion_matrix'], [[3,1],[2,4]])
        self.assertEqual(result['true_counts'], [4,6])
        self.assertEqual(result['predicted_counts'], [5,5])
        deap, hci = [confusion_metrics(result['confusion_matrix'], name) for name in ('DEAP', 'HCI')]
        self.assertEqual(deap['acc'], .7)
        self.assertEqual(deap['precision'], 4/5)
        self.assertEqual(deap['recall'], 4/6)
        self.assertEqual(deap['specificity'], 3/4)
        self.assertEqual(deap['selection_f1'], 8/11)
        self.assertEqual(hci['precision'], 3/5)
        self.assertEqual(hci['recall'], 3/4)
        self.assertEqual(hci['specificity'], 4/6)
        self.assertEqual(hci['selection_f1'], 6/9)

    def test_single_class_zero_and_one_and_reported_candidates(self):
        for zero, one in [(210,190), (160,240)]:
            for label in (0,1):
                y = torch.tensor([0]*zero + [1]*one)
                p = torch.nn.functional.one_hot(torch.full_like(y, label), 2).float()
                d = PredictionDiagnostics(); d.update(p, p, y)
                r = d.finish()['primary']
                self.assertTrue(r['single_class'])
                self.assertTrue(r['extreme_imbalance'])
                cm = r['confusion_matrix']
                m = confusion_metrics(cm, 'DEAP')
                self.assertAlmostEqual(m['acc'], (one if label else zero) / 400)
                self.assertAlmostEqual(m['selection_f1'], 2*one/(400+one) if label else 0.)

    def test_extreme_imbalance_and_factual_disagreement(self):
        y = torch.zeros(200, dtype=torch.long)
        p = torch.tensor([[.9,.1]]).repeat(200,1); p[-1] = torch.tensor([.1,.9])
        d = PredictionDiagnostics(); d.update(p, p.flip(-1), y)
        r = d.finish()
        self.assertFalse(r['primary']['single_class'])
        self.assertTrue(r['primary']['extreme_imbalance'])
        self.assertEqual(r['factual_primary_disagreement'], 1.)
        json.dumps(r, allow_nan=False)

    def test_checkpoint_rule_unchanged_by_train_and_diagnostics(self):
        tracker = BestCheckpoint()
        for epoch, (acc,f1,wanted) in enumerate([(.5,.9,False),(.6,.5,True),(.6,.5,False),(.6,.6,True),(.59,1.,False)],1):
            self.assertEqual(tracker.consider(dict(acc=acc,selection_f1=f1,train_acc=1.),epoch), wanted)
        self.assertEqual(tracker.epoch, 4)


class MediatorTests(unittest.TestCase):
    def make_stats(self, logvar, mean_variation=False):
        cfg = FrontDoorConfig(mediator_dim=3, hidden_dim=4)
        d = MediatorDiagnostics(cfg)
        mu = torch.zeros(4,3)
        if mean_variation:
            mu[:,0] = torch.arange(4.)
        mc = mu[:,None,:] + torch.tensor([-1.,1.])[None,:,None]
        output = dict(mu_c=mu,logvar_c=torch.full_like(mu,logvar),m_c=mc,
                      mu_a=mu[:,None,:].expand(4,2,3),
                      logvar_a=torch.full((4,2,3),logvar),m_a=mc)
        d.update(output,torch.tensor([0,0,1,1]))
        return d.finish()

    def test_mean_variation_separate_from_posterior_and_sampling(self):
        r = self.make_stats(0.)['cns']
        self.assertTrue(r['low_mean_variation'])
        self.assertEqual(r['posterior_std']['mean'], 1.)
        self.assertEqual(r['posterior_variance']['median'], 1.)
        self.assertEqual(r['sampled_within_sample_std']['mean'], 1.)
        small = self.make_stats(-6.)['ans']
        self.assertLess(small['posterior_std']['mean'], .05)
        self.assertTrue(small['low_mean_variation'])
        self.assertGreater(small['posterior_std']['mean'], 0.)
        self.assertFalse(self.make_stats(0.,True)['cns']['low_mean_variation'])

    def test_full_stream_statistics_not_batch_average_and_class_distance(self):
        stats = Moments()
        stats.update(np.zeros((2,3)))
        stats.update(np.ones((2,3))*2)
        np.testing.assert_array_equal(stats.std(),np.ones(3))
        r = self.make_stats(0.,True)['cns']
        self.assertEqual(r['class_mean_distance_l2'], 2.)
        self.assertGreater(r['class_fisher_ratio'],0.)
        json.dumps(r,allow_nan=False)

    def test_single_sample_and_single_draw_are_finite_not_evidence_of_collapse(self):
        d = MediatorDiagnostics(FrontDoorConfig())
        d.update(dict(mu_c=torch.zeros(1,64),logvar_c=torch.zeros(1,64),m_c=torch.ones(1,1,64),
                      mu_a=torch.zeros(1,1,64),logvar_a=torch.zeros(1,1,64),m_a=torch.ones(1,1,64)),torch.tensor([1]))
        r = d.finish()
        self.assertFalse(r['ans']['within_sample_mc_available'])
        self.assertIsNone(r['cns']['class_mean_distance_l2'])
        self.assertIsNone(r['ans']['class_fisher_ratio'])
        json.dumps(r,allow_nan=False)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); seed_everything()
        self.data = TinyWindows()
        self.loader = make_train_evaluation_loader(self.data,2)
        self.model = NeuroFDAN(frontdoor_config=FrontDoorConfig(hidden_dim=8,mediator_dim=4,ref_dim=8,
                         S_train=2,K_train=2,S_eval=2,K_eval=3,dropout=.3))
        self.bank = ReferenceBank(self.data.xc,self.data.xa,self.data.ids,seed=42,dataset='HCI')
        self.bank.raw_windows,self.bank.population_count = True,5
        self.bank.report = {'held_out_subject':3}
        self.device = torch.device('cpu')

    def test_train_eval_parameters_bn_optimizer_modes_rng_unchanged(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=.0001)
        output = self.model(self.data.xc[:2],self.data.xa[:2],reference_bank=self.bank,source_ids=self.data.ids[:2])
        frontdoor_loss(output,self.data.y[:2],self.model.frontdoor_config)['loss'].backward()
        optimizer.step(); self.model.weights_updated()
        self.model.train()
        params = copy.deepcopy(self.model.state_dict()); opt = copy.deepcopy(optimizer.state_dict())
        grads = [p.grad.clone() if p.grad is not None else None for p in self.model.parameters()]
        modes = [m.training for m in self.model.modules()]
        rng = torch.get_rng_state().clone(); loader_rng = self.loader.generator.get_state().clone()
        python_rng,numpy_rng = random.getstate(),np.random.get_state()
        metrics,cm,pred = evaluate_training_set(self.model,self.loader,self.bank,self.device,'HCI',diagnostic_level='full')
        self.assertEqual(metrics['count'],5); self.assertEqual(int(cm.sum()),5)
        self.assertTrue(metrics['complete_evaluation']); self.assertEqual(pred,[])
        assert_nested_equal(self,params,self.model.state_dict()); assert_nested_equal(self,opt,optimizer.state_dict())
        for a,p in zip(grads,self.model.parameters()):
            if a is None: self.assertIsNone(p.grad)
            else: self.assertTrue(torch.equal(a,p.grad))
        self.assertEqual(modes,[m.training for m in self.model.modules()])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(torch.equal(loader_rng,self.loader.generator.get_state()))
        self.assertEqual(python_rng,random.getstate())
        np.testing.assert_array_equal(numpy_rng[1],np.random.get_state()[1])
        json.dumps(metrics,allow_nan=False)

    def test_repeated_cpu_eval_and_extra_eval_do_not_change_test_prediction(self):
        first = evaluate(self.model,self.loader,self.bank,self.device,'HCI')
        evaluate_training_set(self.model,self.loader,self.bank,self.device,'HCI',diagnostic_level='full')
        second = evaluate(self.model,self.loader,self.bank,self.device,'HCI')
        self.assertEqual(first[2],second[2])
        self.assertTrue(torch.equal(first[1],second[1]))

    def test_full_gradient_diagnostics_before_step_and_no_structure_change(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=.0001)
        loader = make_train_evaluation_loader(torch.utils.data.Subset(self.data,range(4)),2)
        metrics,_ = train_one_epoch(self.model,loader,self.bank,optimizer,self.device,'HCI',1,diagnostic_level='full')
        gradients = metrics['gradients']
        for key in ('cns_mediator','ans_mediator','cns2ans','outcome'):
            self.assertGreater(gradients[key]['pre_clip_l2']['max'],0.)
        self.assertGreater(gradients['cns_mediator']['parameter_tensors'],0)
        # surrogate对阈值不返回显式梯度，但复位项仍可连接到后续时间步。
        self.assertIsNotNone(self.model.frontdoor.cns.snn.cell.thresh.grad)
        bad = GradientDiagnostics(self.model)
        next(self.model.parameters()).grad = torch.full_like(next(self.model.parameters()),float('nan'))
        with self.assertLogs('training_diagnostics',level='ERROR'), self.assertRaises(FloatingPointError):
            bad.update()

    def test_basic_and_full_have_same_primary_outputs_and_loss_update(self):
        state = torch.get_rng_state().clone()
        second = copy.deepcopy(self.model)
        torch.set_rng_state(state)
        a,_ = train_one_epoch(self.model,self.loader,self.bank,torch.optim.Adam(self.model.parameters()),
                              self.device,'HCI',1,max_batches=1,diagnostic_level='basic')
        torch.set_rng_state(state)
        b,_ = train_one_epoch(second,self.loader,self.bank,torch.optim.Adam(second.parameters()),
                              self.device,'HCI',1,max_batches=1,diagnostic_level='full')
        self.assertEqual(a['loss'],b['loss'])
        assert_nested_equal(self,self.model.state_dict(),second.state_dict())

    def test_full_fold_files_and_legacy_summary_history_compatibility(self):
        from Datasets import PhysiologicalWindows
        from mainNeuroFDAN import get_args
        # 注入5窗口的小型接口夹具；不创建或读取正式数据文件。
        def population(subject):
            value = object.__new__(PhysiologicalWindows)
            value.x_c,value.x_a,value.Label = self.data.xc,self.data.xa,self.data.y
            value.source_ids = self.data.ids.clone(); value.source_ids[:,0] = subject
            value.dataset,value.label,value.subjects = 'HCI','valence',[subject]
            value.windows_per_trial,value.trials_per_subject = 1,5
            return value
        with tempfile.TemporaryDirectory() as tmp:
            args = get_args(argv=['--dataset','HCI','--label','valence','--device','cpu',
                '--output_dir',tmp,'--run_name','五窗口接口','--held_out_subjects','3','--epochs','1',
                '--batch_size','2','--no_balance','--hidden_dim','8','--mediator_dim','4','--ref_dim','8',
                '--N_ref','3','--S_train','2','--K_train','2','--S_eval','2','--K_eval','3',
                '--max_train_batches','1','--max_test_batches','1','--diagnostic_level','full'])
            with self.assertLogs(level='INFO') as captured:
                run_fold(args,3,population=population(1),test_dataset=population(3))
            folder = Path(tmp)/'HCI/valence/seed42/五窗口接口/fold03'
            summary = json.loads((folder/'summary.json').read_text())
            history = json.loads((folder/'history.jsonl').read_text().splitlines()[-1])
            self.assertEqual(summary['last_test'],history['test'])
            epoch = json.loads((folder/'epoch_metrics.jsonl').read_text())
            self.assertFalse(epoch['complete']); self.assertTrue(epoch['smoke'])
            mediators = json.loads((folder/'mediator_diagnostics.jsonl').read_text())
            self.assertEqual(mediators['train']['cns']['count'],2)
            self.assertEqual(mediators['test']['ans']['count'],2)
            self.assertIn('[训练集]',' '.join(captured.output))
            self.assertIn('[测试集]',' '.join(captured.output))
            self.assertTrue((folder/'epoch_confusion_matrices.npz').exists())


class PersistenceTests(unittest.TestCase):
    def test_server_guide_cli_and_bash_syntax(self):
        from mainNeuroFDAN import get_args
        guide = Path('docs/phase4_server_guide.md').read_text()
        for block in re.findall(r'```bash\n(.*?)```',guide,re.S):
            result = subprocess.run(['bash','-n'],input=block,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
        count = 0
        for line in guide.replace('\\\n',' ').splitlines():
            if 'python mainNeuroFDAN.py' in line and '--dataset' in line:
                get_args(argv=shlex.split(line.split('python mainNeuroFDAN.py',1)[1])); count += 1
        self.assertEqual(count,4)

    def test_jsonl_npz_resume_streak_and_no_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            d = PredictionDiagnostics(); p = torch.tensor([[0.,1.],[0.,1.]])
            d.update(p,p,torch.tensor([0,1]))
            metrics = confusion_metrics([[0,1],[0,1]],'DEAP')
            metrics.update(prediction_diagnostics=d.finish(),prediction_path='probs_do',complete_evaluation=True)
            for path in ('primary','factual'):
                metrics['prediction_diagnostics'][path]['metrics'] = confusion_metrics([[0,1],[0,1]],'DEAP')
            record = dict(epoch=1,fold=1,smoke=False,complete=True,prediction_path='probs_do',
                          diagnostic_level='basic',train=metrics,test=metrics,optimization={},
                          train_confusion_matrix=[[0,1],[0,1]],test_confusion_matrix=[[0,1],[0,1]])
            before = copy.deepcopy(record)
            with self.assertLogs('training_diagnostics',level='WARNING') as captured:
                save_epoch_diagnostics(folder,record,{'train':None,'test':None})
                record['epoch']=2
                save_epoch_diagnostics(folder,record,{'train':None,'test':None})
            self.assertIn('single-class prediction detected',' '.join(captured.output))
            self.assertEqual(before['test'],record['test'])
            rows = [json.loads(line) for line in (folder/'collapse_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['test']['single_class_streak'],2)
            with np.load(folder/'epoch_confusion_matrices.npz',allow_pickle=False) as arrays:
                self.assertEqual(arrays['epochs'].tolist(),[1,2])
                self.assertEqual(arrays['train'].shape,(2,2,2))
                self.assertEqual(arrays['test'][0].tolist(),[[0,1],[0,1]])
            with self.assertLogs('training_diagnostics',level='INFO') as captured:
                log_evaluation('测试集',metrics,torch.tensor([[0,1],[0,1]]))
            self.assertIn('[[0 1]\n [0 1]]',' '.join(captured.output))
            # 回退到epoch1不保留旧epoch2诊断；缺失历史不填零。
            record['epoch']=1
            with self.assertLogs('training_diagnostics',level='WARNING'):
                save_epoch_diagnostics(folder,record,{'train':None,'test':None})
            with np.load(folder/'epoch_confusion_matrices.npz') as arrays:
                self.assertEqual(arrays['epochs'].tolist(),[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
