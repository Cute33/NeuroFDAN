"""新fold/epoch只读探针身份与来源守卫测试，待服务器执行。"""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_checkpoint_gradient_terms import validate_identity, validate_sources


class EpochCheckpointGradientTests(unittest.TestCase):
    def setUp(self):
        args = dict(label='arousal', fd_mode='full')
        self.cfg = dict(dataset='DEAP', arguments=args, held_out_subject=2, seed=42,
                        smoke=False, train_subjects=[1, 3], model=dict(lif_neurons=256))
        self.state = dict(format_version=2, epoch=300, arguments=copy.deepcopy(args),
                          frontdoor_config=dict(lif_neurons=256),
                          reference_report=dict(dataset='DEAP', held_out_subject=2))

    def test_epoch300_fold2_accepted_and_wrong_epoch_rejected(self):
        validate_identity(self.cfg, self.state, 'DEAP', 'arousal', 2, 300)
        with self.assertRaises(ValueError):
            validate_identity(self.cfg, self.state, 'DEAP', 'arousal', 2, 20)
        self.cfg['smoke'] = True
        with self.assertRaises(ValueError):
            validate_identity(self.cfg, self.state, 'DEAP', 'arousal', 2, 300)

    def test_config_mismatch_and_heldout_reference_rejected(self):
        self.state['frontdoor_config']['lif_neurons'] = 128
        with self.assertRaises(ValueError):
            validate_identity(self.cfg, self.state, 'DEAP', 'arousal', 2, 300)
        validate_sources([[1, 0, 0]], [[3, 0, 1]], [[3, 0, 1]], [1, 3], 2)
        with self.assertRaises(ValueError):
            validate_sources([[1, 0, 0]], [[2, 0, 1]], [[2, 0, 1]], [1, 3], 2)
        with self.assertRaises(ValueError):
            validate_sources([[3, 0, 1]], [[3, 0, 1]], [[3, 0, 1]], [1, 3], 2)


if __name__ == '__main__':
    unittest.main()
