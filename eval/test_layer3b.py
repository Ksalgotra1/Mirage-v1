import unittest
import torch
from pathlib import Path
import tempfile
from detector.layer3b_weights import scan

class TestLayer3b(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_clean_model_has_zero_score_and_false_verdict(self):
        # Create a synthetic clean model with standard gaussian weights
        sd = {
            'conv1.weight': torch.randn(64, 3, 7, 7) * 0.05,
            'layer1.0.conv1.weight': torch.randn(64, 64, 3, 3) * 0.05,
            'layer1.0.conv2.weight': torch.randn(64, 64, 3, 3) * 0.05,
            'fc.weight': torch.randn(10, 64) * 0.1,
            'fc.bias': torch.zeros(10),
        }
        pt_path = Path(self.temp_dir.name) / 'clean.pt'
        torch.save(sd, pt_path)
        res = scan(pt_path)
        self.assertFalse(res['verdict'])
        self.assertEqual(res['score'], 0.0)
        self.assertEqual(len(res['flagged_layers']), 0)

    def test_inflated_neuron_trigger(self):
        sd = {
            'conv1.weight': torch.randn(64, 3, 7, 7) * 0.05,
            'layer1.0.conv1.weight': torch.randn(64, 64, 3, 3) * 0.05,
        }
        # Inflate one neuron in layer1.0.conv1 by 50x
        sd['layer1.0.conv1.weight'][0] *= 50.0
        pt_path = Path(self.temp_dir.name) / 'trojan_neuron.pt'
        torch.save(sd, pt_path)
        res = scan(pt_path)
        self.assertTrue(res['verdict'])
        self.assertGreaterEqual(res['score'], 0.35)
        self.assertIn('layer1.0.conv1.weight', res['flagged_layers'])
        self.assertIn('layer1.0.conv1.weight', res['signals']['inflated_neuron'])

    def test_rank_one_spectral_trigger(self):
        w = torch.randn(64, 64, 3, 3) * 0.02
        u = torch.randn(64, 1, 1, 1)
        v = torch.randn(1, 64, 3, 3)
        w_trojan = w + 15.0 * (u * v)
        sd = {
            'conv1.weight': torch.randn(64, 3, 7, 7) * 0.05,
            'layer1.0.conv1.weight': w_trojan,
        }
        pt_path = Path(self.temp_dir.name) / 'trojan_spectral.pt'
        torch.save(sd, pt_path)
        res = scan(pt_path)
        self.assertTrue(res['verdict'])
        self.assertIn('layer1.0.conv1.weight', res['flagged_layers'])
        self.assertIn('layer1.0.conv1.weight', res['signals']['spectral_collapse'])

    def test_pruned_se_bottleneck_and_pointwise_shortcut_not_flagged(self):
        torch.manual_seed(0)
        # 1. Simulate MobileNetV3 SE fc1 (24, 72, 1, 1) with 18 dead channels (~1e-8) and 6 active (~0.75)
        fc1 = torch.randn(24, 72, 1, 1) * 1e-8
        fc1[:6] = torch.randn(6, 72, 1, 1) * 0.09
        # 2. Simulate MobileNetV3 SE fc2 (72, 24, 1, 1) with 18 dead input columns
        fc2 = torch.randn(72, 24, 1, 1) * 1e-8
        fc2[:, :6] = torch.randn(72, 6, 1, 1) * 0.09
        # 3. Simulate ResNeXt 1x1 pointwise downsample shortcut with sparse high-kurtosis weights
        downsample = torch.randn(128, 64, 1, 1) * 0.01
        spikes = (torch.rand(128, 64, 1, 1) < 0.01).float() * (torch.randn(128, 64, 1, 1) * 0.3)
        downsample = downsample + spikes

        sd = {
            'features.4.block.2.fc1.weight': fc1,
            'features.4.block.2.fc2.weight': fc2,
            'layer2.0.downsample.0.weight': downsample,
            'layer2.0.conv2.weight': torch.randn(64, 64, 3, 3) * 0.05,
        }
        pt_path = Path(self.temp_dir.name) / 'clean_architectural_edge_cases.pt'
        torch.save(sd, pt_path)
        res = scan(pt_path)
        self.assertFalse(res['verdict'])
        self.assertEqual(res['score'], 0.0)
        self.assertEqual(res['flagged_layers'], [])


if __name__ == '__main__':
    unittest.main()

