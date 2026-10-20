"""Decoder schedule equivalence without extracting device scalars onto the host."""

import unittest
from unittest.mock import patch

import torch

from model_autoencoder import CADRE, CADREConfig, Decoder


def legacy_decode(model, z, decode_steps, noise):
    b, t_lat = z.shape[:2]
    x = noise
    steps = torch.linspace(0.0, 1.0, decode_steps + 1, device=z.device)
    for i in range(decode_steps):
        r, t = steps[i], steps[i + 1]
        x_hat = model.decoder.flow_forward(
            z,
            x,
            torch.full((b, t_lat), float(r), device=z.device),
            torch.full((b, t_lat), float(t - r), device=z.device),
        )
        x = x + ((t - r) / (1 - r)) * (x_hat - x)
    return x


class DecoderScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.nnpack_context = torch.backends.nnpack.flags(enabled=False)
        cls.nnpack_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.nnpack_context.__exit__(None, None, None)
        torch.set_num_threads(cls.previous_threads)

    def model(self, device):
        torch.manual_seed(11)
        # Construct the actual decoder without the unrelated pretrained DINO encoder.
        model = CADRE.__new__(CADRE)
        torch.nn.Module.__init__(model)
        model.config = CADREConfig(
            tau=2,
            stride=1,
            latent_dim=4,
            resolution=(32, 32),
            num_frames=4,
            dim=32,
            depth=2,
            tail_depth=1,
            num_heads=4,
            refine_dim=8,
            refine_dilations=(1,),
        )
        model.decoder = Decoder(model.config)
        with torch.no_grad():
            model.decoder.patchify.weight.normal_(std=0.02)
            model.decoder.adaln_head[1].weight.normal_(std=0.02)
        return model.eval().to(device)

    def check_equivalence(self, device):
        model = self.model(device)
        z = torch.randn(2, 2, 4, 2, 2, device=device)
        noise = torch.randn(2, 4, 3, 32, 32, device=device)
        with torch.no_grad():
            for steps in (1, 2, 3, 7):
                with self.subTest(device=device, steps=steps):
                    expected = legacy_decode(model, z, steps, noise)
                    actual = model.decode(z, steps, noise=noise)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_decode_matches_original_float_schedule(self):
        self.check_equivalence("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_decode_matches_original_float_schedule(self):
        self.check_equivalence("cuda")

    def test_decode_does_not_convert_tensors_to_python_scalars(self):
        model = self.model("cpu")
        z = torch.randn(2, 2, 4, 2, 2)
        with torch.no_grad(), patch.object(torch.Tensor, "__float__", side_effect=AssertionError("host scalar read")):
            output = model.decode(z, 3, generator=torch.Generator().manual_seed(3))
        self.assertEqual(output.shape, (2, 4, 3, 32, 32))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
