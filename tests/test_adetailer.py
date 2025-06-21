import os
import sys
import importlib
import tempfile
import unittest

sys.argv = [sys.argv[0]]

class TestADetailerDownload(unittest.TestCase):
    def test_ensure_model_path(self):
        temp_dir = tempfile.mkdtemp()
        os.environ['path_adetailer_detection'] = temp_dir
        import modules.config as config
        importlib.reload(config)
        import modules.adetailer as ad
        path = ad.ensure_model('dummy.pt', url=None)
        self.assertTrue(path.startswith(temp_dir))

    def test_apply_adetailer_multi_noop(self):
        import modules.config as config
        importlib.reload(config)
        config.default_adetailer_enable = True
        for i in range(1, 5):
            setattr(config, f'default_adetailer_tab{i}_enable', False)
        from modules.adetailer.adetailer import apply_adetailer_multi
        from PIL import Image
        img = Image.new('RGB', (10, 10))
        result = apply_adetailer_multi(img)
        self.assertEqual(result.size, (10, 10))
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.mode, 'RGB')

    def test_validate_image_output(self):
        from modules.adetailer.adetailer import _validate_image_output
        from PIL import Image

        img = Image.new('L', (5, 5))
        validated = _validate_image_output(img, expected_size=(4, 4))
        self.assertIsInstance(validated, Image.Image)
        self.assertEqual(validated.mode, 'RGB')
        self.assertEqual(validated.size, (4, 4))

    def test_refine_mask_region(self):
        from modules.adetailer.adetailer import _refine_mask_region
        from PIL import Image, ImageDraw

        img = Image.new('RGB', (20, 20), 'white')
        mask = Image.new('L', (20, 20), 0)
        ImageDraw.Draw(mask).rectangle([5, 5, 15, 15], fill=255)

        _refine_mask_region(img, mask)
        self.assertEqual(img.size, (20, 20))
