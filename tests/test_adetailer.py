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
