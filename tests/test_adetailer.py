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
