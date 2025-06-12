import unittest
import modules.prompt_style_system as pss

class TestPromptStyleSystem(unittest.TestCase):
    def test_merge_prompts_ignores_non_string_inputs(self):
        self.assertEqual(pss.merge_prompts(['a', 'b'], 'foo'), 'foo')
        self.assertEqual(pss.merge_prompts('bar', ['foo']), 'bar')

