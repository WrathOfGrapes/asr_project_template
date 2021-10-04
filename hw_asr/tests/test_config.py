import json
import unittest

from hw_asr.utils.parse_config import ConfigParser


class TestConfig(unittest.TestCase):
    def test_create(self):
        configs = ConfigParser.get_default_configs()
        print(json.dumps(configs.config, indent=2))
        pass
