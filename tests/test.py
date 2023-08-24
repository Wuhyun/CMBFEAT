import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cmbfeat

def test_import():
    import cmbfeat
    import cmbfeat.models
    assert hasattr(cmbfeat.models, "LinOscPrimordialPk")
    assert hasattr(cmbfeat.models, "LinEnvOscPrimordialPk")
