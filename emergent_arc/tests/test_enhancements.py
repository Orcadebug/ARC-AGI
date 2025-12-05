import unittest
import numpy as np
from emergent_arc.dsl.primitives import Inside, Outside, AdjacentTo, AlignedWith
from emergent_arc.detection.features import extract_global_features
from emergent_arc.inference.decoder import ProgramDecoder
from emergent_arc.memory.subroutines import SubroutineLibrary
from emergent_arc.inference.inducer import OnlineProgramInducer
from emergent_arc.evolution.island import Island
from emergent_arc.memory.veterans import VeteranPool

class TestEnhancements(unittest.TestCase):
    def test_primitives(self):
        p1 = Inside(obj_a=0, obj_b=1)
        p2 = Outside(obj_a=0, obj_b=1)
        p3 = AdjacentTo(obj_a=0, obj_b=1)
        p4 = AlignedWith(obj_a=0, obj_b=1, axis=0)
        self.assertIsInstance(p1, Inside)
        self.assertIsInstance(p2, Outside)
        self.assertIsInstance(p3, AdjacentTo)
        self.assertIsInstance(p4, AlignedWith)

    def test_features(self):
        grid = np.zeros((10, 10), dtype=int)
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1
        objects = [{'mask': np.ones((2,2)), 'color': 1}]
        
        features = extract_global_features(grid, objects)
        # Expected 12 features now
        self.assertEqual(len(features), 12)
        # Check border feature (index 8)
        self.assertEqual(features[8], 1.0)

    def test_decoder(self):
        from emergent_arc.dsl.primitives import Rotate
        lib = SubroutineLibrary()
        decoder = ProgramDecoder(lib)
        # Sequence: [Rotate, 0, 90, Halt] -> [10, 0, 90, 99]
        # Rotate takes object_id (int) and degrees (int)
        seq = [10, 0, 90, 99]
        prog = decoder.decode_sequence(seq)
        self.assertIsNotNone(prog)
        self.assertEqual(len(prog.statements), 1)
        action = prog.statements[0].action
        self.assertIsInstance(action, Rotate)
        self.assertEqual(action.object_id, 0)
        self.assertEqual(action.degrees, 90)

    def test_executor(self):
        from emergent_arc.dsl.executor import execute_dsl
        from emergent_arc.dsl.grammar import Program, Statement
        from emergent_arc.dsl.primitives import Rotate, Translate
        
        # Test Rotate
        grid = np.zeros((5, 5), dtype=int)
        grid[1:4, 2] = 1 # Vertical line at col 2
        # Mask: (1,2), (2,2), (3,2)
        # Rotate 90 deg -> Horizontal line
        
        prog = Program()
        # Object 0 is the line
        prog.add_statement(Statement(Rotate(object_id=0, degrees=90)))
        
        out = execute_dsl(prog, grid)
        # Expected: Horizontal line. 
        # Centroid of vertical line is (2, 2).
        # Rotated horizontal line should be at row 2, cols 1-3?
        # Let's check if output changed
        self.assertFalse(np.array_equal(out, grid))
        self.assertEqual(np.sum(out), 3) # Area preserved

    def test_inducer(self):
        import jax
        import jax.flatten_util
        from emergent_arc.network.snn import SpikingPolicyNetwork
        
        # Get real param size
        snn = SpikingPolicyNetwork()
        key = jax.random.PRNGKey(0)
        params = snn.init_params(key)
        flat, _ = jax.flatten_util.ravel_pytree(params)
        param_size = len(flat)
        
        def mock_evolver():
            return Island(0, 10, param_size)
            
        lib = SubroutineLibrary()
        vets = VeteranPool()
        inducer = OnlineProgramInducer(mock_evolver, lib, vets)
        
        train_pairs = [(np.zeros((5,5), dtype=int), np.zeros((5,5), dtype=int))]
        test_input = np.zeros((5,5), dtype=int)
        
        # This will now trigger full feature extraction
        candidates = inducer.solve_task(train_pairs, test_input, timeout=1.0)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].shape, (5,5))

if __name__ == '__main__':
    unittest.main()
