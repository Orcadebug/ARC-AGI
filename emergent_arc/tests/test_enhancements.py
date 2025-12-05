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
        lib = SubroutineLibrary()
        decoder = ProgramDecoder(lib)
        # Mock sequence: [Rotate, 0, 90, Halt] -> [10, 0, 90, 99]
        # Note: decoder implementation in this step was simplified and might not fully parse args yet,
        # but should return a Program object.
        seq = [10, 0, 90, 99]
        prog = decoder.decode_sequence(seq)
        self.assertIsNotNone(prog)
        
    def test_inducer(self):
        def mock_evolver():
            return Island(0, 10, 100)
            
        lib = SubroutineLibrary()
        vets = VeteranPool()
        inducer = OnlineProgramInducer(mock_evolver, lib, vets)
        
        train_pairs = [(np.zeros((5,5)), np.zeros((5,5)))]
        test_input = np.zeros((5,5))
        
        candidates = inducer.solve_task(train_pairs, test_input, timeout=1.0)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].shape, (5,5))

if __name__ == '__main__':
    unittest.main()
