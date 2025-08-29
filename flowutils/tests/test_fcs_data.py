"""
Tests for FlowUtils transforms with simulated FCS-like data

This test module validates the transforms work correctly with realistic
flow cytometry data that includes multiple populations, negative values,
and the full dynamic range typically seen in FCS files.
"""
import unittest
import numpy as np
from flowutils import transforms


class FCSDataTestCase(unittest.TestCase):
    
    def setUp(self):
        """Create realistic FCS-like data for testing"""
        np.random.seed(42)  # For reproducible tests
        
        # Create multi-population flow cytometry data similar to real FCS files
        n_events = 1000
        
        # Population 1: Negative/autofluorescence (FL1-, FL2-)
        n1 = 300
        fl1_neg = np.random.normal(-50, 30, n1)
        fl2_neg = np.random.normal(-20, 25, n1)
        
        # Population 2: FL1+ FL2- (single positive)
        n2 = 200
        fl1_pos_single = np.random.lognormal(np.log(500), 0.6, n2)
        fl2_pos_single = np.random.normal(10, 40, n2)
        
        # Population 3: FL1- FL2+ (single positive)
        n3 = 200
        fl1_neg_single = np.random.normal(20, 50, n3)
        fl2_pos_single = np.random.lognormal(np.log(800), 0.5, n3)
        
        # Population 4: FL1+ FL2+ (double positive)
        n4 = 200
        fl1_double_pos = np.random.lognormal(np.log(1200), 0.4, n4)
        fl2_double_pos = np.random.lognormal(np.log(1500), 0.4, n4)
        
        # Population 5: Bright population
        n5 = 100
        fl1_bright = np.random.lognormal(np.log(5000), 0.3, n5)
        fl2_bright = np.random.lognormal(np.log(8000), 0.3, n5)
        
        # Combine populations
        fl1_data = np.concatenate([fl1_neg, fl1_pos_single, fl1_neg_single, 
                                   fl1_double_pos, fl1_bright])
        fl2_data = np.concatenate([fl2_neg, fl2_pos_single, fl2_pos_single, 
                                   fl2_double_pos, fl2_bright])
        
        self.fcs_data = np.column_stack([fl1_data, fl2_data])
        
        # Standard flow cytometry transform parameters
        self.T = 262144
        self.M = 4.5
        self.W = 0.5
        self.A = 0
        
    def test_fcs_data_properties(self):
        """Test that our simulated FCS data has realistic properties"""
        self.assertEqual(self.fcs_data.shape[1], 2)  # FL1, FL2
        self.assertGreater(self.fcs_data.shape[0], 0)  # Has events
        
        # Should have negative values (typical in flow cytometry)
        self.assertTrue(np.any(self.fcs_data[:, 0] < 0))  # FL1 negatives
        self.assertTrue(np.any(self.fcs_data[:, 1] < 0))  # FL2 negatives
        
        # Should have positive values
        self.assertTrue(np.any(self.fcs_data[:, 0] > 0))  # FL1 positives  
        self.assertTrue(np.any(self.fcs_data[:, 1] > 0))  # FL2 positives
        
        # Should span several decades (typical flow cytometry range)
        fl1_range = self.fcs_data[:, 0].max() - self.fcs_data[:, 0].min()
        fl2_range = self.fcs_data[:, 1].max() - self.fcs_data[:, 1].min()
        self.assertGreater(fl1_range, 1000)  # Wide dynamic range
        self.assertGreater(fl2_range, 1000)
        
    def test_logicle_with_fcs_data(self):
        """Test logicle transform with realistic FCS data"""
        # Apply logicle transform to both channels
        result = transforms.logicle(
            self.fcs_data, 
            channel_indices=[0, 1], 
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # Check output properties
        self.assertEqual(result.shape, self.fcs_data.shape)
        self.assertTrue(np.all(np.isfinite(result)))  # No NaN/Inf values
        
        # Transform should produce reasonable output range
        # Note: Logicle can produce slightly negative values for very negative inputs
        self.assertGreater(result.min(), -0.1)  # Allow small negative values
        self.assertLessEqual(result.max(), 1.1)  # Allow slightly above 1
        
        # Test round-trip accuracy
        inverse_result = transforms.logicle_inverse(
            result,
            channel_indices=[0, 1],
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # Should recover original data within numerical precision
        max_error = np.max(np.abs(self.fcs_data - inverse_result))
        self.assertLess(max_error, 1e-10, "Round-trip error too large")
        
    def test_hyperlog_with_fcs_data(self):
        """Test hyperlog transform with realistic FCS data"""
        # Apply hyperlog transform to both channels
        result = transforms.hyperlog(
            self.fcs_data,
            channel_indices=[0, 1],
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # Check output properties
        self.assertEqual(result.shape, self.fcs_data.shape)
        self.assertTrue(np.all(np.isfinite(result)))  # No NaN/Inf values
        
        # Transform should produce reasonable output range
        # Note: Hyperlog can produce slightly negative values for very negative inputs
        self.assertGreater(result.min(), -0.1)  # Allow small negative values
        self.assertLessEqual(result.max(), 1.1)  # Allow slightly above 1
        
        # Test round-trip accuracy
        inverse_result = transforms.hyperlog_inverse(
            result,
            channel_indices=[0, 1],
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # Should recover original data within numerical precision
        max_error = np.max(np.abs(self.fcs_data - inverse_result))
        self.assertLess(max_error, 1e-10, "Round-trip error too large")
        
    def test_negative_value_handling(self):
        """Test that transforms handle negative values properly"""
        # Extract negative values from our FCS data
        negative_fl1 = self.fcs_data[self.fcs_data[:, 0] < 0, 0]
        negative_fl2 = self.fcs_data[self.fcs_data[:, 1] < 0, 1]
        
        self.assertGreater(len(negative_fl1), 0, "Need negative FL1 values for test")
        self.assertGreater(len(negative_fl2), 0, "Need negative FL2 values for test") 
        
        # Test logicle with negative values
        logicle_neg_fl1 = transforms._logicle(negative_fl1, t=self.T, m=self.M, w=self.W, a=self.A)
        logicle_neg_fl2 = transforms._logicle(negative_fl2, t=self.T, m=self.M, w=self.W, a=self.A)
        
        # Should produce finite results
        self.assertTrue(np.all(np.isfinite(logicle_neg_fl1)))
        self.assertTrue(np.all(np.isfinite(logicle_neg_fl2)))
        
        # Test hyperlog with negative values  
        hyperlog_neg_fl1 = transforms._hyperlog(negative_fl1, t=self.T, m=self.M, w=self.W, a=self.A)
        hyperlog_neg_fl2 = transforms._hyperlog(negative_fl2, t=self.T, m=self.M, w=self.W, a=self.A)
        
        # Should produce finite results
        self.assertTrue(np.all(np.isfinite(hyperlog_neg_fl1)))
        self.assertTrue(np.all(np.isfinite(hyperlog_neg_fl2)))
        
    def test_zero_value_handling(self):
        """Test that transforms handle zero values properly"""
        # Test with exact zeros
        zeros = np.array([0.0, 0.0, 0.0])
        
        logicle_zeros = transforms._logicle(zeros, t=self.T, m=self.M, w=self.W, a=self.A)
        hyperlog_zeros = transforms._hyperlog(zeros, t=self.T, m=self.M, w=self.W, a=self.A)
        
        # Should produce finite, identical results for all zeros
        self.assertTrue(np.all(np.isfinite(logicle_zeros)))
        self.assertTrue(np.all(np.isfinite(hyperlog_zeros)))
        self.assertTrue(np.allclose(logicle_zeros, logicle_zeros[0]))
        self.assertTrue(np.allclose(hyperlog_zeros, hyperlog_zeros[0]))
        
    def test_wide_dynamic_range(self):
        """Test transforms with the full dynamic range of flow cytometry"""
        # Test data spanning from very negative to very positive (6+ decades)
        test_range = np.array([-1000, -100, -10, -1, 0, 1, 10, 100, 1000, 10000, 100000])
        
        # Apply transforms
        logicle_range = transforms._logicle(test_range, t=self.T, m=self.M, w=self.W, a=self.A)
        hyperlog_range = transforms._hyperlog(test_range, t=self.T, m=self.M, w=self.W, a=self.A)
        
        # Should handle full range without issues
        self.assertTrue(np.all(np.isfinite(logicle_range)))
        self.assertTrue(np.all(np.isfinite(hyperlog_range)))
        
        # Should be monotonically increasing
        self.assertTrue(np.all(np.diff(logicle_range) >= 0))
        self.assertTrue(np.all(np.diff(hyperlog_range) >= 0))
        
    def test_selective_channel_transform(self):
        """Test that we can transform only specific channels"""
        original_data = self.fcs_data.copy()
        
        # Transform only FL1 (channel 0)
        result_fl1_only = transforms.logicle(
            self.fcs_data.copy(),
            channel_indices=[0],
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # FL1 should be transformed, FL2 should be unchanged
        self.assertFalse(np.array_equal(result_fl1_only[:, 0], original_data[:, 0]))
        self.assertTrue(np.array_equal(result_fl1_only[:, 1], original_data[:, 1]))
        
        # Transform only FL2 (channel 1)
        result_fl2_only = transforms.hyperlog(
            self.fcs_data.copy(),
            channel_indices=[1], 
            t=self.T, m=self.M, w=self.W, a=self.A
        )
        
        # FL2 should be transformed, FL1 should be unchanged
        self.assertTrue(np.array_equal(result_fl2_only[:, 0], original_data[:, 0]))
        self.assertFalse(np.array_equal(result_fl2_only[:, 1], original_data[:, 1]))
        
    def test_parameter_sensitivity(self):
        """Test that different parameters produce different but valid results"""
        test_data = self.fcs_data[:100, :]  # Smaller subset for speed
        
        # Test different T values
        result_t1 = transforms.logicle(test_data, [0, 1], t=10000, m=4.5, w=0.5, a=0)
        result_t2 = transforms.logicle(test_data, [0, 1], t=100000, m=4.5, w=0.5, a=0)
        
        self.assertFalse(np.allclose(result_t1, result_t2))
        self.assertTrue(np.all(np.isfinite(result_t1)))
        self.assertTrue(np.all(np.isfinite(result_t2)))
        
        # Test different W values
        result_w1 = transforms.hyperlog(test_data, [0, 1], t=262144, m=4.5, w=0.2, a=0)
        result_w2 = transforms.hyperlog(test_data, [0, 1], t=262144, m=4.5, w=0.8, a=0)
        
        self.assertFalse(np.allclose(result_w1, result_w2))
        self.assertTrue(np.all(np.isfinite(result_w1)))
        self.assertTrue(np.all(np.isfinite(result_w2)))


if __name__ == '__main__':
    unittest.main()