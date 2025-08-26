#!/usr/bin/env python3

"""
Test script to validate Isaac Gym Humanoid Locomotion project setup
"""

import sys
import importlib
import torch

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        'torch',
        'numpy',
        'isaacgym',
        'rsl_rl',
        'legged_gym'
    ]
    
    success = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            success = False
    
    return success

def test_configs():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
        from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
        print("✓ Configuration classes loaded successfully")
        
        # Test config instantiation
        g1_cfg = G1RoughCfg()
        h1_cfg = H1RoughCfg()
        print("✓ Configuration objects created successfully")
        
        # Test terrain proportions
        expected_proportions = [0.25, 0.25, 0.1, 0.1, 0.1, 0.1, 0.1]
        if g1_cfg.terrain.terrain_proportions == expected_proportions:
            print("✓ Terrain proportions correctly configured")
        else:
            print("✗ Terrain proportions mismatch")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False

def test_task_registry():
    """Test task registry functionality"""
    print("\nTesting task registry...")
    
    try:
        from legged_gym.utils.task_registry import task_registry
        from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
        from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
        
        # Register tasks
        task_registry.register("g1", G1RoughCfg, G1RoughCfgPPO)
        task_registry.register("h1", H1RoughCfg, H1RoughCfgPPO)
        print("✓ Tasks registered successfully")
        
        # Test config retrieval
        g1_env_cfg, g1_train_cfg = task_registry.get_cfgs("g1")
        h1_env_cfg, h1_train_cfg = task_registry.get_cfgs("h1")
        print("✓ Task configurations retrieved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Task registry test failed: {e}")
        return False

def test_device():
    """Test CUDA availability"""
    print("\nTesting device availability...")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠ CUDA not available - training will use CPU (slower)")
        return True

def test_joint_mappings():
    """Test joint mapping configurations"""
    print("\nTesting joint mappings...")
    
    try:
        from legged_gym.envs.g1.g1_config import G1RoughCfg
        from legged_gym.envs.h1.h1_config import H1RoughCfg
        
        g1_cfg = G1RoughCfg()
        h1_cfg = H1RoughCfg()
        
        # Check G1 joint count
        g1_joints = len(g1_cfg.init_state.default_joint_angles)
        if g1_joints == 23:
            print(f"✓ G1 has correct DOF count: {g1_joints}")
        else:
            print(f"✗ G1 DOF mismatch: expected 23, got {g1_joints}")
            return False
        
        # Check H1 joint count  
        h1_joints = len(h1_cfg.init_state.default_joint_angles)
        print(f"✓ H1 DOF count: {h1_joints}")
        
        return True
        
    except Exception as e:
        print(f"✗ Joint mapping test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ISAAC GYM HUMANOID LOCOMOTION - SETUP VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_configs,
        test_task_registry,
        test_device,
        test_joint_mappings
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is ready for training.")
        return 0
    else:
        print("✗ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 