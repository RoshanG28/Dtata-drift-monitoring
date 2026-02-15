"""
Data Drift Detection System
Author: Cyril Anand
Description: Core drift detection engine with PSI and KS-Test implementation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Comprehensive data drift detection system
    
    Methods:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov Test
    - Distribution comparison
    - Automated alerting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize drift detector with configuration"""
        self.config = config or self._default_config()
        self.drift_results = {}
        self.alerts = []
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'psi_thresholds': {
                'no_change': 0.1,
                'small_change': 0.2,
                'significant_change': 0.25
            },
            'ks_test': {
                'significance_level': 0.05
            },
            'alert_enabled': True
        }
    
    def calculate_psi(self, 
                     reference: np.ndarray, 
                     production: np.ndarray, 
                     bins: int = 10) -> Dict:
        """
        Calculate Population Stability Index
        
        Args:
            reference: Reference/baseline distribution
            production: Production/current distribution
            bins: Number of bins for bucketing
            
        Returns:
            Dictionary with PSI score and interpretation
        """
        # Create bins based on reference data
        if len(np.unique(reference)) <= bins:
            # Categorical or low cardinality - use unique values
            breakpoints = sorted(np.unique(reference))
        else:
            # Continuous data - create quantile bins
            breakpoints = np.percentile(
                reference, 
                np.linspace(0, 100, bins + 1)
            )
            breakpoints = np.unique(breakpoints)
        
        # Ensure we have enough bins
        if len(breakpoints) < 2:
            return {
                'psi': 0.0,
                'interpretation': 'insufficient_data',
                'status': 'warning'
            }
        
        # Count observations in each bin
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        prod_counts = np.histogram(production, bins=breakpoints)[0]
        
        # Calculate percentages (add small epsilon to avoid division by zero)
        epsilon = 1e-10
        ref_percent = (ref_counts + epsilon) / (len(reference) + epsilon * len(ref_counts))
        prod_percent = (prod_counts + epsilon) / (len(production) + epsilon * len(prod_counts))
        
        # Calculate PSI
        psi_values = (prod_percent - ref_percent) * np.log(prod_percent / ref_percent)
        psi = np.sum(psi_values)
        
        # Interpret results
        thresholds = self.config['psi_thresholds']
        if psi < thresholds['no_change']:
            interpretation = 'no_change'
            status = 'ok'
        elif psi < thresholds['small_change']:
            interpretation = 'small_change'
            status = 'warning'
        else:
            interpretation = 'significant_change'
            status = 'alert'
        
        return {
            'psi': float(psi),
            'interpretation': interpretation,
            'status': status,
            'bins': len(breakpoints) - 1,
            'breakpoints': breakpoints.tolist()
        }
    
    def kolmogorov_smirnov_test(self, 
                                reference: np.ndarray, 
                                production: np.ndarray) -> Dict:
        """
        Perform Kolmogorov-Smirnov test
        
        Args:
            reference: Reference distribution
            production: Production distribution
            
        Returns:
            Dictionary with KS statistic and p-value
        """
        # Perform KS test
        statistic, p_value = stats.ks_2samp(reference, production)
        
        # Interpret results
        significance = self.config['ks_test']['significance_level']
        is_significant = p_value < significance
        
        return {
            'ks_statistic': float(statistic),
            'p_value': float(p_value),
            'is_significant': bool(is_significant),
            'status': 'alert' if is_significant else 'ok',
            'interpretation': 'drift_detected' if is_significant else 'no_drift'
        }
    
    def calculate_distribution_stats(self, data: np.ndarray) -> Dict:
        """
        Calculate comprehensive distribution statistics
        
        Args:
            data: Input data array
            
        Returns:
            Dictionary with distribution statistics
        """
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
    
    def detect_drift(self, 
                    reference_df: pd.DataFrame, 
                    production_df: pd.DataFrame,
                    features: Optional[List[str]] = None) -> Dict:
        """
        Comprehensive drift detection across all features
        
        Args:
            reference_df: Reference DataFrame
            production_df: Production DataFrame
            features: List of features to monitor (None = all numeric)
            
        Returns:
            Dictionary with drift detection results
        """
        logger.info("Starting drift detection...")
        
        # Determine features to monitor
        if features is None:
            features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_features': len(features),
            'features': {}
        }
        
        drift_count = 0
        
        for feature in features:
            logger.info(f"Analyzing feature: {feature}")
            
            # Extract feature data
            ref_data = reference_df[feature].dropna().values
            prod_data = production_df[feature].dropna().values
            
            # Skip if insufficient data
            if len(ref_data) < 30 or len(prod_data) < 30:
                logger.warning(f"Insufficient data for feature {feature}")
                continue
            
            # Calculate PSI
            psi_result = self.calculate_psi(ref_data, prod_data)
            
            # Perform KS test
            ks_result = self.kolmogorov_smirnov_test(ref_data, prod_data)
            
            # Calculate distribution statistics
            ref_stats = self.calculate_distribution_stats(ref_data)
            prod_stats = self.calculate_distribution_stats(prod_data)
            
            # Determine overall drift status
            if psi_result['status'] == 'alert' or ks_result['status'] == 'alert':
                overall_status = 'alert'
                drift_count += 1
            elif psi_result['status'] == 'warning' or ks_result['status'] == 'warning':
                overall_status = 'warning'
            else:
                overall_status = 'ok'
            
            # Store results
            results['features'][feature] = {
                'psi': psi_result,
                'ks_test': ks_result,
                'reference_stats': ref_stats,
                'production_stats': prod_stats,
                'status': overall_status
            }
            
            # Generate alert if needed
            if overall_status == 'alert' and self.config['alert_enabled']:
                self._generate_alert(feature, psi_result, ks_result)
        
        # Overall system status
        results['overall_status'] = self._determine_overall_status(drift_count, len(features))
        results['drift_count'] = drift_count
        
        # Store results
        self.drift_results = results
        
        logger.info(f"Drift detection complete. Found drift in {drift_count}/{len(features)} features")
        
        return results
    
    def _determine_overall_status(self, drift_count: int, total_features: int) -> str:
        """Determine overall system drift status"""
        drift_percentage = (drift_count / total_features) * 100 if total_features > 0 else 0
        
        if drift_percentage >= 30:
            return 'critical'
        elif drift_percentage >= 10:
            return 'warning'
        else:
            return 'ok'
    
    def _generate_alert(self, feature: str, psi_result: Dict, ks_result: Dict):
        """Generate drift alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'feature': feature,
            'type': 'drift_detected',
            'severity': 'high' if psi_result['psi'] > 0.3 else 'medium',
            'psi_score': psi_result['psi'],
            'ks_p_value': ks_result['p_value'],
            'message': f"Significant drift detected in feature '{feature}'"
        }
        
        self.alerts.append(alert)
        logger.warning(f"DRIFT ALERT: {alert['message']}")
    
    def generate_report(self, output_file: str = 'drift_report.json'):
        """
        Generate comprehensive drift report
        
        Args:
            output_file: Path to save report
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'drift_results': self.drift_results,
            'alerts': self.alerts,
            'summary': {
                'total_features': self.drift_results.get('n_features', 0),
                'drift_detected': self.drift_results.get('drift_count', 0),
                'overall_status': self.drift_results.get('overall_status', 'unknown'),
                'total_alerts': len(self.alerts)
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Drift report saved to {output_file}")
        
        return report
    
    def visualize_drift(self, feature: str, reference: np.ndarray, production: np.ndarray):
        """
        Visualize drift for a specific feature
        
        Args:
            feature: Feature name
            reference: Reference data
            production: Production data
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Drift Analysis: {feature}', fontsize=16, fontweight='bold')
        
        # Distribution comparison
        axes[0, 0].hist(reference, bins=30, alpha=0.5, label='Reference', color='blue')
        axes[0, 0].hist(production, bins=30, alpha=0.5, label='Production', color='red')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # CDF comparison
        ref_sorted = np.sort(reference)
        prod_sorted = np.sort(production)
        ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
        prod_cdf = np.arange(1, len(prod_sorted) + 1) / len(prod_sorted)
        
        axes[0, 1].plot(ref_sorted, ref_cdf, label='Reference', color='blue')
        axes[0, 1].plot(prod_sorted, prod_cdf, label='Production', color='red')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot comparison
        axes[1, 0].boxplot([reference, production], labels=['Reference', 'Production'])
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].set_title('Box Plot Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy.stats import probplot
        probplot(production, dist=stats.norm, sparams=(np.mean(reference), np.std(reference)), 
                plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'drift_visualization_{feature}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved for feature: {feature}")


def main():
    """Example usage"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100000
    
    # Reference data (baseline)
    reference_df = pd.DataFrame({
        'feature_1': np.random.normal(100, 15, n_samples),
        'feature_2': np.random.exponential(50, n_samples),
        'feature_3': np.random.uniform(0, 100, n_samples)
    })
    
    # Production data (with drift in feature_1)
    production_df = pd.DataFrame({
        'feature_1': np.random.normal(110, 20, n_samples),  # Mean shift + variance increase
        'feature_2': np.random.exponential(50, n_samples),   # No drift
        'feature_3': np.random.uniform(0, 100, n_samples)    # No drift
    })
    
    # Initialize detector
    detector = DriftDetector()
    
    # Detect drift
    results = detector.detect_drift(reference_df, production_df)
    
    # Print summary
    print("\n" + "="*60)
    print("DRIFT DETECTION SUMMARY")
    print("="*60)
    print(f"Total Features: {results['n_features']}")
    print(f"Features with Drift: {results['drift_count']}")
    print(f"Overall Status: {results['overall_status']}")
    print("="*60)
    
    # Generate report
    detector.generate_report()
    
    # Visualize drift for feature_1
    detector.visualize_drift(
        'feature_1',
        reference_df['feature_1'].values,
        production_df['feature_1'].values
    )


if __name__ == "__main__":
    main()
