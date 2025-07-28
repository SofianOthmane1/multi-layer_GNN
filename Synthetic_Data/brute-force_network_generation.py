"""
This script emulates a financial network and it's functional participants.
This particular iteration creates a network with 500 participants.
Network statistics are similar to those seen within the UK financial system.
"""


import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set
import warnings
warnings.filterwarnings('ignore')

class SyntheticFinancialNetworkGenerator:
    """
    Generates synthetic financial network data for systemic risk modeling
    based on real financial network statistical properties and empirical data.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the synthetic data generator.

        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        self.scaler = StandardScaler()

        # Define institution types and their characteristics based on empirical data
        self.institution_types = {
            'commercial_bank': {
                'size_range': (1e9, 5e12),  # $1B to $5T assets
                'leverage_range': (0.85, 0.95),
                'liquidity_range': (0.05, 0.20),
                'prevalence': 0.35,
                'connectivity': 'high'
            },
            'investment_bank': {
                'size_range': (5e8, 1e12),
                'leverage_range': (0.90, 0.97),
                'liquidity_range': (0.02, 0.10),
                'prevalence': 0.15,
                'connectivity': 'very_high'
            },
            'insurance_company': {
                'size_range': (1e9, 2e12),
                'leverage_range': (0.80, 0.92),
                'liquidity_range': (0.10, 0.30),
                'prevalence': 0.20,
                'connectivity': 'medium'
            },
            'hedge_fund': {
                'size_range': (1e7, 1e11),
                'leverage_range': (0.70, 0.90),
                'liquidity_range': (0.15, 0.40),
                'prevalence': 0.15,
                'connectivity': 'medium'
            },
            'pension_fund': {
                'size_range': (1e9, 5e11),
                'leverage_range': (0.20, 0.50),
                'liquidity_range': (0.20, 0.50),
                'prevalence': 0.10,
                'connectivity': 'low'
            },
            'asset_manager': {
                'size_range': (1e8, 1e12),
                'leverage_range': (0.30, 0.60),
                'liquidity_range': (0.30, 0.60),
                'prevalence': 0.05,
                'connectivity': 'medium'
            }
        }

        # Allowed connections between institution types (regulatory and business constraints)
        self.allowed_connections = {
            'commercial_bank': {'commercial_bank', 'investment_bank', 'insurance_company',
                               'hedge_fund', 'pension_fund', 'asset_manager'},
            'investment_bank': {'commercial_bank', 'investment_bank', 'hedge_fund',
                               'pension_fund', 'asset_manager'},
            'insurance_company': {'commercial_bank', 'investment_bank', 'insurance_company',
                                 'asset_manager'},
            'hedge_fund': {'commercial_bank', 'investment_bank', 'hedge_fund'},
            'pension_fund': {'commercial_bank', 'investment_bank', 'asset_manager'},
            'asset_manager': {'commercial_bank', 'investment_bank', 'insurance_company',
                            'pension_fund'}
        }

        # Edge type probabilities based on institution types
        self.edge_type_probs = {
            ('commercial_bank', 'commercial_bank'): [0.7, 0.2, 0.1],  # lending, derivatives, securities
            ('commercial_bank', 'investment_bank'): [0.3, 0.5, 0.2],
            ('investment_bank', 'hedge_fund'): [0.2, 0.6, 0.2],
            ('pension_fund', 'asset_manager'): [0.1, 0.1, 0.8],
            'default': [0.5, 0.3, 0.2]
        }

    def generate_network_topology(self,
                                 n_nodes: int = 500,
                                 blend_weights: Dict[str, float] = None) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Generate network topology as a blend of different network types
        based on empirical financial network properties.

        Args:
            n_nodes: Number of financial institutions
            blend_weights: Weights for blending different network types

        Returns:
            NetworkX graph and institution type assignments
        """
        if blend_weights is None:
            # Based on empirical studies of financial networks
            blend_weights = {
                'scale_free': 0.5,      # Power-law degree distribution
                'small_world': 0.3,     # High clustering, short paths
                'core_periphery': 0.2   # Core of highly connected banks
            }

        # First, assign institution types
        institution_assignments = self.assign_institution_types(n_nodes)

        # Create empty graph
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))

        # Add node attributes
        for i in range(n_nodes):
            G.nodes[i]['institution_type'] = institution_assignments.loc[i, 'institution_type']

        # Generate edges considering institution types
        edge_set = set()

        # Scale-free component with type constraints
        if blend_weights.get('scale_free', 0) > 0:
            # Group nodes by connectivity level
            high_conn_nodes = [i for i in range(n_nodes)
                             if self.institution_types[G.nodes[i]['institution_type']]['connectivity']
                             in ['high', 'very_high']]

            # Build scale-free network favoring high-connectivity institutions
            for i in range(n_nodes):
                source_type = G.nodes[i]['institution_type']
                n_connections = self._get_n_connections(source_type, i, n_nodes)

                # Find allowed targets
                allowed_targets = [j for j in range(n_nodes) if j != i and
                                 G.nodes[j]['institution_type'] in self.allowed_connections[source_type]]

                if allowed_targets:
                    # Preferential attachment
                    if i in high_conn_nodes:
                        n_connections = int(n_connections * 1.5)

                    targets = np.random.choice(allowed_targets,
                                             min(n_connections, len(allowed_targets)),
                                             replace=False)
                    for j in targets:
                        edge_set.add((min(i, j), max(i, j)))

        # Small-world component
        if blend_weights.get('small_world', 0) > 0:
            # Create clusters by institution type
            for inst_type in self.institution_types:
                type_nodes = [i for i in range(n_nodes)
                            if G.nodes[i]['institution_type'] == inst_type]

                if len(type_nodes) > 3:
                    # Create ring with local connections
                    for idx, i in enumerate(type_nodes):
                        # Connect to next few nodes in the ring
                        for offset in range(1, min(4, len(type_nodes)//2)):
                            j = type_nodes[(idx + offset) % len(type_nodes)]
                            if self._connection_allowed(G, i, j):
                                edge_set.add((min(i, j), max(i, j)))

                    # Add some long-range connections
                    n_long_range = max(1, len(type_nodes) // 10)
                    for _ in range(n_long_range):
                        i, j = np.random.choice(type_nodes, 2, replace=False)
                        if self._connection_allowed(G, i, j):
                            edge_set.add((min(i, j), max(i, j)))

        # Core-periphery component
        if blend_weights.get('core_periphery', 0) > 0:
            # Core consists of large banks and investment banks
            core_nodes = [i for i in range(n_nodes)
                         if G.nodes[i]['institution_type'] in ['commercial_bank', 'investment_bank']]

            # Limit core size
            core_nodes = core_nodes[:max(5, int(0.1 * n_nodes))]

            # Core is densely connected
            for idx, i in enumerate(core_nodes):
                for j in core_nodes[idx+1:]:
                    if np.random.random() < 0.7 and self._connection_allowed(G, i, j):
                        edge_set.add((min(i, j), max(i, j)))

            # Periphery connects to core
            periphery_nodes = [i for i in range(n_nodes) if i not in core_nodes]
            for i in periphery_nodes:
                # Each periphery node connects to 1-3 core nodes
                n_connections = np.random.randint(1, 4)
                allowed_core = [j for j in core_nodes
                              if self._connection_allowed(G, i, j)]
                if allowed_core:
                    targets = np.random.choice(allowed_core,
                                             min(n_connections, len(allowed_core)),
                                             replace=False)
                    for j in targets:
                        edge_set.add((min(i, j), max(i, j)))

        # Add all edges to graph
        G.add_edges_from(edge_set)

        # Convert to directed graph
        G = G.to_directed()

        return G, institution_assignments

    def _connection_allowed(self, G: nx.Graph, i: int, j: int) -> bool:
        """Check if connection is allowed between two nodes based on institution types."""
        type_i = G.nodes[i]['institution_type']
        type_j = G.nodes[j]['institution_type']
        return type_j in self.allowed_connections[type_i]

    def _get_n_connections(self, inst_type: str, node_id: int, n_nodes: int) -> int:
        """Get number of connections for a node based on institution type."""
        connectivity = self.institution_types[inst_type]['connectivity']
        if connectivity == 'very_high':
            base_connections = int(0.05 * n_nodes)
        elif connectivity == 'high':
            base_connections = int(0.03 * n_nodes)
        elif connectivity == 'medium':
            base_connections = int(0.02 * n_nodes)
        else:  # low
            base_connections = int(0.01 * n_nodes)

        # Add some randomness
        return max(1, int(base_connections * np.random.lognormal(0, 0.3)))

    def assign_institution_types(self, n_nodes: int) -> pd.DataFrame:
        """
        Assign institution types to nodes based on empirical prevalence.

        Args:
            n_nodes: Number of nodes

        Returns:
            DataFrame with institution type assignments
        """
        types = []
        prevalences = []

        for inst_type, config in self.institution_types.items():
            types.append(inst_type)
            prevalences.append(config['prevalence'])

        # Normalize prevalences
        prevalences = np.array(prevalences) / sum(prevalences)

        # Assign types
        assigned_types = np.random.choice(types, n_nodes, p=prevalences)

        return pd.DataFrame({
            'institution_type': assigned_types,
            'type_id': pd.Categorical(assigned_types).codes
        })

    def generate_node_features(self,
                             n_nodes: int,
                             institution_types: pd.DataFrame) -> pd.DataFrame:
        """
        Generate realistic node features with correlations.
        """
        features = pd.DataFrame(index=range(n_nodes))

        # Generate correlated features for each institution type
        for inst_type in self.institution_types:
            mask = institution_types['institution_type'] == inst_type
            n_inst = mask.sum()

            if n_inst > 0:
                config = self.institution_types[inst_type]

                # Base features with correlations
                cov_matrix = np.array([
                    [1.0, 0.3, -0.5, 0.2],   # size correlations
                    [0.3, 1.0, -0.3, 0.1],   # leverage
                    [-0.5, -0.3, 1.0, 0.4],  # liquidity
                    [0.2, 0.1, 0.4, 1.0]     # profitability
                ])

                base_features = np.random.multivariate_normal(
                    mean=[0, 0, 0, 0],
                    cov=cov_matrix * 0.5,
                    size=n_inst
                )

                # Transform to realistic ranges
                features.loc[mask, 'total_assets'] = np.exp(
                    np.random.uniform(
                        np.log(config['size_range'][0]),
                        np.log(config['size_range'][1]),
                        n_inst
                    ) + base_features[:, 0] * 0.5
                )

                features.loc[mask, 'leverage_ratio'] = np.clip(
                    np.random.beta(
                        a=10 * config['leverage_range'][0],
                        b=10 * (1 - config['leverage_range'][0]),
                        size=n_inst
                    ) + base_features[:, 1] * 0.05,
                    0.1, 0.99
                )

                features.loc[mask, 'liquidity_ratio'] = np.clip(
                    np.random.beta(
                        a=5 * config['liquidity_range'][0],
                        b=5 * (1 - config['liquidity_range'][0]),
                        size=n_inst
                    ) + base_features[:, 2] * 0.05,
                    0.01, 0.8
                )

                features.loc[mask, 'profitability'] = np.random.normal(
                    0.01, 0.02, n_inst
                ) + base_features[:, 3] * 0.01

        # Add derived features
        features['tier1_capital'] = features['total_assets'] * (1 - features['leverage_ratio']) * \
                                   np.random.uniform(0.5, 1.0, n_nodes)
        features['risk_weighted_assets'] = features['total_assets'] * \
                                          np.random.uniform(0.3, 0.8, n_nodes)
        features['capital_adequacy_ratio'] = features['tier1_capital'] / features['risk_weighted_assets']

        # Add institution type info
        features['institution_type'] = institution_types['institution_type']
        features['institution_type_id'] = institution_types['type_id']

        return features

    def generate_edge_features(self,
                             G: nx.DiGraph,
                             node_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate realistic edge features based on node characteristics.
        """
        edge_data = []

        for i, j in G.edges():
            source_type = G.nodes[i]['institution_type']
            target_type = G.nodes[j]['institution_type']

            # Get edge type probabilities
            type_pair = (source_type, target_type)
            if type_pair in self.edge_type_probs:
                probs = self.edge_type_probs[type_pair]
            else:
                probs = self.edge_type_probs['default']

            # Determine edge type
            edge_type = np.random.choice(['lending', 'derivatives', 'securities'], p=probs)

            # Generate exposure based on institution sizes and types
            source_assets = node_features.loc[i, 'total_assets']
            target_assets = node_features.loc[j, 'total_assets']

            # Base exposure as fraction of smaller institution's assets
            max_exposure_ratio = 0.1  # 10% max exposure
            if edge_type == 'lending':
                base_exposure = min(source_assets, target_assets) * np.random.beta(2, 20) * max_exposure_ratio
            elif edge_type == 'derivatives':
                base_exposure = min(source_assets, target_assets) * np.random.beta(1, 30) * max_exposure_ratio * 0.5
            else:  # securities
                base_exposure = min(source_assets, target_assets) * np.random.beta(3, 15) * max_exposure_ratio * 0.7

            # Add maturity and interest rate
            if edge_type == 'lending':
                maturity = np.random.choice([0.25, 0.5, 1, 2, 5, 10],
                                          p=[0.2, 0.3, 0.25, 0.15, 0.08, 0.02])
                interest_rate = 0.02 + np.random.exponential(0.02)
            else:
                maturity = np.random.choice([0.083, 0.25, 0.5, 1, 2],
                                          p=[0.3, 0.3, 0.2, 0.15, 0.05])
                interest_rate = 0.01 + np.random.exponential(0.015)

            edge_data.append({
                'source': i,
                'target': j,
                'edge_type': edge_type,
                'exposure': base_exposure,
                'maturity_years': maturity,
                'interest_rate': interest_rate,
                'collateral_ratio': np.random.beta(5, 2) if edge_type == 'lending' else 0,
                'mark_to_market': base_exposure * np.random.normal(1, 0.05),
                'counterparty_risk_weight': 1 - node_features.loc[j, 'capital_adequacy_ratio'] * 0.5
            })

        return pd.DataFrame(edge_data)

    def generate_financial_network(self,
                                 n_nodes: int = 500,
                                 output_prefix: str = 'financial_network') -> Dict[str, pd.DataFrame]:
        """
        Generate complete financial network with node and edge features.

        Args:
            n_nodes: Number of financial institutions
            output_prefix: Prefix for output files

        Returns:
            Dictionary containing node and edge DataFrames
        """
        print(f"Generating financial network with {n_nodes} institutions...")

        # Generate network topology
        G, institution_types = self.generate_network_topology(n_nodes)
        print(f"Generated network with {G.number_of_edges()} edges")

        # Generate node features
        node_features = self.generate_node_features(n_nodes, institution_types)
        print("Generated node features")

        # Generate edge features
        edge_features = self.generate_edge_features(G, node_features)
        print("Generated edge features")

        # Save to CSV files
        node_features.to_csv(f'{output_prefix}_nodes.csv', index_label='node_id')
        edge_features.to_csv(f'{output_prefix}_edges.csv', index=False)

        print(f"\nSaved network data to:")
        print(f"  - {output_prefix}_nodes.csv")
        print(f"  - {output_prefix}_edges.csv")

        # Print summary statistics
        print(f"\nNetwork Summary:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Edges: {len(edge_features)}")
        print(f"  Average degree: {2 * len(edge_features) / n_nodes:.2f}")
        print(f"\nInstitution Type Distribution:")
        print(node_features['institution_type'].value_counts())
        print(f"\nEdge Type Distribution:")
        print(edge_features['edge_type'].value_counts())
        print(f"\nTotal Network Exposure: ${edge_features['exposure'].sum()/1e12:.2f}T")

        return {
            'nodes': node_features,
            'edges': edge_features
        }


# Main execution
if __name__ == "__main__":
    # Create generator instance
    generator = SyntheticFinancialNetworkGenerator(seed=42)

    # Generate network with 500 institutions
    network_data = generator.generate_financial_network(
        n_nodes=500,
        output_prefix='synthetic_financial_network'
    )
