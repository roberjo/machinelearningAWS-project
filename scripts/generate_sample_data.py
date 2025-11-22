import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import os

def generate_users(n=10000):
    """Generate user data"""
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n)],
        'age': np.random.randint(18, 70, n),
        'gender': np.random.choice(['M', 'F', 'Other'], n),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n),
        'join_date': [
            datetime.now() - timedelta(days=np.random.randint(1, 365))
            for _ in range(n)
        ]
    })

def generate_products(n=1000):
    """Generate product catalog"""
    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports', 'Toys']
    return pd.DataFrame({
        'product_id': [f'prod_{i}' for i in range(n)],
        'name': [f'Product {i}' for i in range(n)],
        'category': np.random.choice(categories, n),
        'price': np.random.uniform(10, 500, n).round(2),
        'brand': [f'Brand_{np.random.randint(1, 50)}' for _ in range(n)],
        'rating': np.random.uniform(3, 5, n).round(1),
        'num_reviews': np.random.randint(0, 1000, n)
    })

def generate_interactions(users, products, n=100000):
    """Generate user-product interactions"""
    user_ids = np.random.choice(users['user_id'], n)
    product_ids = np.random.choice(products['product_id'], n)
    
    return pd.DataFrame({
        'interaction_id': [f'int_{i}' for i in range(n)],
        'user_id': user_ids,
        'product_id': product_ids,
        'interaction_type': np.random.choice(
            ['view', 'add_to_cart', 'purchase'], 
            n, 
            p=[0.7, 0.2, 0.1]
        ),
        'timestamp': [
            datetime.now() - timedelta(hours=np.random.randint(1, 720))
            for _ in range(n)
        ],
        'rating': np.random.choice([None, 1, 2, 3, 4, 5], n, p=[0.7, 0.02, 0.03, 0.05, 0.1, 0.1])
    })

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic e-commerce data')
    parser.add_argument('--users', type=int, default=10000, help='Number of users')
    parser.add_argument('--products', type=int, default=1000, help='Number of products')
    parser.add_argument('--interactions', type=int, default=100000, help='Number of interactions')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.users} users...")
    users = generate_users(args.users)
    users.to_csv(os.path.join(args.output_dir, 'users.csv'), index=False)
    
    print(f"Generating {args.products} products...")
    products = generate_products(args.products)
    products.to_csv(os.path.join(args.output_dir, 'products.csv'), index=False)
    
    print(f"Generating {args.interactions} interactions...")
    interactions = generate_interactions(users, products, args.interactions)
    interactions.to_csv(os.path.join(args.output_dir, 'interactions.csv'), index=False)
    
    print(f"Data generation complete. Saved to {args.output_dir}")

if __name__ == '__main__':
    main()
