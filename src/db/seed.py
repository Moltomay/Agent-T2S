from datetime import datetime, timedelta
import random

from src.db.connection import engine, Base, SessionLocal
from src.db.models import Customer, Product, Order, OrderItem


def seed_database():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()

    try:
        if session.query(Customer).count() > 0:
            return

        customers = [
            Customer(name="Alice Martin", email="alice@example.com", country="France", loyalty_tier="gold"),
            Customer(name="Bob Dupont", email="bob@example.com", country="France", loyalty_tier="silver"),
            Customer(name="Charlie Lee", email="charlie@example.com", country="USA", loyalty_tier="bronze"),
            Customer(name="Diana Rossi", email="diana@example.com", country="Italy", loyalty_tier="gold"),
            Customer(name="Eva Kim", email="eva@example.com", country="South Korea", loyalty_tier="silver"),
            Customer(name="Frank Chen", email="frank@example.com", country="China", loyalty_tier="bronze"),
        ]
        session.add_all(customers)
        session.flush()

        products = [
            Product(name="Wireless Mouse", category="Electronics", price=29.99, stock_quantity=150),
            Product(name="Mechanical Keyboard", category="Electronics", price=89.99, stock_quantity=80),
            Product(name="USB-C Hub", category="Electronics", price=45.50, stock_quantity=200),
            Product(name="Running Shoes", category="Sports", price=120.00, stock_quantity=60),
            Product(name="Yoga Mat", category="Sports", price=25.00, stock_quantity=300),
            Product(name="Coffee Mug", category="Home", price=14.99, stock_quantity=500),
            Product(name="Desk Lamp", category="Home", price=39.99, stock_quantity=120),
            Product(name="Notebook", category="Stationery", price=5.99, stock_quantity=1000),
            Product(name="Pen Set", category="Stationery", price=12.50, stock_quantity=400),
            Product(name="Backpack", category="Accessories", price=65.00, stock_quantity=90),
        ]
        session.add_all(products)
        session.flush()

        statuses = ["completed", "pending", "shipped", "cancelled"]
        orders_data = [
            (1, 0, "completed"),
            (1, 15, "shipped"),
            (2, 5, "completed"),
            (2, 20, "pending"),
            (3, 10, "completed"),
            (3, 25, "cancelled"),
            (4, 3, "completed"),
            (4, 12, "shipped"),
            (5, 7, "completed"),
            (5, 18, "pending"),
            (6, 1, "completed"),
            (6, 22, "shipped"),
        ]

        orders = []
        for i, (cust_idx, days_ago, status) in enumerate(orders_data):
            order_date = datetime.now() - timedelta(days=days_ago)
            orders.append(
                Order(
                    customer_id=customers[cust_idx - 1].id,
                    order_date=order_date,
                    status=status,
                )
            )
        session.add_all(orders)
        session.flush()

        item_templates = [
            (0, 0, 1, 29.99),
            (0, 1, 1, 89.99),
            (1, 2, 2, 45.50),
            (1, 8, 3, 12.50),
            (2, 3, 1, 120.00),
            (2, 4, 1, 25.00),
            (3, 5, 4, 14.99),
            (3, 6, 1, 39.99),
            (4, 7, 5, 5.99),
            (4, 9, 1, 65.00),
            (5, 0, 2, 29.99),
            (5, 2, 1, 45.50),
            (6, 3, 1, 120.00),
            (6, 8, 2, 12.50),
            (7, 1, 1, 89.99),
            (7, 9, 1, 65.00),
            (8, 4, 2, 25.00),
            (8, 5, 3, 14.99),
            (9, 6, 1, 39.99),
            (9, 7, 2, 5.99),
            (10, 2, 3, 45.50),
            (10, 0, 1, 29.99),
            (11, 9, 1, 65.00),
            (11, 3, 1, 120.00),
        ]

        for o_idx, p_idx, qty, price in item_templates:
            session.add(
                OrderItem(
                    order_id=orders[o_idx].id,
                    product_id=products[p_idx].id,
                    quantity=qty,
                    unit_price=price,
                )
            )

        session.flush()

        for order in orders:
            items = (
                session.query(OrderItem)
                .filter(OrderItem.order_id == order.id)
                .all()
            )
            order.total_amount = sum(
                item.unit_price * item.quantity for item in items
            )

        session.commit()

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
    print("Database seeded successfully!")
