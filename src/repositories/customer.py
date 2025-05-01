from typing import List
from src.ai.schemas import CustomerProfile

class FakeCustomerRepository:
    def get_profile(self, customer_id: str) -> CustomerProfile:
        return CustomerProfile(
            id=customer_id,
            name="Jane Doe",
            score=87,
            last_purchase="2025-04-20",
            email="jane.doe@example.com",
            phone="+1-555-1234",
            address="123 Main St, Anytown, USA",
            purchase_history=[
                {"date": "2025-04-20", "item": "Widget A", "amount": 29.99},
                {"date": "2025-03-15", "item": "Widget B", "amount": 49.99},
            ]
        )

    def get_feedback(self, customer_id: str) -> List[dict]:
        return [
            {"date": "2025-04-21", "feedback": "Great service!"},
            {"date": "2025-03-16", "feedback": "Product quality could be better."}
        ]

    def get_support_tickets(self, customer_id: str) -> List[dict]:
        return [
            {"ticket_id": "T123", "issue": "Late delivery", "status": "Resolved"},
            {"ticket_id": "T124", "issue": "Wrong item received", "status": "Open"}
        ]
