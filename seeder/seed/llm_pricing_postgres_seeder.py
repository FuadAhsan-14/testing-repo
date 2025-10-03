import sys
import os
from decimal import Decimal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.postgreDb import get_sync_db
from app.models.LlmPricingModel import LlmPricingModel

PRICING_DATA = [
    {
        "model_name": "gemini-2.5-flash-lite",
        "backend_name": "warehouse",
        "input_cost_per_million_tokens": Decimal("0.10"),
        "output_cost_per_million_tokens": Decimal("0.40"),
    },
    {
        "model_name": "gemini-2.5-flash",
        "backend_name": "warehouse",
        "input_cost_per_million_tokens": Decimal("0.30"),
        "output_cost_per_million_tokens": Decimal("2.50"),
    },
]

def seed_pricing_models():
    """
    Seeds the llm_pricing_models table with the latest pricing data.
    It will update existing models or create new ones.
    """
    print("Starting to seed LLM pricing models...")
    db_session_generator = get_sync_db()
    db = next(db_session_generator)
    
    try:
        for data in PRICING_DATA:
            model_name = data["model_name"]
            
            existing_model = db.query(LlmPricingModel).filter(LlmPricingModel.model_name == model_name).first()
            
            if existing_model:
                print(f"Updating pricing for model: {model_name}")
                existing_model.backend_name = data["backend_name"]
                existing_model.input_cost_per_million_tokens = data["input_cost_per_million_tokens"]
                existing_model.output_cost_per_million_tokens = data["output_cost_per_million_tokens"]
            else:
                print(f"Creating new pricing for model: {model_name}")
                new_model = LlmPricingModel(**data)
                db.add(new_model)
        
        db.commit()
        print("\nSuccessfully seeded/updated LLM pricing models.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        db.rollback()
    finally:
        db.close()
        print("Database session closed.")

if __name__ == "__main__":
    seed_pricing_models()