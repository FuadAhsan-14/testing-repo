from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.LlmPricingModel import LlmPricingModel
from app.models.TokenUsageLog import TokenUsageLog
import logging
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def alog_token_usage(
    db: AsyncSession,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    agent_name: str = None
):
    """
    Asynchronously logs token usage and calculates the cost.
    """
    try:
        # Fetch pricing model
        stmt = select(LlmPricingModel).where(LlmPricingModel.model_name == model_name)
        result = await db.execute(stmt)
        pricing_model = result.scalars().first()

        if not pricing_model:
            logger.warning(f"Pricing model for '{model_name}' not found. Skipping cost calculation.")
            total_cost = Decimal("0.0")
        else:
            # Calculate cost using Decimal for precision
            input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing_model.input_cost_per_million_tokens
            output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing_model.output_cost_per_million_tokens
            total_cost = input_cost + output_cost

        # Create and save log entry
        log_entry = TokenUsageLog(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            agent_name=agent_name
        )
        db.add(log_entry)
        await db.commit()
        logger.info(f"Successfully logged token usage for agent '{agent_name}' with model '{model_name}'.")

    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to log token usage: {e}")

def log_token_usage(
    db: Session,
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    agent_name: str = None
):
    """
    Synchronously logs token usage and calculates the cost.
    """
    try:
        # Fetch pricing model
        pricing_model = db.query(LlmPricingModel).filter(LlmPricingModel.model_name == model_name).first()

        if not pricing_model:
            logger.warning(f"Pricing model for '{model_name}' not found. Skipping cost calculation.")
            total_cost = Decimal("0.0")
        else:
            # Calculate cost using Decimal for precision
            input_cost = (Decimal(input_tokens) / Decimal(1_000_000)) * pricing_model.input_cost_per_million_tokens
            output_cost = (Decimal(output_tokens) / Decimal(1_000_000)) * pricing_model.output_cost_per_million_tokens
            total_cost = input_cost + output_cost

        # Create and save log entry
        log_entry = TokenUsageLog(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            agent_name=agent_name
        )
        db.add(log_entry)
        db.commit()
        logger.info(f"Successfully logged token usage for agent '{agent_name}' with model '{model_name}'.")

    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log token usage: {e}")
    finally:
        db.close()