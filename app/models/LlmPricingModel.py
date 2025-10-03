from sqlalchemy import Column, String, Numeric, func, TIMESTAMP
from config.postgreDb import Base

class LlmPricingModel(Base):
    __tablename__ = "llm_pricing_models"
    model_name = Column(String(255), primary_key=True, unique=True, nullable=False)
    backend_name = Column(String(255), nullable=False)
    input_cost_per_million_tokens = Column(Numeric(10, 4), nullable=False)
    output_cost_per_million_tokens = Column(Numeric(10, 4), nullable=False)
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.current_timestamp())