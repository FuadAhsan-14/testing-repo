import uuid
from sqlalchemy import Column, String, Integer, Numeric, func, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from config.postgreDb import Base

class TokenUsageLog(Base):
    __tablename__ = "token_usage_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_cost = Column(Numeric(10, 6), nullable=False)
    agent_name = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)