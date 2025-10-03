from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID
from config.postgreDb import Base

class UserSession(Base):
    __tablename__ = "user_session"
    id = Column(UUID(as_uuid=True), primary_key=True, unique=True, nullable=False)
    username = Column(String(255), nullable=False)
    origin = Column(String(255), nullable=True)
