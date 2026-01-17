from pydantic import BaseModel, Field
from typing import List, Optional


class RetrivalSchema(BaseModel):
    score : int = Field(..., description="Score of the retrieval result")
    description : str = Field(..., description="Short Summary  retrieval result")