# Data Skeleton
from pydantic import BaseModel

class CardioSchema(BaseModel):
    age: int 
    gender: int 
    height: int 
    weight: int
    ap_hi: int
    ap_lo: int
    cholesterol: int 
    gluc: int
    smoke: int
    alco: int
    active: int