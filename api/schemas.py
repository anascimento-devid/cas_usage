from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str

    campaign: int
    pdays: int
    previous: int
    poutcome: str

    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")