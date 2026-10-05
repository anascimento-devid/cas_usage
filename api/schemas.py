from typing import Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    default: Literal["no", "yes", "unknown"]
    housing: Literal["no", "yes", "unknown"]
    loan: Literal["no", "yes", "unknown"]

    contact: Literal["cellular", "telephone"]

    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    day_of_week: Literal[
        "mon", "tue", "wed", "thu", "fri"
    ]

    campaign: int = Field(ge=1)
    pdays: int = Field(ge=0)
    previous: int = Field(ge=0)

    poutcome: Literal[
        "nonexistent",
        "failure",
        "success"
    ]

    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")

    euribor3m: float

    nr_employed: float = Field(alias="nr.employed")