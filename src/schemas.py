from pydantic import BaseModel, Field


class CreditApplication(BaseModel):
    year: int = Field(..., ge=2000, le=2100)
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: float = Field(..., ge=0)
    term: float = Field(..., ge=0)
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float = Field(..., ge=0)
    credit_type: str
    Credit_Score: float = Field(..., ge=0, le=900)
    co_applicant_credit_type: str  # underscore – matches cleaned column name
    age: str
    submission_of_application: str
    Region: str
    Security_Type: str
    dtir1: float = Field(..., ge=0, le=100)
