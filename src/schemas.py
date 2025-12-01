from pydantic import BaseModel


class CreditApplication(BaseModel):
    year: int
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: float
    rate_of_interest: float
    Interest_rate_spread: float
    Upfront_charges: float
    term: float
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    credit_type: str
    Credit_Score: float
    co_applicant_credit_type: str  # underscore – matches cleaned column name
    age: str
    submission_of_application: str
    LTV: float
    Region: str
    Security_Type: str
    dtir1: float