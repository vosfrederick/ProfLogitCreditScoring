from typing import List, NamedTuple

data_dir = '../data'

class DataSet(NamedTuple):
    filename: str
    goal_column: str
    category_columns: List[str]
    numeric_columns: List[str]


    def get_vars(self):
        return self.category_columns + self.numeric_columns

# Choose which variables are categorical, numerical and target for each dataset
german_dataset = DataSet(
    filename=f'{data_dir}/german_credit_data.csv',
    goal_column= 'risk',
    category_columns = ['sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'purpose'],
    numeric_columns = ['age', 'credit_amount', 'duration']
)

credit_risk_dataset = DataSet(
    filename=f'{data_dir}/credit_risk_dataset.csv',
    goal_column= 'loan_status',
    category_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'],
    numeric_columns = ['person_age', 'person_income', 'person_emp_length','loan_amnt','loan_int_rate',
                       'loan_percent_income', 'cb_person_cred_hist_length']
)

hmeq_dataset = DataSet(
    filename=f'{data_dir}/hmeq.csv',
    goal_column= 'BAD',
    category_columns = ['REASON', 'JOB'],
    numeric_columns = ['LOAN', 'MORTDUE', 'VALUE','YOJ','DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO']
)

default_dataset = DataSet(
    filename=f'{data_dir}/default_credit_card_clients.csv',
    goal_column= 'default_payment_next_month',
    category_columns = ['SEX', 'EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],
    numeric_columns = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                       'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
)