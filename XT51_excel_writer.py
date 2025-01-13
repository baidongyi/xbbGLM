
from datetime import datetime
import pandas as pd

def insert_data_to_sheet(excel_file_path:str,question:str,answer:str):
    data1 = pd.DataFrame(pd.read_excel(excel_file_path))

    today = datetime.now()
    my_time = today.strftime("%Y-%m-%d %H:%M:%S")
    data2 = pd.DataFrame({'T':[my_time], 'Q': [question], 'A': [answer]})

    new_data = pd.concat([data1,data2])

    with pd.ExcelWriter(excel_file_path) as writer:
        new_data.to_excel(writer, sheet_name="sheet1", index=False)


# if __name__ == '__main__':
#     excel_path = "history_lib//baido.xlsx"
#     insert_data_to_sheet(excel_path,'q2','a3')