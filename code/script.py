from ContaminationExtractor import Paper_Contamination 

print("Script is running...")


save_fullpath = "keyword_1000000.csv"
folder_path = "paper_folder"
historical_data_fullpath = "Historical_data.xlsx"
DB_fullpath = "Model_DB.xlsx"
errorfile_fullpath = "ErrorFileList.txt"



Paper_Contamination(save_fullpath, folder_path, historical_data_fullpath, DB_fullpath, errorfile_fullpath)

