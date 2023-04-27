import sys 
from src.logger import logging


# get the error details
def error_message_detail(error,error_detail:sys):
    """
     Get the error details:
        error_message and Locate the file and line which caused the error
    """

    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_no = exc_tb.tb_frame.f_lineno
    error_message = f"Caught {file_name} Error in file {line_no} at line {str(error=error)}"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error=error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message